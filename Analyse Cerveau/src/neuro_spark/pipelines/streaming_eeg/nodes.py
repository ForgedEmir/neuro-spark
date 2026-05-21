"""Pipeline Spark Structured Streaming pour l'EEG (BCI + ERD/ERS).

À chaque epoch (2 s = 320 samples @ 160 Hz) on calcule :
  1. **Features FFT** sur C3/Cz/C4 (theta/alpha/beta/gamma + lateralization)
     → prédiction RandomForest sklearn (T0=repos, T1=hand, T2=imagined hand)
  2. **Puissance alpha** (8-13 Hz) sur les 64 électrodes → topographie cérébrale

Pourquoi sklearn et pas le CrossValidatorModel Spark ? Pour la latence : prédire
1 epoch via Spark coûte ~500 ms d'overhead JVM. sklearn fait ça en <10 ms.
Le sklearn RF est ré-entraîné au démarrage sur les Parquet existants — c'est
rapide (~5 s) et fournit un modèle équivalent au modèle batch.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType, LongType, StringType, StructField, StructType,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler

log = logging.getLogger(__name__)

EEG_FS = 160
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 80)}
MOTOR_CHANNELS = ["C3..", "Cz..", "C4.."]
LABEL_NAMES = {"T0": "Repos", "T1": "Mouvement réel", "T2": "Mouvement imaginé"}


# ── Calcul features FFT ──────────────────────────────────────────────────────
def band_power(signal: np.ndarray, fs: int, low: float, high: float) -> float:
    n = len(signal)
    if n < 4:
        return 0.0
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    power = np.abs(np.fft.rfft(signal)) ** 2
    mask = (freqs >= low) & (freqs < high)
    return float(np.mean(power[mask])) if mask.any() else 0.0


def compute_features(epoch_df: pd.DataFrame) -> dict:
    """Features sur C3/Cz/C4 : 3 channels × 4 bandes + 4 latéralisations = 16 features."""
    feats: dict[str, float] = {}
    for ch in MOTOR_CHANNELS:
        ch_clean = ch.replace(".", "")
        sig = epoch_df[ch].values
        for band, (lo, hi) in BANDS.items():
            feats[f"{ch_clean}_{band}"] = band_power(sig, EEG_FS, lo, hi)
    # latéralisation = différence hémisphérique
    for band in BANDS:
        feats[f"diff_{band}"] = feats[f"C3_{band}"] - feats[f"C4_{band}"]
    return feats


FEATURE_ORDER = (
    [f"{ch.replace('.', '')}_{band}" for ch in MOTOR_CHANNELS for band in BANDS]
    + [f"diff_{band}" for band in BANDS]
)


def _safe_log10(X):
    """Top-level (picklable) : log10 avec clip pour le pipeline sklearn."""
    return np.log10(np.clip(X, 1e-12, None))


def compute_alpha_topomap(epoch_df: pd.DataFrame) -> dict[str, float]:
    """Puissance alpha (8-13 Hz) sur chaque électrode EEG → dict {channel: power}."""
    eeg_cols = [
        c for c in epoch_df.columns
        if c not in {"subject_id", "run_id", "time", "task_label",
                     "epoch_id", "epoch_label", "event_time"}
    ]
    topo = {}
    for ch in eeg_cols:
        topo[ch.replace(".", "")] = band_power(epoch_df[ch].values, EEG_FS, 8, 13)
    return topo


# ── Entraînement rapide d'un sklearn RF sur le corpus batch ──────────────────
def train_sklearn_model(parquet_dir: str, model_cache: str) -> tuple:
    """Entraîne ou recharge un RandomForest sklearn pour la prédiction live."""
    cache = Path(model_cache)
    if cache.exists():
        with cache.open("rb") as f:
            payload = pickle.load(f)
        log.info("Modèle sklearn chargé depuis le cache (%s)", cache)
        return payload["model"], payload["encoder"]

    log.info("Pas de cache → entraînement sklearn RandomForest sur les Parquet existants")
    parquets = sorted(Path(parquet_dir).glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"Aucun parquet dans {parquet_dir}. Lance `make ingest` d'abord."
        )

    X, y = [], []
    samples_per_epoch = EEG_FS * 2
    # On échantillonne 60 fichiers pour avoir assez de variance entre sujets
    sample_files = parquets[:60]
    log.info("Computing features sur %d fichiers...", len(sample_files))
    for p in sample_files:
        df = pd.read_parquet(p)
        n_epochs = len(df) // samples_per_epoch
        for i in range(n_epochs):
            chunk = df.iloc[i * samples_per_epoch:(i + 1) * samples_per_epoch]
            try:
                feats = compute_features(chunk)
                X.append([feats[k] for k in FEATURE_ORDER])
                y.append(chunk["task_label"].mode().iloc[0])
            except Exception:
                continue

    X = np.asarray(X)
    y = np.asarray(y)
    enc = LabelEncoder().fit(y)
    y_enc = enc.transform(y)

    log.info("Training sklearn RF on %d epochs, %d features", len(X), X.shape[1])
    # Les band powers FFT couvrent 6+ ordres de magnitude → log10 + StandardScaler.
    # Sinon le RF ne distingue pas les classes proches.
    clf = SkPipeline([
        ("log", FunctionTransformer(_safe_log10, validate=False)),
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=18, min_samples_leaf=2,
            random_state=42, class_weight="balanced", n_jobs=-1,
        )),
    ])
    clf.fit(X, y_enc)
    log.info("Train accuracy: %.3f", clf.score(X, y_enc))

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump({"model": clf, "encoder": enc}, f)
    log.info("Modèle sklearn sauvegardé dans %s", cache)
    return clf, enc


# ── Pipeline streaming Spark ─────────────────────────────────────────────────
def _build_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.streaming.schemaInference", "false")
        .getOrCreate()
    )


def _build_schema(channels: list[str]) -> StructType:
    """Schéma explicite : 64 EEG channels (DoubleType) + metadata."""
    fields = [
        StructField("subject_id", StringType(), True),
        StructField("run_id", StringType(), True),
        StructField("time", DoubleType(), True),
        StructField("task_label", StringType(), True),
        StructField("epoch_id", LongType(), True),
        StructField("epoch_label", StringType(), True),
        StructField("event_time", StringType(), True),
        StructField("run_seq", LongType(), True),     # index séquentiel du run (pour fatigue test)
    ]
    for ch in channels:
        fields.append(StructField(ch, DoubleType(), True))
    return StructType(fields)


def _peek_channels(input_dir: str) -> list[str]:
    """Lit un fichier existant pour découvrir les colonnes EEG."""
    sample = next(Path(input_dir).glob("*.parquet"), None)
    if sample is None:
        # fallback : on charge depuis un parquet batch
        sample = next(Path("data/parquet").glob("*.parquet"))
    df = pd.read_parquet(sample, columns=None)
    return [c for c in df.columns
            if c not in {"subject_id", "run_id", "time", "task_label",
                         "epoch_id", "epoch_label", "event_time"}]


def run_streaming_eeg(
    stream_input: str,
    stream_output: str,
    checkpoint: str,
    parquet_dir: str,
    model_cache: str,
    timeout_seconds: int,
) -> dict:
    """Lance le streaming EEG (BCI prediction + topomap alpha) jusqu'à timeout."""
    Path(stream_input).mkdir(parents=True, exist_ok=True)
    Path(stream_output).mkdir(parents=True, exist_ok=True)

    # 1. Modèle sklearn (entraîné une fois, puis cache)
    clf, enc = train_sklearn_model(parquet_dir, model_cache)

    # 2. Découverte des canaux EEG (le schéma doit les déclarer tous)
    channels = _peek_channels(stream_input)
    log.info("%d canaux EEG détectés", len(channels))
    schema = _build_schema(channels)

    # 3. Spark session + readStream
    spark = _build_spark("neuro-spark-streaming-eeg")
    spark.sparkContext.setLogLevel("WARN")

    events = (
        spark.readStream
        .format("parquet")
        .schema(schema)
        .option("maxFilesPerTrigger", 1)
        .option("latestFirst", "false")
        .load(stream_input)
    )

    # 4. Sinks : prediction + topomap
    pred_path = Path(stream_output) / "predictions"
    topo_path = Path(stream_output) / "topomap"
    pred_path.mkdir(parents=True, exist_ok=True)
    topo_path.mkdir(parents=True, exist_ok=True)

    def process_batch(batch_df, batch_id: int) -> None:
        if batch_df.rdd.isEmpty():
            return
        pdf = batch_df.toPandas()
        if pdf.empty:
            return

        predictions = []
        topomaps = []
        for eid, epoch in pdf.groupby("epoch_id"):
            event_time = epoch["event_time"].iloc[0]
            true_label = epoch["epoch_label"].iloc[0]
            subject = epoch["subject_id"].iloc[0]

            # Prédiction BCI
            try:
                feats = compute_features(epoch)
                x = np.asarray([[feats[k] for k in FEATURE_ORDER]])
                pred_idx = int(clf.predict(x)[0])
                pred_label = enc.inverse_transform([pred_idx])[0]
                proba = clf.predict_proba(x)[0]
                proba_dict = {
                    enc.inverse_transform([i])[0]: float(p)
                    for i, p in enumerate(proba)
                }
            except Exception as exc:
                log.warning("epoch %s: prediction failed (%s)", eid, exc)
                pred_label, proba_dict = "?", {}

            run_id = epoch["run_id"].iloc[0] if "run_id" in epoch.columns else "?"
            run_seq_val = epoch["run_seq"].iloc[0] if "run_seq" in epoch.columns else -1
            predictions.append({
                "event_time": event_time, "epoch_id": int(eid),
                "subject_id": subject, "run_id": str(run_id),
                "run_seq": int(run_seq_val) if run_seq_val is not None else -1,
                "true_label": true_label, "pred_label": pred_label,
                "proba_T0": proba_dict.get("T0", 0.0),
                "proba_T1": proba_dict.get("T1", 0.0),
                "proba_T2": proba_dict.get("T2", 0.0),
                "correct": int(pred_label == true_label),
            })

            # Topomap alpha
            topo = compute_alpha_topomap(epoch)
            for ch, power in topo.items():
                topomaps.append({
                    "event_time": event_time, "epoch_id": int(eid),
                    "channel": ch, "alpha_power": power,
                    "true_label": true_label, "pred_label": pred_label,
                })

        # Write Parquet (un fichier par batch, append-only)
        if predictions:
            pd.DataFrame(predictions).to_parquet(
                pred_path / f"batch_{batch_id:06d}.parquet", index=False,
            )
        if topomaps:
            pd.DataFrame(topomaps).to_parquet(
                topo_path / f"batch_{batch_id:06d}.parquet", index=False,
            )

        log.info("EEG batch %d: %d predictions, %d topomap rows",
                 batch_id, len(predictions), len(topomaps))

    query = (
        events.writeStream
        .outputMode("update")
        .foreachBatch(process_batch)
        .option("checkpointLocation", checkpoint)
        .trigger(processingTime="1 seconds")
        .queryName("eeg_stream")
        .start()
    )

    log.info("EEG streaming démarré (timeout=%ds)", timeout_seconds)
    try:
        query.awaitTermination(timeout=timeout_seconds)
    except KeyboardInterrupt:
        log.info("Ctrl+C — arrêt")
    finally:
        for q in spark.streams.active:
            q.stop()

    return {"timeout_seconds": timeout_seconds, "channels": len(channels)}
