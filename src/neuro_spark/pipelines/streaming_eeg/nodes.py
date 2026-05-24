"""Pipeline Spark Structured Streaming Distribué pour l'EEG.

L'inférence Scikit-Learn est encapsulée dans une fonction Pandas UDF (applyInPandas).
Cela permet de distribuer la prédiction sur l'ensemble du cluster Spark
tout en conservant la latence < 10ms de Scikit-Learn.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType, DoubleType, LongType, StringType, StructField, StructType,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler

# Shared constants imported from core — single source of truth
from neuro_spark.core import BANDS, FS as EEG_FS, MOTOR_CHANNELS

log = logging.getLogger(__name__)

# Feature order must match what train_sklearn_model produces
FEATURE_ORDER = (
    [f"{ch}_{band}" for ch in MOTOR_CHANNELS for band in BANDS]
    + [f"diff_{band}" for band in BANDS]
)

# Metadata columns excluded from EEG channel detection
_META_COLS = {"subject_id", "run_id", "time", "task_label", "epoch_id", "epoch_label", "event_time", "run_seq"}

# ── Schemas ──────────────────────────────────────────────────────────────────

_PRED_STRUCT = StructType([
    StructField("event_time",  StringType(), True),
    StructField("epoch_id",    LongType(),   True),
    StructField("subject_id",  StringType(), True),
    StructField("run_id",      StringType(), True),
    StructField("run_seq",     LongType(),   True),
    StructField("true_label",  StringType(), True),
    StructField("pred_label",  StringType(), True),
    StructField("proba_T0",    DoubleType(), True),
    StructField("proba_T1",    DoubleType(), True),
    StructField("proba_T2",    DoubleType(), True),
    StructField("correct",     LongType(),   True),
])

_TOPO_STRUCT = StructType([
    StructField("event_time",  StringType(), True),
    StructField("epoch_id",    LongType(),   True),
    StructField("channel",     StringType(), True),
    StructField("alpha_power", DoubleType(), True),
    StructField("true_label",  StringType(), True),
    StructField("pred_label",  StringType(), True),
])

_UDF_SCHEMA = StructType([
    StructField("prediction", _PRED_STRUCT,          True),
    StructField("topomaps",   ArrayType(_TOPO_STRUCT), True),
])

_META_STRING_COLS = {"subject_id", "run_id", "task_label", "epoch_label", "event_time"}
_META_LONG_COLS   = {"epoch_id", "run_seq"}

def _input_schema(channels: list[str]) -> StructType:
    """Build the Parquet input schema from a list of EEG channel names."""
    meta_fields = [
        StructField("subject_id",  StringType(), True),
        StructField("run_id",      StringType(), True),
        StructField("time",        DoubleType(), True),
        StructField("task_label",  StringType(), True),
        StructField("epoch_id",    LongType(),   True),
        StructField("epoch_label", StringType(), True),
        StructField("event_time",  StringType(), True),
        StructField("run_seq",     LongType(),   True),
    ]
    channel_fields = [StructField(ch, DoubleType(), True) for ch in channels]
    return StructType(meta_fields + channel_fields)


# ── Feature helpers ───────────────────────────────────────────────────────────

def _band_power(signal: np.ndarray, low: float, high: float) -> float:
    """FFT power in [low, high) Hz. Delegates constants from core."""
    n = len(signal)
    if n < 4:
        return 0.0
    freqs = np.fft.rfftfreq(n, 1.0 / EEG_FS)
    power = np.abs(np.fft.rfft(signal)) ** 2
    mask = (freqs >= low) & (freqs < high)
    return float(np.mean(power[mask])) if mask.any() else 0.0


def compute_features(epoch_df: pd.DataFrame) -> dict:
    """Compute FFT band power features for one epoch using shared MOTOR_CHANNELS and BANDS."""
    feats: dict[str, float] = {}
    for ch in MOTOR_CHANNELS:
        sig = epoch_df[ch].values
        for band, (lo, hi) in BANDS.items():
            feats[f"{ch}_{band}"] = _band_power(sig, lo, hi)
    for band in BANDS:
        feats[f"diff_{band}"] = feats[f"C3_{band}"] - feats[f"C4_{band}"]
    return feats


def compute_alpha_topomap(epoch_df: pd.DataFrame) -> dict[str, float]:
    """Alpha power (8–13 Hz) for every EEG channel in the epoch."""
    eeg_cols = [c for c in epoch_df.columns if c not in _META_COLS]
    lo, hi = BANDS["alpha"]
    return {ch: _band_power(epoch_df[ch].values, lo, hi) for ch in eeg_cols}


# ── Sklearn model ─────────────────────────────────────────────────────────────

def _safe_log10(X):
    return np.log10(np.clip(X, 1e-12, None))


def train_sklearn_model(parquet_dir: str, model_cache: str) -> tuple:
    cache = Path(model_cache)
    if cache.exists():
        with cache.open("rb") as f:
            payload = pickle.load(f)
        log.info("Modèle sklearn chargé depuis le cache (%s)", cache)
        return payload["model"], payload["encoder"]

    log.info("Entraînement sklearn RandomForest (Cache manquant)...")
    parquets = sorted(Path(parquet_dir).glob("*.parquet"))
    samples_per_epoch = EEG_FS * 2
    X, y = [], []
    for p in parquets[:60]:
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

    enc = LabelEncoder().fit(y)
    clf = SkPipeline([
        ("log",    FunctionTransformer(_safe_log10, validate=False)),
        ("scaler", StandardScaler()),
        ("rf",     RandomForestClassifier(
            n_estimators=300, max_depth=18,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )),
    ])
    clf.fit(np.asarray(X), enc.transform(np.asarray(y)))

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump({"model": clf, "encoder": enc}, f)
    return clf, enc


# ── Spark helpers ─────────────────────────────────────────────────────────────

def _build_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.streaming.schemaInference", "false")
        .getOrCreate()
    )


def _peek_channels(input_dir: str) -> list[str]:
    """Return EEG channel names from the first available Parquet file."""
    files = sorted(Path(input_dir).glob("*.parquet"))
    if not files:
        files = sorted(Path("data/parquet").glob("*.parquet"))
    if not files:
        log.warning("Aucun fichier Parquet trouvé — utilisation de la structure par défaut 64 canaux")
        return [
            "AF3", "AF4", "AF7", "AF8", "AFz",
            "C1", "C2", "C3", "C4", "C5", "C6",
            "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz", "Cz",
            "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
            "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCz",
            "FT7", "FT8", "Fz", "FP1", "FP2", "FPz",
            "O1", "O2", "Oz",
            "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8",
            "PO3", "PO4", "PO7", "PO8", "POz", "Pz",
            "T7", "T8", "TP7", "TP8",
        ]
    df = pd.read_parquet(files[0], columns=None)
    return [c for c in df.columns if c not in _META_COLS]


# ── Main streaming entry point ────────────────────────────────────────────────

def run_streaming_eeg(
    stream_input: str,
    stream_output: str,
    checkpoint: str,
    parquet_dir: str,
    model_cache: str,
    timeout_seconds: int,
) -> dict:
    Path(stream_input).mkdir(parents=True, exist_ok=True)
    Path(stream_output).mkdir(parents=True, exist_ok=True)

    spark = _build_spark("neuro-spark-streaming-eeg")
    spark.sparkContext.setLogLevel("WARN")

    clf, enc = train_sklearn_model(parquet_dir, model_cache)
    bc_clf = spark.sparkContext.broadcast(clf)
    bc_enc = spark.sparkContext.broadcast(enc)

    channels = _peek_channels(stream_input)
    schema = _input_schema(channels)
    events = (
        spark.readStream.format("parquet")
        .schema(schema)
        .option("maxFilesPerTrigger", 1)
        .load(stream_input)
    )

    pred_path = Path(stream_output) / "predictions"
    topo_path = Path(stream_output) / "topomap"
    pred_path.mkdir(parents=True, exist_ok=True)
    topo_path.mkdir(parents=True, exist_ok=True)

    def process_epoch_distributed(pdf: pd.DataFrame) -> pd.DataFrame:
        if pdf.empty:
            return pd.DataFrame()
        model, encoder = bc_clf.value, bc_enc.value
        eid        = pdf["epoch_id"].iloc[0]
        event_time = pdf["event_time"].iloc[0]
        true_label = pdf["epoch_label"].iloc[0]
        subject    = pdf["subject_id"].iloc[0]
        run_id     = pdf["run_id"].iloc[0] if "run_id" in pdf.columns else "?"
        run_seq_val = (
            int(pdf["run_seq"].iloc[0])
            if "run_seq" in pdf.columns and not pd.isna(pdf["run_seq"].iloc[0])
            else -1
        )

        try:
            feats = compute_features(pdf)
            x = np.asarray([[feats[k] for k in FEATURE_ORDER]])
            pred_idx   = int(model.predict(x)[0])
            pred_label = encoder.inverse_transform([pred_idx])[0]
            proba_dict = {
                encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(model.predict_proba(x)[0])
            }
        except Exception:
            pred_label, proba_dict = "?", {}

        pred_dict = {
            "event_time": str(event_time),
            "epoch_id":   int(eid),
            "subject_id": str(subject),
            "run_id":     str(run_id),
            "run_seq":    int(run_seq_val),
            "true_label": str(true_label),
            "pred_label": str(pred_label),
            "proba_T0":   float(proba_dict.get("T0", 0.0)),
            "proba_T1":   float(proba_dict.get("T1", 0.0)),
            "proba_T2":   float(proba_dict.get("T2", 0.0)),
            "correct":    int(pred_label == true_label),
        }
        topo_list = [
            {
                "event_time":  str(event_time),
                "epoch_id":    int(eid),
                "channel":     str(ch),
                "alpha_power": float(pw),
                "true_label":  str(true_label),
                "pred_label":  str(pred_label),
            }
            for ch, pw in compute_alpha_topomap(pdf).items()
        ]
        return pd.DataFrame([{"prediction": pred_dict, "topomaps": topo_list}])

    def process_batch(batch_df, batch_id: int) -> None:
        if batch_df.rdd.isEmpty():
            return
        processed_df = (
            batch_df.groupBy("epoch_id")
            .applyInPandas(process_epoch_distributed, schema=_UDF_SCHEMA)
            .cache()
        )
        processed_df.select("prediction.*").write.mode("append").parquet(str(pred_path))
        processed_df.select(F.explode("topomaps").alias("topo")).select("topo.*").write.mode("append").parquet(str(topo_path))
        log.info("Batch %d traité de manière distribuée via applyInPandas.", batch_id)
        processed_df.unpersist()

    query = (
        events.writeStream
        .outputMode("update")
        .foreachBatch(process_batch)
        .option("checkpointLocation", checkpoint)
        .trigger(processingTime="1 seconds")
        .queryName("eeg_stream")
        .start()
    )
    try:
        query.awaitTermination(timeout=timeout_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        for q in spark.streams.active:
            q.stop()

    return {"timeout_seconds": timeout_seconds, "channels": len(channels)}
