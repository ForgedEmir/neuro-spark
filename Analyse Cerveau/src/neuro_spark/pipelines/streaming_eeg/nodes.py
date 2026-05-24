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
    DoubleType, LongType, StringType, StructField, StructType, ArrayType
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler

log = logging.getLogger(__name__)

EEG_FS = 160
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 80)}
MOTOR_CHANNELS = ["C3..", "Cz..", "C4.."]

def band_power(signal: np.ndarray, fs: int, low: float, high: float) -> float:
    n = len(signal)
    if n < 4: return 0.0
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    power = np.abs(np.fft.rfft(signal)) ** 2
    mask = (freqs >= low) & (freqs < high)
    return float(np.mean(power[mask])) if mask.any() else 0.0

def compute_features(epoch_df: pd.DataFrame) -> dict:
    feats: dict[str, float] = {}
    for ch in MOTOR_CHANNELS:
        ch_clean = ch.replace(".", "")
        sig = epoch_df[ch].values
        for band, (lo, hi) in BANDS.items():
            feats[f"{ch_clean}_{band}"] = band_power(sig, EEG_FS, lo, hi)
    for band in BANDS:
        feats[f"diff_{band}"] = feats[f"C3_{band}"] - feats[f"C4_{band}"]
    return feats

FEATURE_ORDER = (
    [f"{ch.replace('.', '')}_{band}" for ch in MOTOR_CHANNELS for band in BANDS]
    + [f"diff_{band}" for band in BANDS]
)

def _safe_log10(X):
    return np.log10(np.clip(X, 1e-12, None))

def compute_alpha_topomap(epoch_df: pd.DataFrame) -> dict[str, float]:
    eeg_cols = [
        c for c in epoch_df.columns
        if c not in {"subject_id", "run_id", "time", "task_label",
                     "epoch_id", "epoch_label", "event_time", "run_seq"}
    ]
    return {ch.replace(".", ""): band_power(epoch_df[ch].values, EEG_FS, 8, 13) for ch in eeg_cols}

def train_sklearn_model(parquet_dir: str, model_cache: str) -> tuple:
    cache = Path(model_cache)
    if cache.exists():
        with cache.open("rb") as f:
            payload = pickle.load(f)
        log.info("Modèle sklearn chargé depuis le cache (%s)", cache)
        return payload["model"], payload["encoder"]

    log.info("Entraînement sklearn RandomForest (Cache manquant)...")
    parquets = sorted(Path(parquet_dir).glob("*.parquet"))
    X, y = [], []
    samples_per_epoch = EEG_FS * 2
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
        ("log", FunctionTransformer(_safe_log10, validate=False)),
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=18, class_weight="balanced", n_jobs=-1, random_state=42)),
    ])
    clf.fit(np.asarray(X), enc.transform(np.asarray(y)))

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f: pickle.dump({"model": clf, "encoder": enc}, f)
    return clf, enc

def _build_spark(app_name: str) -> SparkSession:
    return (SparkSession.builder.appName(app_name)
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.sql.streaming.schemaInference", "false")
            .getOrCreate())

def _peek_channels(input_dir: str) -> list[str]:
    sample = next(Path(input_dir).glob("*.parquet"), None) or next(Path("data/parquet").glob("*.parquet"))
    df = pd.read_parquet(sample, columns=None)
    return [c for c in df.columns if c not in {"subject_id", "run_id", "time", "task_label", "epoch_id", "epoch_label", "event_time", "run_seq"}]

def run_streaming_eeg(stream_input: str, stream_output: str, checkpoint: str, parquet_dir: str, model_cache: str, timeout_seconds: int) -> dict:
    Path(stream_input).mkdir(parents=True, exist_ok=True)
    Path(stream_output).mkdir(parents=True, exist_ok=True)

    spark = _build_spark("neuro-spark-streaming-eeg")
    spark.sparkContext.setLogLevel("WARN")

    clf, enc = train_sklearn_model(parquet_dir, model_cache)
    bc_clf = spark.sparkContext.broadcast(clf)
    bc_enc = spark.sparkContext.broadcast(enc)

    channels = _peek_channels(stream_input)
    fields = [StructField(c, StringType() if c in ["subject_id", "run_id", "task_label", "epoch_label", "event_time"] else DoubleType() if c == "time" else LongType(), True) for c in ["subject_id", "run_id", "time", "task_label", "epoch_id", "epoch_label", "event_time", "run_seq"]] + [StructField(ch, DoubleType(), True) for ch in channels]

    events = spark.readStream.format("parquet").schema(StructType(fields)).option("maxFilesPerTrigger", 1).load(stream_input)

    pred_path, topo_path = Path(stream_output) / "predictions", Path(stream_output) / "topomap"
    pred_path.mkdir(parents=True, exist_ok=True)
    topo_path.mkdir(parents=True, exist_ok=True)

    pred_struct = StructType([StructField(c, StringType() if "label" in c or c in ["event_time", "subject_id", "run_id"] else LongType() if c in ["epoch_id", "run_seq", "correct"] else DoubleType(), True) for c in ["event_time", "epoch_id", "subject_id", "run_id", "run_seq", "true_label", "pred_label", "proba_T0", "proba_T1", "proba_T2", "correct"]])
    topo_struct = StructType([StructField(c, StringType() if c in ["event_time", "channel", "true_label", "pred_label"] else LongType() if c == "epoch_id" else DoubleType(), True) for c in ["event_time", "epoch_id", "channel", "alpha_power", "true_label", "pred_label"]])
    udf_schema = StructType([StructField("prediction", pred_struct, True), StructField("topomaps", ArrayType(topo_struct), True)])

    def process_epoch_distributed(pdf: pd.DataFrame) -> pd.DataFrame:
        if pdf.empty: return pd.DataFrame()
        model, encoder = bc_clf.value, bc_enc.value
        eid, event_time, true_label, subject = pdf["epoch_id"].iloc[0], pdf["event_time"].iloc[0], pdf["epoch_label"].iloc[0], pdf["subject_id"].iloc[0]
        run_id = pdf["run_id"].iloc[0] if "run_id" in pdf.columns else "?"
        run_seq_val = int(pdf["run_seq"].iloc[0]) if "run_seq" in pdf.columns and not pd.isna(pdf["run_seq"].iloc[0]) else -1

        try:
            feats = compute_features(pdf)
            x = np.asarray([[feats[k] for k in FEATURE_ORDER]])
            pred_idx = int(model.predict(x)[0])
            pred_label = encoder.inverse_transform([pred_idx])[0]
            proba_dict = {encoder.inverse_transform([i])[0]: float(p) for i, p in enumerate(model.predict_proba(x)[0])}
        except Exception:
            pred_label, proba_dict = "?", {}

        pred_dict = {"event_time": str(event_time), "epoch_id": int(eid), "subject_id": str(subject), "run_id": str(run_id), "run_seq": int(run_seq_val), "true_label": str(true_label), "pred_label": str(pred_label), "proba_T0": float(proba_dict.get("T0", 0.0)), "proba_T1": float(proba_dict.get("T1", 0.0)), "proba_T2": float(proba_dict.get("T2", 0.0)), "correct": int(pred_label == true_label)}
        topo_list = [{"event_time": str(event_time), "epoch_id": int(eid), "channel": str(ch), "alpha_power": float(pw), "true_label": str(true_label), "pred_label": str(pred_label)} for ch, pw in compute_alpha_topomap(pdf).items()]

        return pd.DataFrame([{"prediction": pred_dict, "topomaps": topo_list}])

    def process_batch(batch_df, batch_id: int) -> None:
        if batch_df.rdd.isEmpty(): return
        processed_df = batch_df.groupBy("epoch_id").applyInPandas(process_epoch_distributed, schema=udf_schema).cache()
        processed_df.select("prediction.*").write.mode("append").parquet(str(pred_path))
        processed_df.select(F.explode("topomaps").alias("topo")).select("topo.*").write.mode("append").parquet(str(topo_path))
        log.info("Batch %d traité de manière distribuée via applyInPandas.", batch_id)
        processed_df.unpersist()

    query = events.writeStream.outputMode("update").foreachBatch(process_batch).option("checkpointLocation", checkpoint).trigger(processingTime="1 seconds").queryName("eeg_stream").start()
    try: query.awaitTermination(timeout=timeout_seconds)
    except KeyboardInterrupt: pass
    finally: [q.stop() for q in spark.streams.active]

    return {"timeout_seconds": timeout_seconds, "channels": len(channels)}
