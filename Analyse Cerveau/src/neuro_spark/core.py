import os
import time
import logging
from typing import Optional, Callable
import numpy as np
import pandas as pd
import mne
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import pandas_udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession, DataFrame

log = logging.getLogger(__name__)

FS: int = 160
EPOCH_SEC: int = 2
MOTOR_CHANNELS: list[str] = ["C3", "Cz", "C4"]
BANDS: dict[str, tuple[int, int]] = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 80),
}
SEED: int = 42
TRAIN_RATIO: float = 0.78
T0_WEIGHT: float = 0.5


def edf_to_dataframe(edf_path: str, subject_id: str, run_id: str) -> pd.DataFrame:
    """Read an EDF file and return a DataFrame with time, channel signals, and task labels."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data, times = raw.get_data(return_times=True)
    df = pd.DataFrame(data.T, columns=raw.ch_names)
    df.insert(0, "time", times)
    df.insert(0, "run_id", run_id)
    df.insert(0, "subject_id", subject_id)
    df["task_label"] = "T0"
    for ann in raw.annotations:
        mask = (df["time"] >= ann["onset"]) & (
            df["time"] < ann["onset"] + ann["duration"]
        )
        df.loc[mask, "task_label"] = ann["description"]
    return df


def batch_convert_edf(edf_dir: str, parquet_dir: str) -> dict:
    """Convert motor imagery EDF files to partitioned Parquet. Skips already converted files."""
    os.makedirs(parquet_dir, exist_ok=True)
    stats = {"converted": 0, "skipped": 0, "errors": 0}
    subjects = sorted(os.listdir(edf_dir))
    motor_runs = [f"R{i:02d}" for i in range(3, 15)]

    for i, subject in enumerate(subjects, 1):
        subject_dir = os.path.join(edf_dir, subject)
        if not os.path.isdir(subject_dir):
            continue
        for run in motor_runs:
            edf_file = os.path.join(subject_dir, f"{subject}{run}.edf")
            partition_path = os.path.join(
                parquet_dir, f"subject_id={subject}", f"run_id={run}"
            )
            if not os.path.exists(edf_file):
                continue
            if os.path.exists(partition_path):
                stats["skipped"] += 1
                continue
            try:
                os.makedirs(partition_path, exist_ok=True)
                out_file = os.path.join(partition_path, "data.parquet")
                edf_to_dataframe(edf_file, subject, run).to_parquet(
                    out_file, index=False
                )
                stats["converted"] += 1
            except Exception as e:
                stats["errors"] += 1
                log.warning(f"Error converting {subject}/{run}: {e}")
        if i % 10 == 0 or i == len(subjects):
            log.info(
                f"[{i}/{len(subjects)}] {subject} - "
                f"{stats['converted']} converted, {stats['skipped']} skipped"
            )
    return stats


def clean_channel_names(df: DataFrame) -> DataFrame:
    """Rename EEG channels by removing trailing dots (e.g. C3.. -> C3)."""
    rename_map = {c: c.replace(".", "") for c in df.columns if "." in c}
    return df.select(
        [F.col(f"`{c}`").alias(rename_map.get(c, c)) for c in df.columns]
    )


def add_epochs(df: DataFrame, epoch_sec: int = EPOCH_SEC) -> DataFrame:
    """Assign an epoch ID to each sample based on a fixed time window."""
    return df.withColumn("epoch_id", F.floor(F.col("time") / epoch_sec).cast("int"))


def _make_band_udf(fs: int, low: int, high: int) -> Callable:
    """Return a Pandas UDF that computes FFT power in a given frequency band."""

    @pandas_udf(DoubleType())
    def band_power_udf(signals: pd.Series) -> pd.Series:
        def compute(vals):
            arr = np.array(vals, dtype=float)
            if len(arr) < 2:
                return 0.0
            freqs = np.fft.rfftfreq(len(arr), 1.0 / fs)
            power = np.abs(np.fft.rfft(arr)) ** 2
            mask = (freqs >= low) & (freqs < high)
            return float(np.mean(power[mask])) if mask.any() else 0.0

        return signals.apply(compute)

    return band_power_udf


def _ordered_signal(canal: str) -> F.Column:
    """Collect signal samples for an epoch in chronological order."""
    struct_col = F.struct(F.col("time"), F.col(canal).alias("v"))
    return F.transform(F.array_sort(F.collect_list(struct_col)), lambda s: s["v"])


def build_band_features(
    df_epochs: DataFrame,
    channels: Optional[list[str]] = None,
    bands: Optional[dict[str, tuple[int, int]]] = None,
    fs: int = FS,
) -> DataFrame:
    """Compute FFT band power per epoch, channel, and frequency band."""
    channels = channels or MOTOR_CHANNELS
    bands = bands or BANDS
    agg_exprs = [F.first("task_label", ignorenulls=True).alias("task_label")]
    for canal in channels:
        signal_expr = _ordered_signal(canal)
        for band_name, (low, high) in bands.items():
            agg_exprs.append(
                _make_band_udf(fs, low, high)(signal_expr).alias(
                    f"{canal}_{band_name}"
                )
            )
    return df_epochs.groupBy("subject_id", "run_id", "epoch_id").agg(*agg_exprs)


def add_lateralization_features(
    df_features: DataFrame, bands: Optional[dict] = None
) -> DataFrame:
    """Add hemispheric lateralization index: diff_band = C3_band - C4_band."""
    bands = bands or BANDS
    return df_features.withColumns(
        {f"diff_{b}": F.col(f"C3_{b}") - F.col(f"C4_{b}") for b in bands}
    )


def normalize_by_subject(
    df_features: DataFrame, feature_cols: list[str], t0_weight: float = T0_WEIGHT
) -> DataFrame:
    """Z-score normalization per subject using broadcast join on aggregated stats."""
    meta = ["subject_id", "run_id", "epoch_id", "task_label"]

    agg_exprs = []
    for fc in feature_cols:
        agg_exprs.extend(
            [F.mean(fc).alias(f"{fc}_mean"), F.stddev(fc).alias(f"{fc}_std")]
        )
    stats = df_features.groupBy("subject_id").agg(*agg_exprs)

    df_joined = df_features.select(meta + feature_cols).join(
        F.broadcast(stats), "subject_id", "left"
    )

    norm_cols = [F.col(c) for c in meta]
    for fc in feature_cols:
        norm_cols.append(
            (
                (F.col(fc) - F.col(f"{fc}_mean"))
                / F.coalesce(F.col(f"{fc}_std"), F.lit(1.0))
            ).alias(fc)
        )

    return df_joined.select(norm_cols).withColumn(
        "weight", F.when(F.col("task_label") == "T0", t0_weight).otherwise(1.0)
    )


def extract_feature_cols(df_features: DataFrame) -> list[str]:
    """Return feature column names, excluding metadata columns."""
    skip = {"subject_id", "run_id", "epoch_id", "task_label", "weight"}
    return [c for c in df_features.columns if c not in skip]


def split_by_subject(df: DataFrame, train_ratio: float = TRAIN_RATIO) -> tuple:
    """Split subjects into train/test sets to avoid data leakage across epochs."""
    all_subjects = sorted(
        [r.subject_id for r in df.select("subject_id").distinct().collect()]
    )
    n_train = int(len(all_subjects) * train_ratio)
    train_subjects = all_subjects[:n_train]
    test_subjects = all_subjects[n_train:]
    return (
        df.filter(F.col("subject_id").isin(train_subjects)),
        df.filter(F.col("subject_id").isin(test_subjects)),
        train_subjects,
        test_subjects,
    )


def build_pipeline(feature_cols: list[str], seed: int = SEED) -> Pipeline:
    """Build a MLlib pipeline: StringIndexer -> VectorAssembler -> RandomForest."""
    return Pipeline(
        stages=[
            StringIndexer(inputCol="task_label", outputCol="label"),
            VectorAssembler(inputCols=feature_cols, outputCol="features"),
            RandomForestClassifier(
                featuresCol="features",
                labelCol="label",
                weightCol="weight",
                seed=seed,
                minInstancesPerNode=2,
            ),
        ]
    )


def train_with_cv(
    pipeline: Pipeline,
    train_df: DataFrame,
    num_trees_grid: Optional[list[int]] = None,
    max_depth_grid: Optional[list[int]] = None,
    num_folds: int = 3,
    seed: int = SEED,
) -> CrossValidatorModel:
    """Run CrossValidator with a grid of num_trees and max_depth values."""
    num_trees_grid = num_trees_grid or [50, 100]
    max_depth_grid = max_depth_grid or [5, 10]
    rf = pipeline.getStages()[-1]
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, num_trees_grid)
        .addGrid(rf.maxDepth, max_depth_grid)
        .build()
    )
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        ),
        numFolds=num_folds,
        seed=seed,
        collectSubModels=False,
        parallelism=2,
    )
    return cv.fit(train_df)


def evaluate_model(model: CrossValidatorModel, test_df: DataFrame) -> dict:
    """Evaluate the model on a test set and return accuracy, F1, precision, recall."""
    predictions = model.transform(test_df)
    metrics = {}
    for name, metric in [
        ("accuracy", "accuracy"),
        ("f1", "f1"),
        ("precision_weighted", "weightedPrecision"),
        ("recall_weighted", "weightedRecall"),
    ]:
        metrics[name] = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName=metric
        ).evaluate(predictions)
    return {"predictions": predictions, "metrics": metrics}


def timed_step(name: str, fn: Callable, *args, **kwargs):
    """Run a function and log how long it took."""
    t0 = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0
    log.info(f"{name} completed in {elapsed:.1f}s")
    return result, elapsed


def optimize_delta_table(spark: SparkSession, path: str) -> None:
    """Compact small Delta files and remove old snapshots."""
    spark.sql(f"OPTIMIZE delta.`{path}`")
    spark.sql(f"VACUUM delta.`{path}` RETAIN 168 HOURS")
