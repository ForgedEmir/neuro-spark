"""
NeuroSpark — Core functions.
"""
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
from pyspark.sql.window import Window
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession, DataFrame

log = logging.getLogger(__name__)

# ── Constants ──
FS: int = 160
EPOCH_SEC: int = 2
MOTOR_CHANNELS: list[str] = ["C3..", "Cz..", "C4.."]
BANDS: dict[str, tuple[int, int]] = {
    "theta": (4, 8), "alpha": (8, 13),
    "beta": (13, 30), "gamma": (30, 80),
}
SEED: int = 42
TRAIN_RATIO: float = 0.78
T0_WEIGHT: float = 0.5


def create_spark_session(delta_enabled: bool = True) -> SparkSession:
    """Crée une SparkSession avec config optimale + Delta Lake optionnel."""
    builder = (SparkSession.builder
        .appName('EEG-PoC-v2')
        .master('spark://spark-master:7077')
        .config('spark.executor.memory', '8g')
        .config('spark.driver.memory', '4g')
        .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
        .config('spark.sql.shuffle.partitions', '32')
        .config('spark.sql.adaptive.enabled', 'true')
        .config('spark.sql.adaptive.coalescePartitions.enabled', 'true'))
    if delta_enabled:
        builder = (builder
            .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension')
            .config('spark.sql.catalog.spark_catalog',
                    'org.apache.spark.sql.delta.catalog.DeltaCatalog'))
    return builder.getOrCreate()


def edf_to_dataframe(edf_path: str, subject_id: str, run_id: str) -> pd.DataFrame:
    """Ouvre un fichier EDF, extrait les données et les labels T0/T1/T2."""
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    data, times = raw.get_data(return_times=True)
    df = pd.DataFrame(data.T, columns=raw.ch_names)
    df.insert(0, 'time', times)
    df.insert(0, 'run_id', run_id)
    df.insert(0, 'subject_id', subject_id)
    df['task_label'] = 'T0'
    for ann in raw.annotations:
        mask = (df['time'] >= ann['onset']) & (df['time'] < ann['onset'] + ann['duration'])
        df.loc[mask, 'task_label'] = ann['description']
    return df


def batch_convert_edf(edf_dir: str, parquet_dir: str) -> dict:
    """
    Convertit tous les runs moteurs EDF en Parquet (idempotent).
    Écrit en structure partitionnée subject_id=/run_id=/ → partition pruning.
    """
    os.makedirs(parquet_dir, exist_ok=True)
    stats = {'converted': 0, 'skipped': 0, 'errors': 0}
    subjects = sorted(os.listdir(edf_dir))
    motor_runs = [f'R{i:02d}' for i in range(3, 15)]
    log.info(f'Sujets trouvés : {len(subjects)}')

    for i, subject in enumerate(subjects, 1):
        subject_dir = os.path.join(edf_dir, subject)
        if not os.path.isdir(subject_dir):
            continue
        for run in motor_runs:
            edf_file = os.path.join(subject_dir, f'{subject}{run}.edf')
            partition_path = os.path.join(parquet_dir, f'subject_id={subject}', f'run_id={run}')
            if not os.path.exists(edf_file):
                continue
            if os.path.exists(partition_path):
                stats['skipped'] += 1
                continue
            try:
                os.makedirs(partition_path, exist_ok=True)
                out_file = os.path.join(partition_path, 'data.parquet')
                edf_to_dataframe(edf_file, subject, run).to_parquet(out_file, index=False)
                stats['converted'] += 1
            except Exception as e:
                stats['errors'] += 1
                log.warning(f'Erreur {subject}/{run} : {e}')
        if i % 10 == 0 or i == len(subjects):
            log.info(f"[{i}/{len(subjects)}] {subject} — {stats['converted']} convertis, "
                     f"{stats['skipped']} déjà présents")
    log.info(f"Terminé : {stats['converted']} nouveaux, {stats['skipped']} skip, "
             f"{stats['errors']} erreurs")
    return stats


def clean_channel_names(df: DataFrame) -> DataFrame:
    """Supprime les points dans les noms de canaux (C3.. → C3)."""
    rename_map = {c: c.replace('.', '') for c in df.columns if '.' in c}
    return df.select([F.col(f'`{c}`').alias(rename_map.get(c, c)) for c in df.columns])


def add_epochs(df: DataFrame, epoch_sec: int = EPOCH_SEC) -> DataFrame:
    """Découpe le signal en fenêtres de 2s (320 samples à 160 Hz)."""
    return df.withColumn('epoch_id', F.floor(F.col('time') / epoch_sec).cast('int'))


def _make_band_udf(fs: int, low: int, high: int) -> Callable:
    """Pandas UDF pour calculer la puissance FFT dans une bande de fréquences."""
    @pandas_udf(DoubleType())
    def band_power_udf(signals: pd.Series) -> pd.Series:
        def compute(vals):
            arr = np.array(vals, dtype=float)
            if len(arr) < 2:
                return 0.0
            freqs = np.fft.rfftfreq(len(arr), 1.0 / fs)
            power = np.abs(np.fft.rfft(arr)) ** 2
            mask  = (freqs >= low) & (freqs < high)
            return float(np.mean(power[mask])) if mask.any() else 0.0
        return signals.apply(compute)
    return band_power_udf


def _ordered_signal(canal: str) -> F.Column:
    """Garantit l'ordre temporel après collect_list via sort_array + struct."""
    struct_col = F.struct(F.col('time'), F.col(canal).alias('v'))
    return F.transform(F.array_sort(F.collect_list(struct_col)), lambda s: s['v'])


def build_band_features(df_epochs: DataFrame, channels: Optional[list[str]] = None,
                        bands: Optional[dict[str, tuple[int, int]]] = None,
                        fs: int = FS) -> DataFrame:
    """Calcule la puissance FFT par bande × canal pour chaque epoch."""
    channels = channels or MOTOR_CHANNELS
    bands = bands or BANDS
    agg_exprs = [F.first('task_label').alias('task_label')]
    for canal in channels:
        signal_expr = _ordered_signal(canal)
        for band_name, (low, high) in bands.items():
            agg_exprs.append(
                _make_band_udf(fs, low, high)(signal_expr).alias(f'{canal}_{band_name}'))
    return df_epochs.groupBy('subject_id', 'run_id', 'epoch_id').agg(*agg_exprs)


def add_lateralization_features(df_features: DataFrame, bands: Optional[dict] = None) -> DataFrame:
    """Ajoute diff_band = C3_band - C4_band (latéralisation hémisphérique)."""
    bands = bands or BANDS
    return df_features.withColumns(
        {f'diff_{b}': F.col(f'C3.._{b}') - F.col(f'C4.._{b}') for b in bands})


def normalize_by_subject(df_features: DataFrame, feature_cols: list[str],
                         t0_weight: float = T0_WEIGHT) -> DataFrame:
    """
    Z-score par sujet.
    OPTIMISATION: 1 groupBy agg + broadcast join au lieu de N Window functions.
    """
    meta = ['subject_id', 'run_id', 'epoch_id', 'task_label']

    agg_exprs = []
    for fc in feature_cols:
        agg_exprs.extend([F.mean(fc).alias(f'{fc}_mean'),
                          F.stddev(fc).alias(f'{fc}_std')])
    stats = df_features.groupBy('subject_id').agg(*agg_exprs)

    df_joined = df_features.select(meta + feature_cols).join(
        F.broadcast(stats), 'subject_id', 'left')

    norm_cols = [F.col(c) for c in meta]
    for fc in feature_cols:
        norm_cols.append(((F.col(fc) - F.col(f'{fc}_mean'))
                          / F.coalesce(F.col(f'{fc}_std'), F.lit(1.0))).alias(fc))

    return df_joined.select(norm_cols).withColumn(
        'weight', F.when(F.col('task_label') == 'T0', t0_weight).otherwise(1.0))


def extract_feature_cols(df_features: DataFrame) -> list[str]:
    """Retourne les colonnes de features (tout sauf métadonnées)."""
    skip = {'subject_id', 'run_id', 'epoch_id', 'task_label', 'weight'}
    return [c for c in df_features.columns if c not in skip]


def split_by_subject(df: DataFrame, train_ratio: float = TRAIN_RATIO) -> tuple:
    """Split par sujet — pas de data leakage."""
    all_subjects = sorted([r.subject_id for r in df.select('subject_id').distinct().collect()])
    n_train = int(len(all_subjects) * train_ratio)
    train_subjects, test_subjects = all_subjects[:n_train], all_subjects[n_train:]
    return (df.filter(F.col('subject_id').isin(train_subjects)),
            df.filter(F.col('subject_id').isin(test_subjects)),
            train_subjects, test_subjects)


def build_pipeline(feature_cols: list[str], seed: int = SEED) -> Pipeline:
    """Pipeline MLlib : StringIndexer → VectorAssembler → RandomForest."""
    return Pipeline(stages=[
        StringIndexer(inputCol='task_label', outputCol='label'),
        VectorAssembler(inputCols=feature_cols, outputCol='features'),
        RandomForestClassifier(featuresCol='features', labelCol='label',
                               weightCol='weight', seed=seed, minInstancesPerNode=2),
    ])


def train_with_cv(pipeline: Pipeline, train_df: DataFrame,
                  num_trees_grid: Optional[list[int]] = None,
                  max_depth_grid: Optional[list[int]] = None,
                  num_folds: int = 3, seed: int = SEED) -> CrossValidatorModel:
    """CrossValidator: 4 combos × 3 folds = 12 trainings. Optimisé."""
    num_trees_grid = num_trees_grid or [50, 100]
    max_depth_grid = max_depth_grid or [5, 10]
    rf = pipeline.getStages()[-1]
    param_grid = (ParamGridBuilder()
        .addGrid(rf.numTrees, num_trees_grid)
        .addGrid(rf.maxDepth, max_depth_grid)
        .build())
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid,
        evaluator=MulticlassClassificationEvaluator(labelCol='label',
            predictionCol='prediction', metricName='accuracy'),
        numFolds=num_folds, seed=seed, collectSubModels=False, parallelism=2)
    return cv.fit(train_df)


def evaluate_model(model: CrossValidatorModel, test_df: DataFrame) -> dict:
    """Évalue avec accuracy, F1, precision, recall weighted."""
    predictions = model.transform(test_df)
    metrics = {}
    for name, metric in [('accuracy', 'accuracy'), ('f1', 'f1'),
                          ('precision_weighted', 'weightedPrecision'),
                          ('recall_weighted', 'weightedRecall')]:
        metrics[name] = MulticlassClassificationEvaluator(
            labelCol='label', predictionCol='prediction', metricName=metric
        ).evaluate(predictions)
    return {'predictions': predictions, 'metrics': metrics}


def timed_step(name: str, fn: Callable, *args, **kwargs):
    """Mesure le temps d'une étape. Résultat + float secondes."""
    log.info(f'▶ {name}...')
    t0 = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0
    log.info(f'✓ {name}: {elapsed:.1f}s')
    return result, elapsed


def optimize_delta_table(spark: SparkSession, path: str) -> None:
    """Compacte les petits fichiers Delta et nettoie les vieux snapshots."""
    spark.sql(f"OPTIMIZE delta.`{path}`")
    spark.sql(f"VACUUM delta.`{path}` RETAIN 168 HOURS")
    log.info(f'Delta table optimisée: {path}')


print('core.py v2.2 — type hints + ruff clean')
