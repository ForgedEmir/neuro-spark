"""
NeuroSpark — Core functions extracted from poc_eeg.ipynb.
These are the exact functions used in the notebook, made importable.
"""
import os
import numpy as np
import pandas as pd
import mne
import logging
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import pandas_udf, col, collect_list
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

log = logging.getLogger(__name__)

# ── Spark Session ──
def create_spark_session():
    """Crée une SparkSession avec config optimale pour 66 sujets EEG."""
    spark = (SparkSession.builder
        .appName('EEG-PoC')
        .master('spark://spark-master:7077')
        .config('spark.executor.memory', '8g')
        .config('spark.driver.memory', '4g')
        .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
        .config('spark.sql.shuffle.partitions', '32')
        .getOrCreate())
    log.info('SparkSession prête')
    return spark


# ── Constants ──
FS = 160  # Hz
EPOCH_SEC = 2  # secondes
MOTOR_CHANNELS = ["C3..", "Cz..", "C4.."]
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 80)}
SEED = 42
TRAIN_RATIO = 0.78  # ~52/66 sujets
T0_WEIGHT = 0.5

# --- Ingestion : EDF → Parquet ---
# (Franchement, la partie la plus longue c'est de convertir tous les fichiers la première fois...)
def edf_to_dataframe(edf_path, subject_id, run_id):
    # Ouvre le fichier EDF, extrait les données et les labels (T0/T1/T2)
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

def batch_convert_edf(edf_dir, parquet_dir):
    # Convertit tous les runs moteurs EDF en Parquet (idempotent, donc safe à relancer)
    os.makedirs(parquet_dir, exist_ok=True)
    stats = {'converted': 0, 'skipped': 0, 'errors': 0}
    subjects = sorted(os.listdir(edf_dir))
    log.info(f'Sujets trouvés : {len(subjects)}')
    for i, subject in enumerate(subjects, 1):
        subject_dir = os.path.join(edf_dir, subject)
        if not os.path.isdir(subject_dir):
            continue
        for run in MOTOR_RUNS:
            edf_file     = os.path.join(subject_dir, f'{subject}{run}.edf')
            parquet_file = os.path.join(parquet_dir, f'{subject}_{run}.parquet')
            if not os.path.exists(edf_file):
                continue
            if os.path.exists(parquet_file):
                stats['skipped'] += 1
                continue
            try:
                edf_to_dataframe(edf_file, subject, run).to_parquet(parquet_file, index=False)
                stats['converted'] += 1
            except Exception as e:
                stats['errors'] += 1
                log.warning(f'Erreur {subject}/{run} : {e}')
        if i % 10 == 0 or i == len(subjects):
            log.info(f"[{i}/{len(subjects)}] {subject} — {stats['converted']} convertis, {stats['skipped']} déjà présents")
    log.info(f"Terminé : {stats['converted']} nouveaux Parquet, {stats['skipped']} déjà existants, {stats['errors']} erreurs")
    return stats

# --- Feature engineering : nettoyage, epoching, FFT, normalisation ---
def clean_channel_names(df):
    # Certains noms de canaux ont des points, ce qui embête Spark. On les vire une fois pour toutes.
    rename_map = {c: c.replace('.', '') for c in df.columns if '.' in c}
    return df.select([F.col(f'`{c}`').alias(rename_map.get(c, c)) for c in df.columns])

def add_epochs(df, epoch_sec=EPOCH_SEC):
    # Découpe le signal en fenêtres de 2s (320 samples à 160 Hz)
    return df.withColumn('epoch_id', F.floor(F.col('time') / epoch_sec).cast('int'))

def _make_band_udf(fs, low, high):
    # Pandas UDF pour calculer la puissance dans une bande de fréquences (FFT)
    @pandas_udf(DoubleType())
    def band_power_udf(signals):
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

def _ordered_signal(canal):
    # collect_list() seul ne garantit pas l'ordre après un shuffle, donc on trie à la main
    struct_col = F.struct(F.col('time'), F.col(canal).alias('v'))
    return F.transform(F.array_sort(F.collect_list(struct_col)), lambda s: s['v'])

def build_band_features(df_epochs, channels=MOTOR_CHANNELS, bands=BANDS, fs=FS):
    # Pour chaque epoch, calcule la puissance FFT par bande x canal
    agg_exprs = [F.first('task_label').alias('task_label')]
    for canal in channels:
        signal_expr = _ordered_signal(canal)
        for band_name, (low, high) in bands.items():
            agg_exprs.append(_make_band_udf(fs, low, high)(signal_expr).alias(f'{canal}_{band_name}'))
    return df_epochs.groupBy('subject_id', 'run_id', 'epoch_id').agg(*agg_exprs)

def add_lateralization_features(df_features, bands=BANDS):
    # Ajoute diff_band = C3_band - C4_band pour voir la latéralisation hémisphérique
    return df_features.withColumns({f'diff_{b}': F.col(f'C3_{b}') - F.col(f'C4_{b}') for b in bands})

def normalize_by_subject(df_features, feature_cols, t0_weight=T0_WEIGHT):
    # Z-score par sujet pour neutraliser les différences d'amplitude inter-sujet
    w = Window.partitionBy('subject_id')
    meta = ['subject_id', 'run_id', 'epoch_id', 'task_label']
    norm_exprs = [F.col(c) for c in meta]
    for fc in feature_cols:
        std = F.coalesce(F.stddev(fc).over(w), F.lit(1.0))
        norm_exprs.append(((F.col(fc) - F.mean(fc).over(w)) / (std + 1e-10)).alias(fc))
    return df_features.select(norm_exprs).withColumn(
        'weight', F.when(F.col('task_label') == 'T0', t0_weight).otherwise(1.0))

def extract_feature_cols(df_features):
    # Retourne les colonnes de features (tout sauf métadonnées)
    return [c for c in df_features.columns if c not in {'subject_id', 'run_id', 'epoch_id', 'task_label', 'weight'}]

# --- MLlib ---
# (Attention à ne pas faire de data leakage. Split par sujet, pas par epoch)
def split_by_subject(df, train_ratio=TRAIN_RATIO):
    all_subjects = sorted([r.subject_id for r in df.select('subject_id').distinct().collect()])
    n_train = int(len(all_subjects) * train_ratio)
    train_subjects, test_subjects = all_subjects[:n_train], all_subjects[n_train:]
    return (df.filter(F.col('subject_id').isin(train_subjects)),
            df.filter(F.col('subject_id').isin(test_subjects)),
            train_subjects, test_subjects)

def build_pipeline(feature_cols, seed=SEED):
    # Pipeline MLlib : StringIndexer → VectorAssembler → RandomForest
    return Pipeline(stages=[
        StringIndexer(inputCol='task_label', outputCol='label'),
        VectorAssembler(inputCols=feature_cols, outputCol='features'),
        RandomForestClassifier(featuresCol='features', labelCol='label', weightCol='weight', seed=seed),
    ])

def train_with_cv(pipeline, train_df, num_trees_grid=None, max_depth_grid=None, num_folds=3, seed=SEED):
    # CrossValidator sur grille numTrees × maxDepth — 4 combos × 3 folds = 12 entraînements
    num_trees_grid = num_trees_grid or [50, 100]
    max_depth_grid = max_depth_grid or [5, 10]
    rf = pipeline.getStages()[-1]
    param_grid = (ParamGridBuilder()
        .addGrid(rf.numTrees, num_trees_grid)
        .addGrid(rf.maxDepth, max_depth_grid)
        .build())
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy'),
        numFolds=num_folds, seed=seed)
    return cv.fit(train_df)

def evaluate_model(model, test_df):
    # Évalue le modèle avec accuracy, F1, precision et recall weighted
    predictions = model.transform(test_df)
    metrics = {}
    for name, metric in [('accuracy', 'accuracy'), ('f1', 'f1'),
                          ('precision_weighted', 'weightedPrecision'),
                          ('recall_weighted', 'weightedRecall')]:
        metrics[name] = MulticlassClassificationEvaluator(
            labelCol='label', predictionCol='prediction', metricName=metric).evaluate(predictions)
    return {'predictions': predictions, 'metrics': metrics}

# --- Export dashboard ---
# (Ici, on extrait juste des petits bouts pour le dashboard, pas tout le dataset)
def export_dashboard_data(df_raw, df_features, predictions, cv_model, feature_cols,
                          sample_subject='S001', sample_run='R03', output_dir=DASHBOARD_DIR):
    os.makedirs(output_dir, exist_ok=True)
    (df_raw.filter((F.col('subject_id') == sample_subject) & (F.col('run_id') == sample_run))
        .select('time', 'Cz', 'task_label').orderBy('time').limit(3200)
        .toPandas().to_parquet(f'{output_dir}/signal_sample.parquet', index=False))
    df_features.toPandas().to_parquet(f'{output_dir}/features.parquet', index=False)
    (predictions.select('subject_id', 'task_label', 'prediction')
        .toPandas().to_parquet(f'{output_dir}/predictions.parquet', index=False))
    rf_model = cv_model.bestModel.stages[-1]
    (pd.DataFrame({'feature': feature_cols, 'importance': rf_model.featureImportances.toArray()})
        .sort_values('importance', ascending=False)
        .to_parquet(f'{output_dir}/feature_importance.parquet', index=False))
    log.info(f'Export terminé → {output_dir}')

print('Fonctions chargées.')
