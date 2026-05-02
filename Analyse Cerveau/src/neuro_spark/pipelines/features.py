"""
Pipeline de feature engineering : FFT spectrale sur signaux EEG.
Extrait du notebook poc_eeg.ipynb — rendu exécutable standalone.
"""
import os
import sys
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.functions import pandas_udf, col, collect_list, floor
from pyspark.sql.types import ArrayType, DoubleType as SparkDoubleType

# ── Configuration ──
PARQUET_DIR = os.environ.get('PARQUET_DIR', '/opt/spark/data/parquet/')
FEATURES_DIR = os.environ.get('FEATURES_DIR', '/opt/spark/data/features/')
SPARK_MASTER = os.environ.get('SPARK_MASTER', 'spark://spark-master:7077')

# Paramètres
EPOCH_SEC = 2
FS = 160
BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 80),
}
MOTOR_CHANNELS = ['C3..', 'Cz..', 'C4..']


def create_spark_session():
    return SparkSession.builder \
        .appName('EEG-Features') \
        .master(SPARK_MASTER) \
        .config('spark.executor.memory', '4g') \
        .config('spark.driver.memory', '2g') \
        .config('spark.sql.adaptive.enabled', 'true') \
        .getOrCreate()


def band_power_udf(band_name, low_freq, high_freq):
    """
    Crée une UDF qui calcule la puissance spectrale dans une bande de fréquence.
    Retourne la puissance moyenne dans [low_freq, high_freq] Hz.
    """
    @pandas_udf(SparkDoubleType())
    def _udf(signal_series):
        def compute(signal):
            if signal is None or len(signal) < 4:
                return 0.0
            arr = np.array(signal)
            fft = np.abs(np.fft.rfft(arr)) ** 2
            freqs = np.fft.rfftfreq(len(arr), 1 / FS)
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            if mask.sum() == 0:
                return 0.0
            return float(np.mean(fft[mask]))
        return signal_series.apply(compute)
    return _udf


def main():
    spark = create_spark_session()

    # ── Lecture avec schéma explicite ──
    meta_fields = [
        StructField('subject_id', StringType(), True),
        StructField('run_id', StringType(), True),
        StructField('time', DoubleType(), True),
        StructField('task_label', StringType(), True),
    ]

    canal_names = [
        'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
        'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
        'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.',
        'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.',
        'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..',
        'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.',
        'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..',
        'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..',
    ]
    canal_fields = [StructField(name, DoubleType(), True) for name in canal_names]
    schema = StructType(meta_fields + canal_fields)

    df = spark.read.schema(schema).parquet(PARQUET_DIR)
    print(f'Lecture: {df.count():,} lignes, {len(df.columns)} colonnes')

    # ── Renommage des canaux (supprimer les points) ──
    for c in df.columns:
        if '.' in c:
            df = df.withColumnRenamed(c, c.replace('.', ''))

    # ── Découpage en epochs ──
    df = df.withColumn('epoch_id', floor(col('time') / EPOCH_SEC).cast('int'))

    # ── Feature engineering par canal moteur ──
    motor_clean = [c.replace('.', '') for c in MOTOR_CHANNELS]
    meta_cols = ['subject_id', 'run_id', 'time', 'task_label', 'epoch_id']

    # Agrégation par epoch : collecter les samples + labels
    agg_exprs = [F.first(c).alias(c) for c in meta_cols if c != 'time']
    for ch in motor_clean:
        agg_exprs.append(collect_list(col(ch)).alias(f'{ch}_samples'))

    df_epochs = df.groupBy('epoch_id').agg(*agg_exprs)

    # Calcul de la puissance par bande pour chaque canal
    for ch in motor_clean:
        for band_name, (low, high) in BANDS.items():
            udf = band_power_udf(band_name, low, high)
            df_epochs = df_epochs.withColumn(
                f'{ch}_{band_name}',
                udf(col(f'{ch}_samples'))
            )
        # Nettoyer la colonne de samples (lourde en mémoire)
        df_epochs = df_epochs.drop(f'{ch}_samples')

    # ── Feature: Différence C3-C4 (latéralisation) ──
    for band_name in BANDS:
        c3_col = f'C3_{band_name}'
        c4_col = f'C4_{band_name}'
        df_epochs = df_epochs.withColumn(
            f'diff_C3_C4_{band_name}',
            col(c3_col) - col(c4_col)
        )

    # ── Normalisation intra-sujet ──
    feature_cols = (
        [f'{ch}_{band}' for ch in motor_clean for band in BANDS] +
        [f'diff_C3_C4_{band}' for band in BANDS]
    )

    from pyspark.sql.window import Window
    w = Window.partitionBy('subject_id')

    for fc in feature_cols:
        mean_col = F.mean(fc).over(w)
        std_col = F.stddev(fc).over(w)
        df_epochs = df_epochs.withColumn(fc, (col(fc) - mean_col) / std_col)

    # ── Sauvegarde ──
    os.makedirs(FEATURES_DIR, exist_ok=True)
    df_epochs.write.mode('overwrite').parquet(FEATURES_DIR)

    n_epochs = df_epochs.count()
    n_features = len(feature_cols)
    print(f'✓ Features: {n_epochs:,} epochs × {n_features} features sauvegardées dans {FEATURES_DIR}')

    spark.stop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
