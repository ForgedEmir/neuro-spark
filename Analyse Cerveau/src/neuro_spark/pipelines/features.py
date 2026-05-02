"""
Pipeline de feature engineering : FFT spectrale + normalisation.
Utilise les fonctions extraites du notebook poc_eeg.ipynb.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_spark.core import (
    create_spark_session,
    clean_channel_names,
    add_epochs,
    build_band_features,
    add_lateralization_features,
    normalize_by_subject,
    extract_feature_cols,
    T0_WEIGHT,
)

PARQUET_DIR = os.environ.get('PARQUET_DIR', '/opt/spark/data/parquet/')
FEATURES_DIR = os.environ.get('FEATURES_DIR', '/opt/spark/data/features/')
SPARK_MASTER = os.environ.get('SPARK_MASTER', 'spark://spark-master:7077')


def main():
    spark = create_spark_session()

    # Lecture avec schéma explicite (correction grille)
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType
    meta_fields = [
        StructField("subject_id", StringType(), True),
        StructField("run_id", StringType(), True),
        StructField("time", DoubleType(), True),
        StructField("task_label", StringType(), True),
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
    df = clean_channel_names(df)
    print(f'Lecture: {df.count():,} lignes, {len(df.columns)} colonnes')

    # Feature engineering
    df_epochs = add_epochs(df)
    df_features = build_band_features(df_epochs)
    df_features = add_lateralization_features(df_features)
    feature_cols = extract_feature_cols(df_features)
    df_norm = normalize_by_subject(df_features, feature_cols, t0_weight=T0_WEIGHT)
    df_norm.cache()

    n_epochs = df_norm.count()
    print(f'✓ Features: {n_epochs:,} epochs × {len(feature_cols)} features')

    # Sauvegarde
    os.makedirs(FEATURES_DIR, exist_ok=True)
    df_norm.write.mode('overwrite').parquet(FEATURES_DIR)
    print(f'✓ Sauvegardé dans {FEATURES_DIR}')

    spark.stop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
