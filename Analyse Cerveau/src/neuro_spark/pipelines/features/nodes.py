from pyspark.sql import DataFrame
from neuro_spark.core import (
    clean_channel_names,
    add_epochs,
    build_band_features,
    add_lateralization_features,
    normalize_by_subject,
    extract_feature_cols,
)


def build_features(parquet_data: DataFrame, t0_weight: float) -> DataFrame:
    """FFT par bande de fréquences + normalisation Z-score par sujet."""
    df = clean_channel_names(parquet_data)
    print(f"Lecture : {df.count():,} lignes, {len(df.columns)} colonnes")

    df_epochs = add_epochs(df)
    df_features = build_band_features(df_epochs)
    df_features = add_lateralization_features(df_features)
    feature_cols = extract_feature_cols(df_features)
    df_norm = normalize_by_subject(df_features, feature_cols, t0_weight=t0_weight)

    n = df_norm.count()
    print(f"✓ Features : {n:,} epochs × {len(feature_cols)} features")
    return df_norm
