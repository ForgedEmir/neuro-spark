import logging
from pyspark.sql import DataFrame
from neuro_spark.core import (
    clean_channel_names,
    add_epochs,
    build_band_features,
    add_lateralization_features,
    normalize_by_subject,
    extract_feature_cols,
)

log = logging.getLogger(__name__)


def build_features(parquet_data: DataFrame, t0_weight: float) -> DataFrame:
    """Compute FFT band power features and Z-score normalization per subject."""
    df = clean_channel_names(parquet_data)
    log.info("Input: %d rows, %d columns", df.count(), len(df.columns))

    df_epochs = add_epochs(df)
    df_features = build_band_features(df_epochs)
    df_features = add_lateralization_features(df_features)
    feature_cols = extract_feature_cols(df_features)
    df_norm = normalize_by_subject(df_features, feature_cols, t0_weight=t0_weight)

    log.info("Features done: %d epochs x %d features", df_norm.count(), len(feature_cols))
    return df_norm
