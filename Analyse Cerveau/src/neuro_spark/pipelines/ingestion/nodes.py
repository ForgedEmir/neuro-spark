import logging
from neuro_spark.core import batch_convert_edf

log = logging.getLogger(__name__)


def ingest_edf(edf_dir: str, parquet_dir: str) -> dict:
    """Convert EDF files to partitioned Parquet. Skips already converted files."""
    stats = batch_convert_edf(edf_dir, parquet_dir)
    log.info(
        "Ingestion done: %d converted, %d skipped, %d errors",
        stats["converted"], stats["skipped"], stats["errors"],
    )
    return stats
