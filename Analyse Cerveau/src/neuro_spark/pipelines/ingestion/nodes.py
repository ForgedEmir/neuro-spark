from neuro_spark.core import batch_convert_edf


def ingest_edf(edf_dir: str, parquet_dir: str) -> dict:
    """Convertit les fichiers EDF en Parquet partitionné (idempotent)."""
    stats = batch_convert_edf(edf_dir, parquet_dir)
    print(f"✓ Ingestion : {stats['converted']} convertis, "
          f"{stats['skipped']} ignorés, {stats['errors']} erreurs")
    return stats
