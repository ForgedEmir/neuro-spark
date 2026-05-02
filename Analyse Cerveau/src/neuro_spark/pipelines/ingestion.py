"""
Pipeline d'ingestion : EDF → Parquet.
Utilise les fonctions extraites du notebook poc_eeg.ipynb.
"""
import os
import sys

# Ajouter le parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_spark.core import batch_convert_edf

EDF_DIR = os.environ.get('RAW_EEG_DIR', '/opt/spark/data/eeg/')
PARQUET_DIR = os.environ.get('PARQUET_DIR', '/opt/spark/data/parquet/')


def main():
    stats = batch_convert_edf(EDF_DIR, PARQUET_DIR)
    print(f"\n✓ Ingestion terminée : {stats['converted']} convertis, "
          f"{stats['skipped']} ignorés, {stats['errors']} erreurs")
    return 0 if stats['errors'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
