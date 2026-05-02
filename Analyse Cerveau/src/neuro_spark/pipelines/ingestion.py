"""
Pipeline d'ingestion : conversion EDF → Parquet.
Extrait du notebook poc_eeg.ipynb — rendu exécutable standalone.
"""
import os
import sys
import glob
import pandas as pd
import numpy as np
import mne

mne.set_log_level('WARNING')

# ── Chemins (lus depuis le Data Catalog YAML ou variables d'env) ──
EDF_DIR = os.environ.get('RAW_EEG_DIR', '/opt/spark/data/eeg/')
PARQUET_DIR = os.environ.get('PARQUET_DIR', '/opt/spark/data/parquet/')
MOTOR_RUNS = [f'R{i:02d}' for i in range(3, 15)]  # R03 à R14


def edf_to_dataframe(edf_path, subject_id, run_id):
    """
    Lit un fichier EDF et retourne un DataFrame avec :
    subject_id | run_id | time | task_label | 64 canaux EEG

    Le task_label est extrait des annotations EDF :
    T0 = repos, T1 = imagerie main gauche, T2 = imagerie main droite
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']  # 160 Hz
    data, times = raw.get_data(return_times=True)

    df = pd.DataFrame(data.T, columns=raw.ch_names)
    df['time'] = times
    df['subject_id'] = subject_id
    df['run_id'] = run_id

    # Labels : T0 par défaut, mis à jour selon les annotations
    df['task_label'] = 'T0'
    if raw.annotations is not None:
        for onset, duration, description in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description
        ):
            if description in ('T1', 'T2'):
                mask = (df['time'] >= onset) & (df['time'] < onset + duration)
                df.loc[mask, 'task_label'] = description

    return df


def main():
    os.makedirs(PARQUET_DIR, exist_ok=True)

    converted = 0
    errors = 0

    subjects = sorted(os.listdir(EDF_DIR))
    print(f'Sujets trouvés : {len(subjects)}')

    for subject in subjects:
        subject_dir = os.path.join(EDF_DIR, subject)
        if not os.path.isdir(subject_dir):
            continue

        for run in MOTOR_RUNS:
            edf_file = os.path.join(subject_dir, f'{subject}{run}.edf')
            parquet_file = os.path.join(PARQUET_DIR, f'{subject}_{run}.parquet')

            if not os.path.exists(edf_file):
                continue
            if os.path.exists(parquet_file):
                continue  # Déjà converti

            try:
                df = edf_to_dataframe(edf_file, subject, run)
                df.to_parquet(parquet_file, index=False)
                converted += 1
            except Exception as e:
                errors += 1
                print(f'Erreur {subject}/{run} : {e}')

        print(f'{subject} converti ({converted} fichiers au total)', end='\r')

    print(f'\n✓ Conversion terminée : {converted} fichiers Parquet créés, {errors} erreurs')
    return 0 if errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
