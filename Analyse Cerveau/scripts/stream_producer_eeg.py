"""Producteur de streaming EEG : rejoue les Parquet PhysioNet en micro-batches.

Chaque epoch de 2 secondes (= 320 samples @ 160 Hz) devient un fichier Parquet
déposé dans `data/stream_eeg/input/`. Spark le lit via file source streaming
et calcule en temps réel :
  - la prédiction BCI (T0/T1/T2) via le modèle Random Forest déjà entraîné
  - la puissance alpha (8-13 Hz) par électrode → topographie cérébrale

Usage :
    python scripts/stream_producer_eeg.py --subject S001 --run R04 --rate 0.5
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PARQUET_DIR = Path("data/parquet")
STREAM_INPUT = Path("data/stream_eeg/input")
EEG_FS = 160                # Hz
EPOCH_SEC = 2               # une fenêtre = 2 secondes = 320 samples


def load_run(subject: str, run: str) -> pd.DataFrame:
    """Charge un (subject, run) depuis le Parquet ingéré."""
    path = PARQUET_DIR / f"{subject}_{run}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Lance d'abord `make ingest` pour générer les Parquet."
        )
    df = pd.read_parquet(path)
    log.info("Loaded %s/%s : %d samples (%.1fs), %d channels",
             subject, run, len(df), len(df) / EEG_FS, len(df.columns) - 4)
    return df


def chunk_epochs(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Découpe en epochs de 2 s (= 320 samples). Chaque epoch garde un label majoritaire."""
    samples_per_epoch = EEG_FS * EPOCH_SEC
    n_epochs = len(df) // samples_per_epoch
    epochs = []
    for i in range(n_epochs):
        chunk = df.iloc[i * samples_per_epoch:(i + 1) * samples_per_epoch].copy()
        # label majoritaire (l'epoch peut chevaucher 2 task_label)
        chunk["epoch_id"] = i
        chunk["epoch_label"] = chunk["task_label"].mode().iloc[0]
        epochs.append(chunk)
    return epochs


def stream_epochs(
    epochs: list[pd.DataFrame], rate: float, duration_sec: int,
) -> None:
    """Écrit les epochs dans STREAM_INPUT au rythme `rate` epochs/seconde."""
    STREAM_INPUT.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp.now(tz="UTC")
    interval = 1.0 / rate

    max_epochs = min(len(epochs), int(duration_sec * rate))
    log.info("Streaming %d epochs (rate=%.1f Hz, total ~%ds wall-clock)",
             max_epochs, rate, int(max_epochs / rate))

    # cast tous les channels EEG en float64 pour cohérence avec Spark
    channel_cols = None
    for i in range(max_epochs):
        epoch = epochs[i].copy()
        if channel_cols is None:
            channel_cols = [c for c in epoch.columns
                            if c not in {"subject_id", "run_id", "time",
                                         "task_label", "epoch_id", "epoch_label"}]
        epoch[channel_cols] = epoch[channel_cols].astype(np.float64)
        epoch["time"] = epoch["time"].astype(np.float64)

        # event_time : un timestamp par epoch (le centre de l'epoch)
        ts = start_ts + pd.Timedelta(seconds=i * EPOCH_SEC)
        epoch["event_time"] = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")

        fname = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.parquet"
        epoch.to_parquet(STREAM_INPUT / fname, index=False)

        if i % 5 == 0:
            label = epoch["epoch_label"].iloc[0]
            log.info("  epoch %d/%d → label=%s, ts=%s",
                     i, max_epochs, label, ts.strftime("%H:%M:%S"))
        time.sleep(interval)

    log.info("Streaming EEG terminé (%d epochs envoyées)", max_epochs)


def reset_stream_dir() -> None:
    for p in [STREAM_INPUT,
              Path("data/stream_eeg/checkpoint"),
              Path("data/stream_eeg/output")]:
        if p.exists():
            shutil.rmtree(p)
            log.info("rm -rf %s", p)
    STREAM_INPUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="S001")
    p.add_argument("--run", default="R04",
                   help="Run ID (R03-R14). R03/R07/R11=hands, R04/R08/R12=imagined hands.")
    p.add_argument("--rate", type=float, default=0.5,
                   help="Epochs/sec (0.5 = un epoch toutes les 2s, conforme à la vraie BCI)")
    p.add_argument("--duration", type=int, default=300,
                   help="Durée max en secondes (défaut 5 min)")
    p.add_argument("--reset", action="store_true")
    p.add_argument("--loop", action="store_true",
                   help="Boucler à la fin du run (utile pour démos longues)")
    args = p.parse_args()

    if args.reset:
        reset_stream_dir()

    df = load_run(args.subject, args.run)
    epochs = chunk_epochs(df)
    log.info("%d epochs disponibles dans %s/%s", len(epochs), args.subject, args.run)

    if args.loop:
        # répète les epochs jusqu'à atteindre la durée
        repeats = (int(args.duration * args.rate) // len(epochs)) + 1
        epochs = epochs * repeats
        log.info("Mode loop : %d epochs après répétition x%d", len(epochs), repeats)

    stream_epochs(epochs, args.rate, args.duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
