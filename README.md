<p align="center">
  <img src="assets/neuro-spark-demo.gif" alt="NeuroSpark Demo" width="720"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/spark-3.5.5-E25A1C?style=flat&logo=apachespark&logoColor=white" alt="Spark"/>
  <img src="https://img.shields.io/badge/python-3.11-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/MLflow-tracking-0194E2?style=flat&logo=mlflow&logoColor=white" alt="MLflow"/>
  <img src="https://img.shields.io/badge/DVC-versioning-945DD6?style=flat&logo=dvc&logoColor=white" alt="DVC"/>
  <img src="https://img.shields.io/badge/docker-spark--cluster-2496ED?style=flat&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat" alt="License"/>
</p>

<h1 align="center">NeuroSpark</h1>
<p align="center"><em>Distributed EEG Motor Imagery Decoding — from raw neural signals to classified thoughts</em></p>

---

## What

A distributed Big Data pipeline that decodes **imagined hand movements** from EEG brain signals. 66 subjects, 64 electrodes, 160 Hz — ~15 million data points processed across a Spark cluster.

The core question: *can a distributed system distinguish between thinking about moving your left hand vs. your right hand — just from brainwaves?*

**Pipeline:** `Raw EDF → Parquet → FFT Spectral Features → CrossValidator RandomForest → Interactive Dashboard`

---

## Why it matters

Motor imagery is the foundation of Brain-Computer Interfaces — systems that let people control devices with thought alone. The neurophysiological phenomenon at play is **Event-Related Desynchronization (ERD)**: when you imagine a movement, alpha waves (~8–13 Hz) decrease over the contralateral motor cortex. This pipeline measures that.

> *"The brain does not distinguish between a real experience and one that is vividly imagined."* — Bruce Lipton, *The Biology of Belief*

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Docker Cluster                        │
│                                                        │
│  ┌──────────────┐    ┌──────────────┐                 │
│  │ spark-master  │    │ spark-worker  │                │
│  │ + Jupyter     │───▶│ (20 GB RAM)   │                │
│  │ :8889         │    │               │                 │
│  └──────┬───────┘    └──────┬────────┘                 │
│         │                    │                          │
│         ▼                    ▼                          │
│  ┌──────────────────────────────────┐                  │
│  │  /opt/spark/data/  (shared)       │                  │
│  │  ├── eeg/       (EDF raw)         │                  │
│  │  ├── parquet/   (783 files)       │                  │
│  │  ├── features/  (FFT band power)  │                  │
│  │  ├── models/    (RF + CV)         │                  │
│  │  └── dashboard/ (Dash data)       │                  │
│  └──────────────────────────────────┘                  │
│                                                        │
│  ┌──────────────┐    ┌──────────────┐                 │
│  │ spark-history │    │  Dashboard    │                 │
│  │ :18080        │    │  :8050        │                 │
│  └──────────────┘    └──────────────┘                 │
│  ┌──────────────┐                                     │
│  │ MLflow UI     │                                     │
│  │ :5000         │                                     │
│  └──────────────┘                                     │
└──────────────────────────────────────────────────────┘
```

---

## Pipeline

| Stage | What | Tech |
|---|---|---|
| **A. Ingestion** | EDF medical format → Parquet (columnar, compressed) | MNE-Python |
| **B. Feature Engineering** | 2s windows → FFT → band power (θ/α/β/γ) × C3/Cz/C4 | PySpark UDF + NumPy |
| **C. Normalization** | Z-score per subject (eliminates inter-subject amplitude bias) | Spark Window |
| **D. Classification** | RandomForest with CrossValidator (4 combos × 3 folds = 12 runs) | Spark MLlib |
| **E. Tracking** | Auto-log params, metrics, model to MLflow | MLflow |
| **F. Dashboard** | Interactive brain topography, confusion matrix, band analysis | Dash/Plotly |

---

## Results

| Version | Features | Class Weight | Accuracy | Insight |
|---|---|---|---|---|
| Baseline (always T0) | — | — | ~50% | Useless — class imbalance trap |
| RF naïve | 192 (64 ch) | None | 51.78% | Same problem — confusion matrix reveals all |
| RF aggressive | 192 | 0.33 (T0) | 30.93% | Over-corrected — worse than chance |
| **RF motor** | 9 (C3/Cz/C4) | 0.5 | 41.82% | Beat random (33%) |
| **RF + diff + norm + CV** | 16 (4 bands + diff) | 0.5 | **45.17%** | **Final** — subject-independent split |

**Per-class recall (final model, unseen subjects S053–S066):**

| Class | Recall | Interpretation |
|---|---|---|
| T0 — Rest | 62.8% | Well recognized — distinct background signal |
| T1 — Left Hand | 32.9% | Often confused with rest |
| T2 — Right Hand | 19.8% | Hardest — patterns very close to T1 |

Research papers on this dataset achieve 70–85% with CSP + LSTM. As a PoC validating the full distributed pipeline, 45% against 33% random baseline is solid.

---

## Dashboard

Interactive neuro-imaging dashboard (Dash/Plotly):

| Panel | What it shows |
|---|---|
| **Brain Topography** | Spatial activity map — see C3/C4 invert with imagined hand |
| **Band Power Analysis** | θ/α/β/γ decomposition per channel per task |
| **Confusion Matrix** | T0/T1/T2 predicted vs. actual |
| **Recall per Class** | vs. 33% random baseline |
| **Feature Importance** | Which bands/channels drive classification |

<p align="center">
  <em>Clinical warm palette — Apple Health / Oura inspired</em>
</p>

---

## Quick Start

```bash
git clone https://github.com/ForgedEmir/neuro-spark.git
cd neuro-spark

# Launch Spark cluster
cd "Analyse Cerveau"
docker-compose up -d

# Open Jupyter → http://localhost:8889
# Run poc_eeg.ipynb (Parts A–F)

# Or run as standalone pipelines:
python src/neuro_spark/pipelines/ingestion.py
python src/neuro_spark/pipelines/features.py
python src/neuro_spark/pipelines/training.py
python src/neuro_spark/pipelines/evaluation.py
python src/neuro_spark/pipelines/export_dashboard.py

# Launch dashboard → http://localhost:8050
python dashboard.py

# MLflow UI → http://localhost:5000
mlflow ui --backend-store-uri /opt/spark/mlruns
```

---

## Project Structure

```
neuro-spark/
├── Analyse Cerveau/
│   ├── warehouse/etape1_poc/
│   │   ├── poc_eeg.ipynb              # Main pipeline notebook (36 cells)
│   │   └── poc_learn.ipynb            # Exploratory notebook
│   ├── src/neuro_spark/
│   │   ├── core.py                     # Extracted functions (reusable)
│   │   └── pipelines/
│   │       ├── ingestion.py            # EDF → Parquet
│   │       ├── features.py             # FFT + normalization
│   │       ├── training.py             # CrossValidator + MLflow
│   │       ├── evaluation.py           # Metrics
│   │       └── export_dashboard.py     # Dashboard data
│   ├── conf/base/
│   │   ├── catalog.yml                 # Data paths (Kedro-inspired)
│   │   └── parameters.yml             # Hyperparameters
│   ├── scripts/download_eeg.py         # PhysioNet downloader
│   ├── dashboard.py                    # Dash/Plotly app (924 lines)
│   ├── assets/style.css                # External CSS (Dash auto-loads)
│   ├── rapport_reflexif_poc_eeg.md     # Reflexive report (FR)
│   ├── Dockerfile                      # Spark 3.5.5 + Jupyter + MLflow
│   ├── docker-compose.yml              # 3-node cluster
│   └── entrypoint.sh
├── dvc.yaml                            # DVC pipeline (6 stages)
├── requirements.txt
├── assets/                             # Screenshots, demo video
└── README.md
```

---

## Key Learnings

- **Lazy evaluation** — Spark builds a DAG, only computes on action (`.count()`, `.show()`)
- **Parquet > CSV** — Columnar format: 10-100x faster for analytical queries  
- **Accuracy is a trap** — 50% accuracy on imbalanced data = predicting the majority class
- **Subject-based split** — Random split leaks subject identity; split by subject for real generalization
- **ERD** — Alpha power drops in contralateral motor cortex during motor imagery — measurable, real
- **CrossValidator** — 12 trainings (4 combos × 3 folds) automates hyperparameter tuning
- **MLflow** — One `autolog()` call tracks everything: params, metrics, model, environment

---

## Challenges Solved

| Problem | Solution |
|---|---|
| Spark can't read `.edf` | MNE-Python conversion to Parquet via UDFs |
| Column names like `C3..` break Spark | Systematic renaming (strip dots) |
| Class imbalance (T0 = 50%) | Inverse frequency weighting + CrossValidator |
| Inter-subject amplitude variance | Z-score normalization per subject (Spark Window) |
| FFT on distributed data | Pandas UDF with sorted `collect_list` |
| HTML/CSS in Python strings | External `assets/style.css` loaded by Dash |
| `inferSchema` suboptimal | Explicit `StructType` with 64 channel types |
| `toPandas().to_parquet()` | `spark.write.parquet()` — stays distributed |

---

## Stack

`Python 3.11` · `PySpark 3.5.5` · `Spark MLlib` · `MNE 1.8` · `NumPy` · `Pandas` · `Plotly` · `Dash` · `MLflow 2.20` · `DVC` · `Docker` · `Parquet` · `FFT`

---

## References

- [PhysioNet EEG-MMID Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
- [MNE-Python](https://mne.tools/)
- [PySpark MLlib](https://spark.apache.org/mllib/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [DVC](https://dvc.org/)
- [Kedro](https://kedro.org/) (philosophy, not dependency)
- Lipton, B. — *The Biology of Belief* (2005)

---

## License

MIT © Emir Makhtsaev
