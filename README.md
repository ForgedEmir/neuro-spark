# NeuroSpark — EEG Motor Imagery Pipeline

Pipeline Big Data pour la classification d'imagerie motrice à partir de signaux EEG (PhysioNet Motor Movement/Imagery dataset, 109 sujets, 1308 fichiers EDF).

Le projet illustre une chaîne complète **ingestion → features → entraînement → évaluation → dashboard** avec trois outils du quotidien des équipes Data : **Kedro**, **MLflow** et **DVC**.

## Pourquoi ces trois outils

Le notebook initial (`warehouse/etape1_poc/poc_eeg.ipynb`) faisait tout dans un seul fichier. C'était un PoC, pas un projet réutilisable. Voici ce que chaque outil apporte concrètement :

### Kedro — structure et orchestration

Le notebook a été découpé en cinq pipelines indépendantes (`ingestion`, `features`, `training`, `evaluation`, `export`). Chaque pipeline est un package Python avec ses nodes et son DAG. Le bénéfice :

- **Séparation config / code** : les hyperparamètres et chemins sont dans `conf/base/parameters.yml`, le code n'a pas de magie cachée
- **Catalog déclaratif** (`conf/base/catalog.yml`) : les datasets Spark/JSON sont définis une fois et appelés par nom dans les nodes
- **Visualisation du DAG** (`make kedro-viz` → http://localhost:4141)
- **Run partiel** : `kedro run --pipeline=features` pour tester une seule étape

### MLflow — traçabilité des expériences ML

À chaque entraînement, MLflow enregistre automatiquement :
- les **hyperparamètres** (`num_trees_grid`, `max_depth_grid`, `cv_folds`)
- les **métriques sur le test set** (`test_accuracy`, `test_f1`, `test_precision_weighted`, `test_recall_weighted`)
- le **modèle** sérialisé (artifact)
- les **runs internes** de la CrossValidation (4 folds par exemple)

Backend SQLite (`mlruns/mlflow.db`) — recommandé par la doc MLflow car le backend fichier est déprécié.

UI : `make mlflow-ui` → http://localhost:5000 → onglet **Experiments** → `neuro-spark-eeg-v2`.

### DVC — versioning des données

Le dataset EEG fait plusieurs Go, impossible à mettre dans Git. DVC le verse dans un cache local (et un remote S3/GCS si besoin) et garde un hash de chaque sortie de pipeline. Conséquence :

- `dvc repro` ne relance **que les étapes dont les inputs ont changé** (code, data, params)
- `dvc dag` montre le graphe des dépendances entre étapes
- `dvc.lock` enregistre les hashes courants — c'est ce qu'on commit dans Git, pas les données elles-mêmes

Pattern utilisé : chaque stage DVC appelle un `make` correspondant à un pipeline Kedro. Les deux outils ne se chevauchent pas, ils se complètent (orchestration côté Kedro, invalidation cache côté DVC).

## Lancer le projet

Prérequis : Linux, Java 17, Python 3.11+, environnement virtuel `.venv`.

```bash
# Installation
.venv/bin/pip install -r requirements.txt

# Téléchargement du dataset (1308 fichiers EDF, ~3 Go)
make download

# Pipeline complète
make run

# Ou étape par étape
make ingest      # EDF -> Parquet partitionné
make features    # FFT band power + normalisation Z-score
make train       # CrossValidator RandomForest + tracking MLflow
make evaluate    # Métriques sur test set
make export      # Données pour le dashboard
```

## Visualisations

```bash
make kedro-viz       # DAG Kedro      -> http://localhost:4141
make mlflow-ui       # Tracking ML    -> http://localhost:5000
make mlflow-report   # PNG comparant les runs -> data/mlflow_report.png
make dashboard       # Dashboard Dash -> http://localhost:8050
```

## DVC

```bash
make dvc-dag         # Graphe des stages
make dvc-status      # État du cache (skip si à jour)
make dvc-repro       # Rejouer en sautant ce qui n a pas changé
```

## Structure

```
src/neuro_spark/
├── core.py                       # Algos EEG : FFT, normalisation, MLlib pipeline
├── hooks.py                      # SparkSession (config dans conf/base/spark.yml)
├── settings.py                   # Configuration Kedro
├── pipeline_registry.py          # Enregistre les 5 pipelines
└── pipelines/
    ├── ingestion/                # EDF -> Parquet (un fichier par sujet/run)
    ├── features/                 # FFT theta/alpha/beta/gamma + lateralization
    ├── training/                 # CrossValidator + log MLflow
    ├── evaluation/               # Charge le modèle, predict, metrics
    └── export_dashboard/         # Échantillons Parquet pour le dashboard

conf/base/
├── catalog.yml                   # SparkDataset, JSONDataset
├── parameters.yml                # Hyperparamètres ML, chemins, bandes EEG
├── spark.yml                     # spark.executor.memory, arrow, etc.
└── logging.yml                   # Config rich logging

scripts/
├── download_eeg.py               # Récupère le dataset PhysioNet
└── mlflow_report.py              # Génère le rapport graphique des runs

dvc.yaml                          # Définit les 6 stages DVC qui appellent make
dashboard.py                      # Dashboard Dash (visualisation EEG, prédictions)
```

## Tests

```bash
make test            # pytest
make lint            # ruff (style)
```

## Données

Le pipeline manipule (à grands traits) :
- **25 millions de lignes** de signal brut (1308 fichiers EDF, 64 canaux, 160 Hz)
- **80 000 epochs** de 2 secondes après découpage temporel
- **16 features** par epoch (4 bandes × 3 canaux moteurs C3/Cz/C4 + 4 lateralization)

Tâches : repos (T0), imagerie de la main gauche (T1), imagerie de la main droite (T2). Accuracy actuelle ~44 % sur 3 classes (random = 33 %).

## Licence

MIT
