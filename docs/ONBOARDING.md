# NeuroSpark — Guide d'onboarding

> Tu viens d'arriver sur le projet. Ce document te donne tout ce qu'il faut pour comprendre ce qu'on fait, pourquoi, et comment le code est organisé. Lis-le dans l'ordre.

---

## Table des matières

1. [C'est quoi ce projet ?](#1-cest-quoi-ce-projet-)
2. [La science : EEG et imagerie motrice](#2-la-science--eeg-et-imagerie-motrice)
3. [Le dataset PhysioNet](#3-le-dataset-physionet)
4. [Les features : du signal brut aux 16 colonnes ML](#4-les-features--du-signal-brut-aux-16-colonnes-ml)
5. [Architecture du code](#5-architecture-du-code)
6. [Les outils : Kedro, MLflow, DVC](#6-les-outils--kedro-mlflow-dvc)
7. [Optimisations Spark](#7-optimisations-spark)
8. [Streaming EEG en temps réel](#8-streaming-eeg-en-temps-réel)
9. [Résultats et interprétation](#9-résultats-et-interprétation)
10. [Lancer le projet](#10-lancer-le-projet)
11. [Glossaire complet](#11-glossaire-complet)

---

## 1. C'est quoi ce projet ?

**NeuroSpark** est un pipeline Big Data qui prend des signaux cérébraux (EEG) et essaie de deviner ce que la personne est en train d'imaginer faire — bouger la main gauche, la main droite, ou ne rien faire du tout.

Ce n'est pas une démonstration théorique. Les données sont réelles : 109 personnes ont porté un casque EEG pendant qu'on leur demandait d'imaginer des mouvements. Le projet traite ces données avec Apache Spark et entraîne un modèle de classification.

**Pourquoi c'est intéressant ?** Parce que quand tu *imagines* bouger ta main, ton cerveau envoie quand même les signaux électriques — comme si tu le faisais vraiment. C'est le principe des interfaces cerveau-machine (BCI) : piloter un curseur, une prothèse, ou un fauteuil roulant uniquement par la pensée.

**Ce que le pipeline fait de bout en bout :**

```
Fichiers EDF (format médical)
    ↓  ingestion
Parquet partitionné par sujet/run
    ↓  features
16 features FFT par epoch (fenêtre de 2s)
    ↓  training
RandomForest avec CrossValidation + MLflow
    ↓  evaluation
Métriques sur sujets jamais vus
    ↓  streaming
Prédiction en temps réel sur flux EEG
    ↓  dashboard
Visualisation live (topographie cérébrale, probabilités)
```

---

## 2. La science : EEG et imagerie motrice

### Qu'est-ce qu'un EEG ?

Un **électroencéphalogramme (EEG)** enregistre l'activité électrique du cerveau via des électrodes posées sur le crâne. Les neurones qui s'activent ensemble génèrent de petits champs électriques (quelques microvolts, µV) mesurables à la surface.

Concrètement : **N électrodes × temps × amplitude**. Avec 64 électrodes à 160 Hz pendant 1 minute :
```
64 canaux × 60 s × 160 Hz = 614 400 valeurs
```

### Pourquoi C3, Cz, C4 et pas les 64 canaux ?

Le cerveau humain est **contralatéral** : l'hémisphère gauche contrôle le côté droit du corps, et inversement.

```
       Main GAUCHE imaginée        Main DROITE imaginée
              ↓                              ↓
     C4 (hémisphère droit)        C3 (hémisphère gauche)
```

Dans le **système international 10-20** (placement standardisé des électrodes) :
- **C3** — au-dessus du cortex moteur **gauche** → réagit quand on imagine bouger la main droite
- **Cz** — au centre du crâne (vertex), référence neutre
- **C4** — au-dessus du cortex moteur **droit** → réagit quand on imagine bouger la main gauche

Les 61 autres canaux captent des zones (frontale, occipitale, temporale) qui ne sont pas impliquées dans l'imagerie motrice des mains. On les ignore.

### ERD et ERS : ce qu'on cherche à détecter

Deux phénomènes se produisent pendant l'imagerie motrice :

**ERD — Event-Related Desynchronization**
La bande **alpha (8–13 Hz)** *baisse* au-dessus du cortex moteur actif pendant l'imagerie. Le cerveau "sort" de son état de repos électrique pour simuler le mouvement.

**ERS — Event-Related Synchronization**
La bande **beta (13–30 Hz)** *remonte* après le mouvement (rebond post-mouvement).

Exemple concret :
- Sujet imagine bouger la main droite → **C3_alpha baisse**, C4_alpha reste stable
- Sujet imagine bouger la main gauche → **C4_alpha baisse**, C3_alpha reste stable
- Sujet au repos → alpha haut partout

C'est exactement ce signal qu'on essaie d'extraire et de classifier.

### Les bandes de fréquence

| Bande | Plage (Hz) | Rôle neurologique |
|-------|-----------|-------------------|
| **theta** | 4 – 8 | Préparation motrice, mémoire de travail |
| **alpha** | 8 – 13 | **Se désynchronise pendant l'imagerie (ERD)** — feature clé |
| **beta** | 13 – 30 | **Rebond post-mouvement (ERS)** — feature secondaire |
| **gamma** | 30 – 80 | Traitement cognitif fin, attention focalisée |

On ignore < 4 Hz (delta, artefacts oculaires) et > 80 Hz (bruit musculaire, secteur électrique).

---

## 3. Le dataset PhysioNet

### Origine

Dataset **EEG Motor Movement/Imagery** — PhysioNet / BCI2000 (Schalk et al., 2004). C'est la référence historique en BCI, utilisée dans des centaines de publications.

- **109 sujets** (on en utilise ~66 après filtrage des fichiers corrompus)
- **64 électrodes EEG** à **160 Hz**
- **14 runs par sujet** (~2 min chacun)
- Format : **EDF** (European Data Format) — format binaire standard en médecine

### Les runs : quoi est quoi

| Runs | Contenu | Utilisé ? |
|------|---------|-----------|
| R01, R02 | Baseline (yeux ouverts / fermés) | Non |
| R03, R07, R11 | Ouvrir/fermer poing G ou D — **réel** | Oui |
| R04, R08, R12 | **Imaginer** ouvrir/fermer poing G ou D | Oui |
| R05, R09, R13 | Ouvrir/fermer deux poings ou deux pieds — réel | Oui |
| R06, R10, R14 | **Imaginer** deux poings ou deux pieds | Oui |

On utilise **R03 à R14** (12 runs moteurs par sujet).

### Structure d'un fichier EDF

```
┌─────────────────────────────────────────┐
│  Header                                 │
│  - 64 canaux, 160 Hz                    │
│  - Noms des canaux : "C3..", "Cz.."     │
├─────────────────────────────────────────┤
│  Annotations (labels temporels)         │
│  - T0 de 0.0s à 4.2s (repos)           │
│  - T1 de 4.2s à 8.3s (main gauche)     │
│  - T2 de 12.5s à 16.7s (main droite)   │
├─────────────────────────────────────────┤
│  Data                                   │
│  - 64 canaux × ~20 000 samples         │
│  - valeurs en Volts (ordre µV)          │
└─────────────────────────────────────────┘
```

### Ce que devient une ligne après ingestion

| subject_id | run_id | time | task_label | C3 | Cz | C4 | (61 autres) |
|-----------|--------|------|-----------|-----|-----|-----|-------------|
| S001 | R03 | 0.000 | T0 | 0.000412 | -0.000091 | 0.000065 | … |
| S001 | R03 | 0.006 | T0 | 0.000398 | -0.000103 | 0.000071 | … |
| S001 | R03 | 4.200 | T1 | … | … | … | … |

**À l'échelle du projet :** ~15 millions de lignes, 68 colonnes. C'est là que Spark devient nécessaire — Pandas crasherait en RAM.

### Limitation importante des labels

Le sens de T1 et T2 **dépend du run** :
- Dans R04 : T1 = imaginer main gauche, T2 = imaginer main droite
- Dans R06 : T1 = imaginer deux poings, T2 = imaginer deux pieds

Le pipeline les traite comme des classes unifiées, ce qui introduit de l'ambiguïté dans les labels et explique en partie la performance limitée (~44% accuracy). C'est une limite documentée, pas un bug.

---

## 4. Les features : du signal brut aux 16 colonnes ML

### Pourquoi pas donner le signal brut au modèle ?

Un epoch de 2 secondes = **320 samples × 64 canaux = 20 480 valeurs**. Donner ça brut à un RandomForest, c'est :
1. Noyer le modèle (malédiction de la dimensionnalité)
2. Perdre la structure fréquentielle (l'info utile est dans le spectre, pas la forme d'onde)
3. Exploser en temps de calcul

**Solution : FFT sur fenêtres de 2 secondes.** On résume chaque epoch en 16 scalaires.

### Étape 1 : Découpage en epochs

```python
# floor(time / 2) → epoch_id
# time=0.000 → epoch_id=0, time=2.000 → epoch_id=1, etc.
df.withColumn("epoch_id", F.floor(F.col("time") / 2).cast("int"))
```

Pourquoi 2 secondes ?
- **Assez long** : résolution fréquentielle de 0.5 Hz (suffisant pour distinguer alpha de beta)
- **Assez court** : 2 epochs par essai d'imagerie → plus d'exemples d'entraînement

### Étape 2 : FFT par bande (Pandas UDF)

```python
# Pour chaque epoch et chaque canal (C3, Cz, C4) :
freqs = np.fft.rfftfreq(320, 1/160)       # fréquences en Hz
power = np.abs(np.fft.rfft(signal)) ** 2  # puissance spectrale
mask  = (freqs >= 8) & (freqs < 13)       # bande alpha
alpha_power = np.mean(power[mask])         # scalaire
```

**Piège critique** : `collect_list()` dans Spark ne garantit pas l'ordre temporel après un shuffle. Si les 320 samples arrivent dans le désordre, la FFT calcule des fréquences fictives. La solution :

```python
# Collecte des structs (time, valeur) → tri par time → extraction des valeurs
F.transform(F.array_sort(F.collect_list(F.struct("time", canal))), lambda s: s["v"])
```

### Les 16 features

**12 features spectrales (3 canaux × 4 bandes) :**

| Feature | Sens |
|---------|------|
| `C3_theta` | Préparation motrice hémisphère gauche |
| `C3_alpha` | **↓ pendant imagerie main droite (ERD)** |
| `C3_beta` | Rebond post-mouvement hémisphère gauche |
| `C3_gamma` | Activité cognitive fine hémisphère gauche |
| `Cz_theta` | Préparation centrale |
| `Cz_alpha` | Repos central |
| `Cz_beta` | Rebond central |
| `Cz_gamma` | Cognition centrale |
| `C4_theta` | Préparation motrice hémisphère droit |
| `C4_alpha` | **↓ pendant imagerie main gauche (ERD)** |
| `C4_beta` | Rebond post-mouvement hémisphère droit |
| `C4_gamma` | Activité cognitive fine hémisphère droit |

**4 features de latéralisation (diff C3 − C4) :**

| Feature | Formule | Sens |
|---------|---------|------|
| `diff_theta` | C3_theta − C4_theta | Asymétrie préparation motrice |
| `diff_alpha` | C3_alpha − C4_alpha | **Feature reine** : + si main droite, − si main gauche |
| `diff_beta` | C3_beta − C4_beta | Asymétrie rebond |
| `diff_gamma` | C3_gamma − C4_gamma | Asymétrie cognitive |

**Pourquoi `diff_alpha` est la feature la plus importante :**

```
Imaginer main droite → C3 (hémisphère gauche) s'active → C3_alpha ↓ → diff_alpha < 0
Imaginer main gauche → C4 (hémisphère droit) s'active  → C4_alpha ↓ → diff_alpha > 0
Repos               → alpha stable partout             → diff_alpha ≈ 0
```

La différence encode directement la direction de l'imagerie, là où C3 ou C4 seuls ne le font pas.

### Étape 3 : Normalisation Z-score par sujet

Chaque personne a une amplitude EEG différente (épaisseur du crâne, gel conducteur, etc.). Sans normaliser, le modèle apprend à reconnaître *qui* est le sujet, pas *ce qu'il fait*.

```python
# Pour chaque feature et chaque sujet :
feature_normalisée = (feature − moyenne_sujet) / écart_type_sujet
```

Implémentation Spark : `groupBy("subject_id").agg(mean, stddev)` + `broadcast join` (1 shuffle au lieu de 16 Window functions).

---

## 5. Architecture du code

```
src/neuro_spark/
├── core.py                     ← Toutes les fonctions réutilisables
├── hooks.py                    ← SparkSession (config dans spark.yml)
├── settings.py                 ← Configuration Kedro
├── pipeline_registry.py        ← Enregistre les 5 pipelines
└── pipelines/
    ├── ingestion/              ← EDF → Parquet partitionné
    │   ├── nodes.py            ← ingest_edf()
    │   └── pipeline.py         ← DAG Kedro
    ├── features/               ← FFT + normalisation
    │   ├── nodes.py            ← build_features()
    │   └── pipeline.py
    ├── training/               ← CrossValidator + MLflow
    │   ├── nodes.py            ← train_model()
    │   └── pipeline.py
    ├── evaluation/             ← Métriques test set
    │   ├── nodes.py            ← evaluate()
    │   └── pipeline.py
    └── streaming_eeg/          ← BCI live (Spark Structured Streaming)
        ├── nodes.py            ← run_streaming_eeg()
        └── pipeline.py

conf/base/
├── catalog.yml                 ← Datasets Spark/JSON déclarés par nom
├── parameters.yml              ← Hyperparamètres, chemins, bandes EEG
├── spark.yml                   ← Config Spark (mémoire, shuffle partitions)
└── logging.yml                 ← Niveaux de log

scripts/
├── download_eeg.py             ← Télécharge PhysioNet (~3 Go)
├── stream_producer_eeg.py      ← Envoie des epochs dans le stream
└── mlflow_report.py            ← Graphique comparatif des runs MLflow
```

### Règle d'or : `core.py` est la source de vérité

Toutes les constantes (`FS=160`, `MOTOR_CHANNELS`, `BANDS`) et toutes les fonctions (FFT, normalisation, split, pipeline MLlib) sont dans `core.py`. Les `nodes.py` de chaque pipeline **importent** depuis `core.py`, ils ne redéfinissent rien. Si tu vois une constante dupliquée ailleurs, c'est un bug.

### Séparation des responsabilités

| Module | Fait quoi |
|--------|-----------|
| `core.py` | Algorithmique pure — FFT, normalisation, MLlib, split |
| `pipelines/*/nodes.py` | Orchestre les appels à `core.py`, log les résultats |
| `conf/base/parameters.yml` | Hyperparamètres — pas de magic numbers dans le code |
| `conf/base/catalog.yml` | Chemins et formats des datasets — pas de chemins en dur |
| `hooks.py` | Crée la SparkSession depuis `spark.yml` |

### Split train/test : par sujet, pas par epoch

C'est le choix méthodologique le plus important du projet.

**Split aléatoire sur les epochs (mauvais) :**
Un même sujet peut avoir des epochs dans le train ET dans le test. Le modèle apprend le "style EEG" de chaque personne (amplitude, forme du crâne) → 85% d'accuracy factice → s'effondre en production sur un nouveau sujet.

**Split par sujet (ce qu'on fait) :**
~80% des personnes en train, ~20% en test. Au test, le modèle voit des gens jamais vus. C'est le vrai critère de généralisation.

```python
# On récupère la liste réelle des sujets présents (pas une plage hardcodée)
all_subjects = sorted([r.subject_id for r in df.select("subject_id").distinct().collect()])
```

---

## 6. Les outils : Kedro, MLflow, DVC

### Kedro — structure et orchestration

Kedro découpe le pipeline en étapes indépendantes avec un système de catalogue déclaratif.

**Catalog (`conf/base/catalog.yml`)** — les datasets sont nommés, pas des chemins en dur :
```yaml
features_data:
  type: kedro_datasets.spark.SparkDataset
  filepath: data/features/
  file_format: parquet
```

Dans le code, on écrit juste `features_data` — Kedro sait où lire/écrire.

**Lancer une étape seule :**
```bash
kedro run --pipeline features   # seulement le feature engineering
kedro run --pipeline training   # seulement l'entraînement
```

**Visualiser le DAG :**
```bash
make kedro-viz  # → http://localhost:4141
```

### MLflow — traçabilité des expériences

À chaque entraînement, MLflow enregistre automatiquement :
- Les hyperparamètres (`num_trees_grid`, `max_depth_grid`, `cv_folds`)
- Les métriques sur le test set (`test_accuracy`, `test_f1`, `test_precision_weighted`, `test_recall_weighted`)
- Le modèle sérialisé (artifact)
- L'environnement Python (versions des packages)

Backend : SQLite (`mlruns/mlflow.db`).

```bash
make mlflow-ui    # → http://localhost:5000 → onglet "neuro-spark-eeg-v2"
```

### DVC — versioning des données

Le dataset EEG fait ~3 Go. On ne le met pas dans Git. DVC suit les fichiers lourds via des hashes et permet de rejouer le pipeline uniquement si une dépendance a changé.

```bash
make dvc-repro   # rejoue tout, saute ce qui n'a pas changé
make dvc-dag     # graphe des dépendances
make dvc-status  # état du cache
```

**Pattern utilisé :** chaque stage DVC appelle un `make` correspondant à un pipeline Kedro. DVC gère l'invalidation du cache, Kedro gère l'exécution. Ils ne se chevauchent pas.

---

## 7. Optimisations Spark

### 1. Partition pruning (Hive-style)

**Avant :** fichiers plats `parquet/S001_R03.parquet` → Spark lit tout pour filtrer.

**Après :** structure `parquet/subject_id=S001/run_id=R03/` → Spark ne lit que les dossiers filtrés.

Gain : ~20-30% sur la lecture. Visible dans Spark UI → SQL → "PartitionFilters".

### 2. Normalisation : 1 agrégation au lieu de 16 Window functions

**Avant :**
```python
# 16 shuffles séparés
for fc in feature_cols:
    df = df.withColumn(fc, (col(fc) - F.mean(fc).over(w)) / F.stddev(fc).over(w))
```

**Après :**
```python
# 1 shuffle + 1 broadcast join
stats = df.groupBy("subject_id").agg(*[F.mean(c), F.stddev(c) for c in feature_cols])
df = df.join(F.broadcast(stats), "subject_id")
```

Gain : ~40-60% sur l'étape de normalisation.

### 3. CrossValidator optimisé

```python
CrossValidator(
    collectSubModels=False,  # garde seulement le bestModel (pas les 12)
    parallelism=2,           # 2 folds en parallèle
)
```

Gain : mémoire divisée par 12, temps réduit de ~30%.

### 4. AQE (Adaptive Query Execution)

```python
.config("spark.sql.adaptive.enabled", "true")
.config("spark.sql.adaptive.coalescePartitions.enabled", "true")
```

Spark réoptimise le plan d'exécution *pendant* le run en fonction des vraies tailles de partitions. Utile sur les données EEG où les sujets ont des tailles variables.

### 5. Pandas UDF (vectorisé) vs UDF classique

| UDF classique | Pandas UDF |
|--------------|------------|
| Sérialise chaque ligne (pickle, JVM→Python) | Envoie des batchs via Apache Arrow (mémoire partagée) |
| 576 000 appels Python | ~100 batchs |
| ~30 minutes | ~30 secondes |

### 6. `shuffle.partitions = 32` au lieu de 200

Avec 66 sujets, 200 partitions → 134 vides. On règle à 32 : assez pour le parallélisme, pas de tâches vides inutiles.

---

## 8. Streaming EEG en temps réel

### Vue d'ensemble

```
stream_producer_eeg.py        → data/stream_eeg/input/
(écrit 1 fichier Parquet       Spark readStream (file source)
 par epoch de 2s)                     ↓
                               foreachBatch
                                  ├── compute_features (FFT sklearn)
                                  ├── RF.predict → T0/T1/T2 + probas
                                  └── compute_alpha_topomap (64 canaux)
                                         ↓
                              data/stream_eeg/output/
                              ├── predictions/
                              └── topomap/
                                         ↓
                              dashboard_eeg.py (polling 1.5s)
                              → http://localhost:8052
```

### Pourquoi sklearn en streaming et pas le modèle MLlib ?

Latence : prédire 1 epoch via le modèle Spark MLlib coûte ~500 ms d'overhead JVM. Sklearn fait ça en ~10 ms. En BCI temps réel, chaque milliseconde compte.

Le modèle sklearn est entraîné au démarrage du streaming sur les fichiers Parquet existants (60 premiers fichiers ≈ 3700 epochs), puis mis en cache pickle.

### Broadcast du modèle

```python
bc_clf = spark.sparkContext.broadcast(clf)
```

Le modèle est broadcasté sur tous les workers une seule fois au démarrage, pas re-sérialisé à chaque tâche. Pattern indispensable pour distribuer l'inférence sklearn sur Spark.

### `applyInPandas` par epoch

```python
batch_df.groupBy("epoch_id").applyInPandas(process_epoch_distributed, schema=_UDF_SCHEMA)
```

Chaque epoch (groupe de ~320 lignes) est traité de manière indépendante par un worker. La classification et le calcul de topographie sont distribués.

### Source-agnosticité

Le `readStream.format("parquet")` est interchangeable avec Kafka en changeant uniquement le bloc source. Le reste du pipeline (foreachBatch, prédiction, écriture) reste identique.

---

## 9. Résultats et interprétation

### Métriques finales

| Métrique | Valeur | Référence |
|----------|--------|-----------|
| Accuracy | ~44% | Hasard uniforme : 33%, toujours T0 : 50% |
| F1 weighted | ~43% | Confirme que le modèle ne s'effondre pas sur T0 |
| Precision weighted | ~44% | |
| Recall weighted | ~44% | |

### Pourquoi pas 90% ?

Trois raisons documentées et inhérentes :

1. **Ambiguïté des labels T1/T2 selon le run** — le sens change entre R04 et R06 (cf. section 3)
2. **EEG de surface bruité** — muscles, yeux, secteur 50 Hz. Les meilleurs papiers plafonnent à 70-80% avec du deep learning (EEGNet, ATCNet) sur ce même dataset
3. **Split par sujet honnête** — la littérature rapporte souvent 85-90% avec un split par epoch (data leakage inclus). Notre 44% est l'équivalent honnête de leur 85%

### Feature importance attendue

Le RandomForest devrait classer `diff_alpha` en premier. Si ce n'est pas le cas, c'est un signal d'alarme : soit l'ordre temporel du signal est corrompu (bug dans `_ordered_signal`), soit la normalisation a gommé le signal utile.

---

## 10. Lancer le projet

### Prérequis

- Linux / WSL (les scripts bash ne tournent pas nativement sur Windows)
- Java 17 (`JAVA_HOME=/usr/lib/jvm/java-17-openjdk`)
- Python 3.11+
- Environnement virtuel `.venv`

### Installation

```bash
.venv/bin/pip install -r requirements.txt
```

### Pipeline complète

```bash
make download     # Télécharge le dataset EEG depuis PhysioNet (~3 Go)
make run          # Exécute tout (ingestion → features → training → evaluation → export)

# Ou étape par étape :
make ingest       # EDF → Parquet partitionné
make features     # FFT + normalisation
make train        # CrossValidator RandomForest + MLflow
make evaluate     # Métriques sur le test set
make export       # Export pour le dashboard
```

### Visualisations

```bash
make kedro-viz    # DAG du pipeline  → http://localhost:4141
make mlflow-ui    # Tracking ML      → http://localhost:5000
make dashboard    # Dashboard Dash   → http://localhost:8051
```

### Streaming BCI live

```bash
make demo         # Lance tout (producer + streaming + dashboard) → http://localhost:8052
# Ctrl+C pour arrêter

# Ou séparément :
make eeg-stream-run   # Lance Spark Streaming
make eeg-producer     # Envoie les epochs (S001/R04 par défaut)
make eeg-dashboard    # Dashboard live
```

### Tests et qualité

```bash
make test    # pytest — 17 tests unitaires
make lint    # ruff — vérification de style
```

### Commandes DVC

```bash
make dvc-repro    # Rejoue le pipeline, saute ce qui n'a pas changé
make dvc-status   # Vérife l'état du cache
make dvc-dag      # Graphe des stages
```

---

## 11. Glossaire complet

### Termes EEG / neurosciences

| Terme | Définition |
|-------|-----------|
| **EEG** | Électroencéphalogramme : enregistrement de l'activité électrique du cerveau via des électrodes posées sur le crâne |
| **Électrode** | Capteur posé sur le crâne qui mesure la différence de potentiel électrique local |
| **µV (microvolt)** | Unité de mesure du signal EEG. Très faible : 10-100 µV typiquement |
| **Hz (hertz)** | Mesure de fréquence. 160 Hz = 160 mesures par seconde |
| **Epoch** | Fenêtre temporelle découpée dans le signal continu (ici 2 secondes = 320 samples à 160 Hz) |
| **Cortex moteur** | Zone du cerveau qui planifie et exécute les mouvements volontaires |
| **Contralatéralité** | L'hémisphère cérébral gauche contrôle le côté droit du corps (et inversement) |
| **Système 10-20** | Standard international de placement des électrodes EEG sur le crâne |
| **C3 / Cz / C4** | Électrodes au sommet du crâne au-dessus du cortex moteur (gauche / centre / droit) |
| **Band power** | Énergie du signal dans une plage de fréquences donnée, calculée via FFT |
| **ERD** | Event-Related Desynchronization : baisse de la puissance alpha pendant l'imagerie motrice |
| **ERS** | Event-Related Synchronization : rebond de la puissance beta après le mouvement |
| **Latéralisation** | Asymétrie entre les deux hémisphères cérébraux (`diff_alpha = C3_alpha − C4_alpha`) |
| **Imagerie motrice** | Imaginer un mouvement sans le faire réellement. Active le cortex moteur comme si c'était réel |

### Termes BCI

| Terme | Définition |
|-------|-----------|
| **BCI** | Brain-Computer Interface : système qui décode l'activité cérébrale pour commander un appareil sans mouvement physique |
| **T0 / T1 / T2** | Labels PhysioNet : T0 = repos, T1 = tâche motrice 1, T2 = tâche motrice 2 |
| **Inter-subject variability** | Chaque cerveau est différent. Un modèle entraîné sur un sujet fonctionne mal sur un autre |
| **CSP** | Common Spatial Patterns : algorithme qui extrait des filtres spatiaux optimaux pour discriminer les classes EEG. Plus puissant que la FFT simple |
| **EEGNet / ATCNet** | Architectures deep learning spécialisées pour l'EEG. Atteignent 70-80% sur ce dataset |

### Termes Big Data / Spark

| Terme | Définition |
|-------|-----------|
| **Lazy evaluation** | Spark ne calcule rien tant qu'une *action* (`.count()`, `.write()`) n'est pas appelée. Les transformations (`.filter()`, `.groupBy()`) construisent un plan |
| **DAG** | Directed Acyclic Graph : le graphe d'exécution que Spark construit à partir de ton code |
| **Catalyst** | L'optimiseur de plans de requêtes de Spark SQL. Réordonne les opérations pour minimiser le coût |
| **Shuffle** | Redistribution des données entre partitions après un `groupBy` ou `join`. Coûteux car nécessite du réseau |
| **Partition** | Morceau de données traité par un seul worker en parallèle |
| **Partition pruning** | Spark ne lit que les partitions pertinentes grâce à la structure Hive-style des dossiers |
| **Broadcast join** | Envoie le petit DataFrame à tous les workers pour éviter un shuffle. Utilisé pour la normalisation |
| **Pandas UDF** | Fonction Python vectorisée via Apache Arrow. Bien plus rapide qu'un UDF classique ligne-par-ligne |
| **Arrow** | Format colonnaire en mémoire partagée entre la JVM Spark et Python. Évite la sérialisation pickle |
| **AQE** | Adaptive Query Execution : Spark réoptimise son plan *pendant* l'exécution en fonction des vraies tailles de données |
| **Spark Structured Streaming** | Extension de l'API DataFrame pour le traitement de flux en temps réel. Même API que le batch |
| **File source** | Source streaming qui surveille un répertoire. Chaque nouveau fichier = un micro-batch |
| **Micro-batch** | Petit paquet de données traité atomiquement dans le streaming |
| **foreachBatch** | Hook Spark Streaming : "pour chaque micro-batch, exécute cette fonction Python" |
| **Checkpoint** | Répertoire où Spark Streaming sauvegarde l'état pour reprendre après un crash |
| **Parquet** | Format colonnaire compressé, standard Big Data. Spark lit uniquement les colonnes nécessaires |

### Termes ML / Data Science

| Terme | Définition |
|-------|-----------|
| **Feature** | Variable d'entrée du modèle ML, ici une des 16 valeurs extraites de l'EEG par epoch |
| **Feature engineering** | Processus de transformation du signal brut en features pertinentes pour le modèle |
| **FFT** | Fast Fourier Transform : algorithme qui décompose un signal temporel en somme de sinusoïdes |
| **Z-score** | Normalisation : `(valeur − moyenne) / écart_type`. Rend les données comparables entre sujets |
| **CrossValidator** | Validation croisée : entraîne le modèle sur plusieurs sous-ensembles pour trouver les meilleurs hyperparamètres |
| **RandomForest** | Ensemble d'arbres de décision. Robuste au bruit, pas d'hypothèse linéaire, donne la feature importance |
| **StringIndexer** | Transforme les labels texte (T0/T1/T2) en entiers pour MLlib |
| **VectorAssembler** | Concatène les colonnes features en un seul vecteur dense pour MLlib |
| **Data leakage** | Information du test set qui fuite dans le train → métriques surestimées → modèle inutilisable en production |
| **F1 score** | Moyenne harmonique de la précision et du rappel. Plus fiable que l'accuracy sur des classes déséquilibrées |
| **Weighted metrics** | Métriques pondérées par la fréquence de chaque classe |
| **Class imbalance** | T0 = 50% des données, T1/T2 = 25% chacun. Sans compensation, le modèle prédit toujours T0 |

### Termes outils du projet

| Terme | Définition |
|-------|-----------|
| **Kedro** | Framework Python pour structurer des projets data science en pipelines reproductibles avec catalog et paramètres |
| **MLflow** | Plateforme de tracking d'expériences ML : log des paramètres, métriques, et artefacts à chaque run |
| **DVC** | Data Version Control : versionne les fichiers de données lourds (comme Git, mais pour les données) |
| **EDF** | European Data Format : format binaire standard pour les signaux biomédicaux (EEG, ECG…) |
| **MNE** | Librairie Python spécialisée dans le traitement des données EEG/MEG |

---

*NeuroSpark — Pipeline Big Data pour la classification EEG d'imagerie motrice*
*HELMo 2025-2026 — UE28 Big Data*
