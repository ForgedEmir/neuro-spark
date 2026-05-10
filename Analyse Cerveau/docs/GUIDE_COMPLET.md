# Guide complet — Projet EEG Motor Imagery

> Document autonome. Si tu lis ce fichier en entier, tu comprends :
> 1. **La science** — ce qu'est un EEG, d'où viennent les données, chaque colonne, chaque feature, avec exemples concrets.
> 2. **Le code** — chaque module, chaque fonction, ligne par ligne, avec le « pourquoi » de chaque choix Spark.
> 3. **Les optimisations** — pourquoi ces patterns plutôt que d'autres, quel coût on évite.
> 4. **Les résultats** — ce que signifient les métriques et pourquoi l'accuracy seule ment.

---

## Sommaire

- [Partie 1 — La science derrière le projet](#partie-1--la-science-derrière-le-projet)
- [Partie 2 — Le dataset PhysioNet en détail](#partie-2--le-dataset-physionet-en-détail)
- [Partie 3 — Les features : du signal brut aux 16 colonnes ML](#partie-3--les-features--du-signal-brut-aux-16-colonnes-ml)
- [Partie 4 — Architecture du code](#partie-4--architecture-du-code)
- [Partie 5 — Le pipeline ligne par ligne](#partie-5--le-pipeline-ligne-par-ligne)
- [Partie 6 — Les optimisations Spark expliquées](#partie-6--les-optimisations-spark-expliquées)
- [Partie 7 — Résultats et interprétation](#partie-7--résultats-et-interprétation)
- [Partie 8 — Comment lancer le projet](#partie-8--comment-lancer-le-projet)

---

## Partie 1 — La science derrière le projet

### 1.1 Le problème : « lire » une intention motrice

Quand tu **imagines** bouger ta main droite (sans la bouger réellement), une petite région de ton cortex moteur gauche change d'activité électrique. Cette modulation est détectable à la surface du crâne avec des électrodes. C'est le principe des **Brain-Computer Interfaces (BCI)** : piloter un curseur, un fauteuil, une prothèse, uniquement par la pensée.

**Objectif du projet** : entraîner un modèle de classification qui, à partir d'une fenêtre de 2 secondes d'EEG, prédise si le sujet :
- **T0** — est au repos
- **T1** — imagine un mouvement de la main gauche
- **T2** — imagine un mouvement de la main droite

### 1.2 Qu'est-ce qu'un EEG ?

Un **électroencéphalogramme** enregistre des différences de potentiel électrique à la surface du crâne. Les neurones, quand ils s'activent en synchronie, génèrent des champs électriques mesurables (quelques microvolts, µV).

Concrètement, un EEG c'est **N électrodes qui mesurent une tension en fonction du temps**. Si tu as 64 électrodes et que tu enregistres pendant 1 minute à 160 Hz (160 mesures/seconde), tu produis :

```
64 canaux × 60 secondes × 160 Hz = 614 400 valeurs
```

Chaque valeur est une tension en volts. C'est un **signal multicanal temporel**.

### 1.3 Contralatéralité : pourquoi C3, Cz, C4 ?

Le cerveau humain est **contralatéral** : l'hémisphère gauche contrôle le côté droit du corps, et inversement.

```
       Main GAUCHE imaginée              Main DROITE imaginée
              ↓                                    ↓
       C4 (hémisphère droit)           C3 (hémisphère gauche)
       activation locale                activation locale
```

Dans le système international 10-20 de placement des électrodes :
- **C3** — électrode au-dessus du cortex moteur **gauche** (contrôle main droite)
- **Cz** — électrode au **centre vertex** (référence neutre)
- **C4** — électrode au-dessus du cortex moteur **droit** (contrôle main gauche)

Le projet ne garde que ces 3 canaux parmi les 64 disponibles. Les 61 autres (occipital, frontal…) captent du bruit non pertinent pour distinguer main gauche vs droite.

### 1.4 ERD/ERS : le phénomène physique qu'on capture

Deux phénomènes clés pendant l'imagerie motrice :

- **ERD (Event-Related Desynchronization)** — la bande **alpha (8–13 Hz)** se **désynchronise** (baisse de puissance) au-dessus du cortex moteur contralatéral pendant l'imagerie.
- **ERS (Event-Related Synchronization)** — la bande **beta (13–30 Hz)** fait un **rebond** (hausse de puissance) après le mouvement.

C'est littéralement **ça** qu'on essaie de détecter : une baisse de puissance alpha sur C3 quand le sujet imagine sa main droite, une baisse sur C4 quand il imagine sa main gauche.

### 1.5 Les bandes de fréquence

| Bande | Plage (Hz) | Rôle cognitif |
|-------|-----------|---------------|
| theta | 4 – 8     | Préparation motrice, mémoire de travail |
| alpha | 8 – 13    | **Se désynchronise pendant l'imagerie (ERD)** — feature #1 |
| beta  | 13 – 30   | **Rebond post-mouvement (ERS)** — feature #2 |
| gamma | 30 – 80   | Traitement cognitif fin, attention focalisée |

On ne garde pas les fréquences < 4 Hz (delta, artefacts oculaires) ni > 80 Hz (bruit musculaire, secteur).

---

## Partie 2 — Le dataset PhysioNet en détail

### 2.1 Origine des données

Dataset **EEG Motor Movement/Imagery** de **PhysioNet / BCI2000** (Schalk et al. 2004). 109 sujets (on en utilise 66 après filtrage). Pour chaque sujet, 14 « runs » (séances) de ~2 min chacune :

| Runs | Contenu | Utilisé ? |
|------|---------|-----------|
| R01, R02 | Baseline yeux ouverts / fermés | Non (pas d'imagerie) |
| R03, R07, R11 | Ouvrir/fermer poing gauche ou droit (**réel**) | Oui |
| R04, R08, R12 | **Imaginer** ouvrir/fermer poing gauche ou droit | Oui |
| R05, R09, R13 | Ouvrir/fermer les deux poings ou deux pieds (réel) | Oui |
| R06, R10, R14 | **Imaginer** les deux poings ou deux pieds | Oui |

Le projet utilise **R03 à R14** (voir `motor_run_range` dans `config.yaml`).

### 2.2 Anatomie d'un fichier EDF

Chaque run est stocké en format **EDF (European Data Format)** — un format binaire standard en médecine pour les signaux biomédicaux.

**Structure d'un fichier EDF :**
```
┌──────────────────────────────────────────────┐
│  Header                                      │
│  - Infos patient, date                       │
│  - Fréquence d'échantillonnage : 160 Hz      │
│  - Nombre de canaux : 64                     │
│  - Noms des canaux : "Fc5.", "C3..", etc.    │
├──────────────────────────────────────────────┤
│  Annotations (labels temporels)              │
│  - T0 de 0.0s à 4.2s                         │
│  - T1 de 4.2s à 8.3s                         │
│  - T0 de 8.3s à 12.5s                        │
│  - T2 de 12.5s à 16.7s ...                   │
├──────────────────────────────────────────────┤
│  Data                                        │
│  - 64 canaux × ~20 000 samples               │
│  - chaque sample en microvolts               │
└──────────────────────────────────────────────┘
```

Un seul fichier EDF fait ~700 Ko. L'ensemble du dataset fait ~550 Mo.

### 2.3 Structure d'une ligne après conversion EDF → DataFrame

Après [ingestion.py](scripts/eeg/ingestion.py#L23), un fichier EDF devient un DataFrame Pandas, puis Parquet, de la forme :

| subject_id | run_id | time  | task_label | Fc5. | Fc3. | ... | C3.. | Cz.. | C4.. | ... | O2.. |
|-----------|--------|-------|-----------|------|------|-----|------|------|------|-----|------|
| S001      | R03    | 0.000 | T0        | -0.0000234 | 0.0000187 | ... | 0.0000412 | -0.0000091 | 0.0000065 | ... | 0.0000023 |
| S001      | R03    | 0.006 | T0        | -0.0000219 | 0.0000201 | ... | 0.0000398 | -0.0000103 | 0.0000071 | ... | 0.0000019 |
| S001      | R03    | 0.012 | T0        | -0.0000203 | 0.0000215 | ... | 0.0000381 | -0.0000112 | 0.0000077 | ... | 0.0000015 |
| ...       | ...    | ...   | ...       |            |          |     |          |          |          |     |          |
| S001      | R03    | 4.200 | T1        | ...        | ...      |     | ...      | ...      | ...      |     | ...      |

**Explication de chaque colonne :**

| Colonne | Type | Sens | Exemple |
|---------|------|------|---------|
| `subject_id` | string | Identifiant du sujet (1 à 109) | `"S001"`, `"S042"` |
| `run_id` | string | Numéro du run (R03–R14) | `"R03"` |
| `time` | double | Timestamp en secondes depuis le début du run | `0.006` = 6 ms après le début |
| `task_label` | string | Étiquette de tâche à cet instant | `"T0"` (repos), `"T1"` (main gauche imaginée), `"T2"` (main droite imaginée) |
| `Fc5.`, `Fc3.`, … | double | Tension en **volts** sur chaque électrode (avec `.` car EDF met des points) | `-0.0000234` (soit −23.4 µV) |

**À l'échelle du projet complet :**

```
66 sujets × 12 runs × ~20 000 samples = ~15 000 000 lignes
```

Chaque ligne a **~68 colonnes** (4 méta + 64 canaux). C'est là que **Spark devient nécessaire** — 15M lignes en RAM pandas sur un laptop crashe.

### 2.4 Les annotations : comment T0/T1/T2 sont assignés

Un fichier EDF stocke les labels comme des **intervalles temporels**, pas ligne par ligne. Par exemple :

```
onset=0.0   duration=4.2   description="T0"
onset=4.2   duration=4.1   description="T1"
onset=8.3   duration=4.2   description="T0"
onset=12.5  duration=4.2   description="T2"
...
```

Dans [ingestion.py:41-46](scripts/eeg/ingestion.py#L41-L46), on itère sur ces annotations et on applique chaque label aux samples dans la fenêtre `[onset, onset+duration)` :

```python
df['task_label'] = 'T0'  # valeur par défaut
for ann in raw.annotations:
    onset = ann['onset']
    end = onset + ann['duration']
    mask = (df['time'] >= onset) & (df['time'] < end)
    df.loc[mask, 'task_label'] = ann['description']
```

**Signification concrète des labels :**

- **T0 (repos)** — le sujet ne fait rien de particulier. C'est la baseline. Représente ~50% du dataset.
- **T1 (main gauche imaginée)** — dans R04, R08, R12. Dans R05, R09, R13, c'est « deux poings ». Dans R06, R10, R14, c'est « imaginer deux poings ».
- **T2 (main droite imaginée)** — idem, dépend du run.

**Limite scientifique (importante)** : le sens de T1/T2 **dépend du run**. Dans R04 c'est « imaginer main gauche », dans R06 c'est « imaginer deux poings ». Le projet ne tient pas compte de ça — il les traite comme deux classes unifiées. C'est une limite documentée dans le rapport réflexif et qui explique en partie le 45% d'accuracy (les labels T1/T2 sont ambigus selon le run).

---

## Partie 3 — Les features : du signal brut aux 16 colonnes ML

### 3.1 Pourquoi on ne donne pas le signal brut au modèle

Un epoch de 2 secondes = **320 samples × 64 canaux = 20 480 valeurs**. Donner ça brut à un RandomForest, c'est :

1. Noyer le modèle sous la dimensionnalité (malédiction de la dimension).
2. Perdre la **structure fréquentielle** : l'info utile (ERD alpha) est dans le spectre, pas dans la forme d'onde.
3. Exploser en temps de calcul sans gain.

**Solution : extraction de features spectrales.** On résume chaque epoch par la puissance dans chaque bande, sur chaque canal. On passe de 20 480 valeurs à **16 valeurs** par epoch.

### 3.2 Les 16 features en détail

Pour chaque epoch de 2 secondes, on calcule :

**Partie 1 — Puissance par canal × bande (12 features) :**

| Feature | Formule | Sens |
|---------|---------|------|
| `C3_theta` | Puissance FFT dans [4, 8] Hz sur C3 | Préparation motrice hémisphère gauche |
| `C3_alpha` | Puissance FFT dans [8, 13] Hz sur C3 | **Désynchronisation alpha = imagerie main droite** |
| `C3_beta`  | Puissance FFT dans [13, 30] Hz sur C3 | Rebond beta hémisphère gauche |
| `C3_gamma` | Puissance FFT dans [30, 80] Hz sur C3 | Activité cognitive fine hémisphère gauche |
| `Cz_theta` | ... sur Cz (vertex) | Référence neutre |
| `Cz_alpha` | ... |
| `Cz_beta`  | ... |
| `Cz_gamma` | ... |
| `C4_theta` | ... sur C4 (hémisphère droit) | |
| `C4_alpha` | ... | **Désynchronisation alpha = imagerie main gauche** |
| `C4_beta`  | ... | |
| `C4_gamma` | ... | |

**Partie 2 — Features de latéralisation (4 features) :**

| Feature | Formule | Sens |
|---------|---------|------|
| `diff_theta` | C3_theta − C4_theta | Asymétrie inter-hémisphérique bande theta |
| `diff_alpha` | C3_alpha − C4_alpha | **Signature directe de la latéralité imaginée** |
| `diff_beta`  | C3_beta − C4_beta  | |
| `diff_gamma` | C3_gamma − C4_gamma | |

**Pourquoi `diff_alpha` est la feature reine** :
- Si le sujet imagine sa **main droite** → C3 se désynchronise → C3_alpha **baisse** → `diff_alpha` **baisse**
- Si le sujet imagine sa **main gauche** → C4 se désynchronise → C4_alpha **baisse** → `diff_alpha` **monte**

La différence encode directement la **direction**, là où C3 seul ou C4 seul ne le font pas.

### 3.3 Comment la FFT calcule la « puissance dans une bande »

**Signal temporel** → **spectre de puissance** via la **Transformée de Fourier rapide (FFT)**.

Pour un epoch de 320 samples à 160 Hz :

```python
freqs = np.fft.rfftfreq(320, 1/160)   # → [0, 0.5, 1.0, 1.5, ..., 80] Hz
power = np.abs(np.fft.rfft(signal)) ** 2   # puissance à chaque fréquence

# Puissance dans la bande alpha [8, 13] :
mask = (freqs >= 8) & (freqs < 13)
alpha_power = np.mean(power[mask])
```

La **FFT** décompose le signal en somme de sinusoïdes. Le carré du module donne la **puissance** à chaque fréquence. On moyenne dans la bande d'intérêt pour obtenir un scalaire.

Exemple concret (ordres de grandeur typiques, unités arbitraires après normalisation) :

| Epoch | C3_alpha | C4_alpha | diff_alpha | task_label |
|-------|----------|----------|-----------|-----------|
| #123 | 0.8  | 1.2  | −0.4 | T2 (main droite imaginée) |
| #124 | 1.3  | 0.7  |  +0.6 | T1 (main gauche imaginée) |
| #125 | 1.0  | 1.0  |  0.0 | T0 (repos) |

Le modèle apprend : `diff_alpha < 0 → T2`, `diff_alpha > 0 → T1`, `diff_alpha ≈ 0 → T0`.

### 3.4 La normalisation z-score par sujet

Problème : chaque personne a une **amplitude EEG différente**. Crâne plus épais, cheveux, gel conducteur — tout ça change les valeurs absolues. Sans normaliser, le modèle apprend à reconnaître **qui** est le sujet, pas **ce qu'il fait**.

**Z-score par sujet** :

```
feature_normalisée = (feature − moyenne_du_sujet) / écart_type_du_sujet
```

Après normalisation, chaque sujet a les mêmes stats (moyenne 0, écart-type 1). Le modèle ne peut plus tricher sur l'identité.

---

## Partie 4 — Architecture du code

### 4.1 Vue d'ensemble

```
Analyse Cerveau/
├── scripts/
│   ├── config.yaml                 ← toutes les constantes ajustables
│   ├── download_eeg.py             ← télécharge le dataset PhysioNet
│   └── eeg/                        ← package Python du pipeline
│       ├── __init__.py
│       ├── config.py               ← charge config.yaml
│       ├── logger.py               ← factory de loggers
│       ├── spark_utils.py          ← SparkSession factory
│       ├── ingestion.py            ← EDF → Parquet
│       ├── features.py             ← FFT + normalisation
│       ├── training.py             ← RandomForest + CrossValidator
│       └── export.py               ← Parquet → dashboard
├── warehouse/
│   └── etape1_poc/
│       └── poc_eeg.ipynb           ← notebook orchestrateur (32 cellules)
├── data/
│   ├── eeg/                        ← fichiers EDF bruts (550 Mo)
│   ├── parquet/                    ← Parquet convertis
│   └── dashboard/                  ← exports pour le dashboard
├── dashboard.py                    ← app Dash/Plotly
└── GUIDE_COMPLET.md                ← ce fichier
```

### 4.2 Séparation des responsabilités

| Module | Responsabilité unique |
|--------|----------------------|
| `config.py` | Charger `config.yaml` et exposer les constantes Python |
| `logger.py` | Fournir des loggers configurés de manière cohérente |
| `spark_utils.py` | Créer une `SparkSession` avec les bons réglages |
| `ingestion.py` | Parser les EDF (**Pandas**, pas Spark — un seul fichier à la fois) |
| `features.py` | Transformations Spark distribuées (FFT UDF, normalisation) |
| `training.py` | Pipeline MLlib + CrossValidator + évaluation |
| `export.py` | Écriture finale Parquet pour le dashboard |

**Principe directeur** : chaque module fait **une chose** et est testable indépendamment. Le notebook ne contient **que l'orchestration** (imports + appels). Aucune logique métier dans le notebook.

### 4.3 Pourquoi `config.yaml` centralisé

**Avant** : constantes éparpillées dans 5 fichiers Python, `FS=160` dans l'un, `FS=160` dans l'autre, tôt ou tard une incohérence.

**Après** : un seul `config.yaml`, lu par [config.py:19-20](scripts/eeg/config.py#L19-L20), puis exposé en tant que constantes Python :

```yaml
# config.yaml
acquisition:
  sampling_rate_hz: 160
  epoch_seconds: 2
```

```python
# config.py
FS = _cfg['acquisition']['sampling_rate_hz']           # 160
EPOCH_SEC = _cfg['acquisition']['epoch_seconds']       # 2
EPOCH_SAMPLES = FS * EPOCH_SEC                         # 320
```

**Gain** : pour tester une autre bande alpha, un autre ratio train/test, un autre nombre d'arbres — **zéro ligne de code Python à modifier**. Juste le YAML.

### 4.4 Pourquoi un logger dédié (pas `print`)

Avant : `print("Sujet S001 converti")` partout. Impossible de désactiver, pas de niveau, pas d'horodatage.

Après : `log.info("Sujet S001 converti")`. Le niveau est piloté par `config.yaml` (`logging.level: INFO`). Passer en `DEBUG` pour debug, `WARNING` pour runs silencieux. Format uniforme :

```
14:55:54 [INFO] eeg.ingestion: Sujets trouvés : 66
14:55:54 [INFO] eeg.ingestion: [10/66] S010 — 12 convertis, 108 déjà présents
```

Le `force=True` dans [logger.py:26](scripts/eeg/logger.py#L26) écrase la config par défaut que Jupyter et PySpark installent — sinon les logs seraient invisibles dans le notebook.

---

## Partie 5 — Le pipeline ligne par ligne

### 5.1 `download_eeg.py` — téléchargement parallèle

[scripts/download_eeg.py](scripts/download_eeg.py)

```python
with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(download_one, rec): rec for rec in todo}
```

**ThreadPoolExecutor avec 10 workers** : le téléchargement est **I/O-bound** (attente réseau), pas CPU-bound. Les threads Python contournent le GIL pour I/O, 10× plus rapide qu'une boucle séquentielle. Résultat : ~550 Mo téléchargés en 2-3 minutes au lieu de 20+.

**Idempotence** (lignes 13-16) : on compare avec l'existant, on ne retélécharge jamais ce qui est déjà présent. Relancer le script après un crash = reprise exacte.

### 5.2 `ingestion.py` — EDF → Parquet

[scripts/eeg/ingestion.py:23](scripts/eeg/ingestion.py#L23)

```python
def edf_to_dataframe(edf_path, subject_id, run_id):
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    data, times = raw.get_data(return_times=True)
    df = pd.DataFrame(data.T, columns=raw.ch_names)
```

- **`mne.io.read_raw_edf`** — parse le binaire EDF, retourne un objet `Raw`.
- **`.get_data()`** retourne `(n_canaux, n_samples)`. On **transpose** (`.T`) pour avoir `(n_samples, n_canaux)` — une ligne = un instant, une colonne = un canal. C'est la structure naturelle d'un DataFrame.
- **`raw.ch_names`** donne les noms `['Fc5.', 'Fc3.', 'Fc1.', ..., 'Oz..']` qu'on utilise directement comme noms de colonnes.

**Pourquoi Pandas et pas Spark ici ?** Un fichier EDF fait 20 000 lignes max (~700 Ko). Lancer une SparkSession pour ça coûte 30 secondes d'overhead par fichier. Pandas les parse en 0.2 seconde. **Règle** : Spark pour les gros volumes, Pandas pour les petits.

[scripts/eeg/ingestion.py:51](scripts/eeg/ingestion.py#L51)

```python
def batch_convert_edf(edf_dir, parquet_dir):
    ...
    if os.path.exists(parquet_file):
        stats['skipped'] += 1
        continue
```

**Idempotence au niveau fichier** : on skippe ce qui existe déjà. Si le notebook crashe à S045, on reprend à S045 sans reconvertir les 44 premiers.

**Pourquoi Parquet** :
- **Colonnaire** : Spark lit uniquement les colonnes demandées (`.select('C3', 'Cz', 'C4')` ne charge pas les 61 autres).
- **Compressé** : ~3-5× plus petit que CSV.
- **Typé** : pas de re-parsing string → float à chaque lecture.

### 5.3 `spark_utils.py` — création de la SparkSession

[scripts/eeg/spark_utils.py:35-42](scripts/eeg/spark_utils.py#L35-L42)

```python
spark = (SparkSession.builder
    .appName(app_name)
    .master(master)
    .config('spark.executor.memory', executor_memory)
    .config('spark.driver.memory', driver_memory)
    .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
    .config('spark.sql.shuffle.partitions', str(shuffle_partitions))
    .getOrCreate())
```

Chaque config expliquée :

| Config | Valeur | Pourquoi |
|--------|--------|----------|
| `executor.memory` | `8g` | Le worker fait les **vrais calculs** (FFT sur 48k epochs, groupBy sur 15M lignes). Il lui faut de la RAM. |
| `driver.memory` | `4g` | Le driver (notebook) orchestre, il collecte peu. 4 Go suffit. |
| `arrow.pyspark.enabled` | `true` | **Critique**. Transfert Spark→Pandas via mémoire partagée Arrow (colonnaire, zéro-copie) au lieu de pickle sérialisé. Peut être **10× plus rapide**. |
| `shuffle.partitions` | `32` | Par défaut Spark crée 200 partitions après un shuffle. Avec 66 sujets, 134 partitions seraient **vides** (overhead de scheduling pour rien). 32 suffit. |

**`.getOrCreate()`** réutilise la session si déjà créée. Pratique en notebook où on ré-exécute la cellule.

### 5.4 `features.py` — le cœur du pipeline Spark

#### 5.4.1 `clean_channel_names` — renommer `C3..` en `C3`

[scripts/eeg/features.py:22](scripts/eeg/features.py#L22)

```python
rename_map = {c: c.replace('.', '') for c in df.columns if '.' in c}
return df.select([
    F.col(f'`{c}`').alias(rename_map.get(c, c))
    for c in df.columns
])
```

**Problème** : EDF nomme les canaux `C3..`, `Cz..`. Dans Spark, le **point est un séparateur SQL** (`table.colonne`). `F.col('C3..')` lève `AnalysisException: cannot resolve 'C3'`.

**Solution** :
- **Backticks** (`` `C3..` ``) pour échapper le point dans la sélection.
- **Un seul `select()`** qui renomme tout en une passe, au lieu d'une boucle `withColumnRenamed` qui **recrée un plan Catalyst à chaque itération**. Sur 64 colonnes, la boucle prendrait 64 re-planifications ; le `select` unique en fait 1.

#### 5.4.2 `add_epochs` — découper en fenêtres de 2 secondes

[scripts/eeg/features.py:36](scripts/eeg/features.py#L36)

```python
return df.withColumn('epoch_id', F.floor(F.col('time') / epoch_sec).cast('int'))
```

Si `time = 0.0 à 1.999` → `epoch_id = 0`.
Si `time = 2.0 à 3.999` → `epoch_id = 1`.
Et ainsi de suite.

**Pourquoi 2 secondes ?** 
- **Assez long** : 320 samples à 160 Hz donnent une **résolution fréquentielle de 0.5 Hz** (FFT). Assez fin pour distinguer alpha (8-13) de beta (13-30).
- **Assez court** : un essai d'imagerie motrice dure ~4 s. On veut 2 epochs par essai pour doubler le nombre d'exemples d'entraînement.

#### 5.4.3 `_make_band_udf` — le Pandas UDF FFT

[scripts/eeg/features.py:45](scripts/eeg/features.py#L45)

```python
@pandas_udf(DoubleType())
def band_power_udf(signals: pd.Series) -> pd.Series:
    def compute(vals):
        arr = np.array(vals, dtype=float)
        freqs = np.fft.rfftfreq(len(arr), 1.0 / fs)
        power = np.abs(np.fft.rfft(arr)) ** 2
        mask = (freqs >= low) & (freqs < high)
        return float(np.mean(power[mask])) if mask.any() else 0.0
    return signals.apply(compute)
return band_power_udf
```

**UDF classique vs Pandas UDF** — la différence est **fondamentale** :

| UDF classique | Pandas UDF (vectorisé) |
|--------------|------------------------|
| Spark sérialise **chaque ligne** JVM→Python (pickle) | Spark envoie un **batch** de lignes via Arrow (mémoire partagée, zéro-copie) |
| 48k epochs × 12 bandes = **576 000 appels Python** | ~100 batchs Arrow |
| ~30 minutes | ~30 secondes |

**Factory pattern** (`_make_band_udf(fs, low, high)`) : on génère un UDF **par bande** en **fermant** sur `low` et `high` via une closure. Pas besoin de passer 3 colonnes à chaque appel.

#### 5.4.4 `_ordered_signal` — le bug subtil du `collect_list`

[scripts/eeg/features.py:68](scripts/eeg/features.py#L68)

```python
struct_col = F.struct(F.col('time'), F.col(canal).alias('v'))
sorted_col = F.array_sort(F.collect_list(struct_col))
return F.transform(sorted_col, lambda s: s['v'])
```

**LE piège qui casse tout** : `collect_list()` **ne garantit pas l'ordre** après un shuffle. Sur un groupBy, les 320 samples d'un epoch peuvent arriver dans n'importe quel ordre. Pour une FFT, l'ordre temporel est **indispensable** — la FFT d'un signal mélangé donne un spectre faux.

**Solution en 3 étapes :**
1. **`F.struct(time, valeur)`** — on attache le timestamp à chaque valeur pour préserver l'info temporelle.
2. **`F.array_sort(F.collect_list(...))`** — on collecte les structs, puis on les trie par `time` (premier champ du struct).
3. **`F.transform(sorted, lambda s: s['v'])`** — on extrait uniquement les valeurs du tableau trié.

Sans ce pattern, les features sont du bruit. **C'est le bug le plus insidieux du projet** — le code tournerait sans erreur, mais les résultats seraient aléatoires.

#### 5.4.5 `build_band_features` — l'agrégation principale

[scripts/eeg/features.py:81](scripts/eeg/features.py#L81)

```python
agg_exprs = [F.first('task_label').alias('task_label')]
for canal in channels:
    signal_expr = _ordered_signal(canal)
    for band_name, (low, high) in bands.items():
        udf_fn = _make_band_udf(fs, low, high)
        agg_exprs.append(udf_fn(signal_expr).alias(f'{canal}_{band_name}'))

return df_epochs.groupBy('subject_id', 'run_id', 'epoch_id').agg(*agg_exprs)
```

Pour chaque (`subject_id`, `run_id`, `epoch_id`) :
- On prend le `task_label` majoritaire (`F.first`) — un epoch de 2s partage le même label à 99%.
- Pour chaque canal ∈ {C3, Cz, C4} et chaque bande ∈ {theta, alpha, beta, gamma} :
  - On collecte le signal trié
  - On applique l'UDF FFT
  - On nomme la colonne `C3_theta`, `C3_alpha`, …, `C4_gamma`

**Résultat** : un DataFrame à **13 colonnes** (4 méta + `task_label` + 12 features) et **~48 000 lignes** (une par epoch).

#### 5.4.6 `add_lateralization_features` — les diffs C3-C4

[scripts/eeg/features.py:102](scripts/eeg/features.py#L102)

```python
diff_exprs = {
    f'diff_{band}': F.col(f'C3_{band}') - F.col(f'C4_{band}')
    for band in bands.keys()
}
return df_features.withColumns(diff_exprs)
```

**`.withColumns(dict)`** (Spark 3.3+) vs une boucle `.withColumn(…)` :

| withColumn (boucle) | withColumns (dict) |
|---------------------|---------------------|
| 4 appels → 4 analyses Catalyst → 4 optimisations | 1 appel → 1 analyse → toutes les colonnes ajoutées ensemble |
| O(n) passes de planification | O(1) passe |

Sur 4 colonnes c'est négligeable, mais sur 50 colonnes ça fait une différence mesurable.

#### 5.4.7 `normalize_by_subject` — Z-score via Window

[scripts/eeg/features.py:120](scripts/eeg/features.py#L120)

```python
w = Window.partitionBy('subject_id')
meta_cols = ['subject_id', 'run_id', 'epoch_id', 'task_label']

norm_exprs = [F.col(c) for c in meta_cols]
for fc in feature_cols:
    std = F.coalesce(F.stddev(fc).over(w), F.lit(1.0))
    norm_exprs.append(
        ((F.col(fc) - F.mean(fc).over(w)) / (std + 1e-10)).alias(fc)
    )

df_norm = df_features.select(norm_exprs)
```

**Concept clé : `Window.partitionBy('subject_id')`** — c'est une **agrégation « virtuelle »** : Spark calcule la moyenne/écart-type **par sujet**, mais garde toutes les lignes (contrairement à `groupBy` qui les agrège).

```
       avant Window                 après normalisation
sub  C3_alpha                    sub  C3_alpha
S001   2.5          →         S001   −0.8  (dans les valeurs basses de S001)
S001   4.1                     S001   +0.9
S002   12.3                    S002   −0.3  (dans les valeurs basses de S002, bien que 12.3 absolument)
S002   18.7                    S002   +1.1
```

**`F.coalesce(F.stddev(fc).over(w), F.lit(1.0))`** — sécurité : si un sujet a un écart-type de 0 (signal constant, cas pathologique), on évite la division par zéro en remplaçant par 1.

**Le `+ 1e-10`** — garde-fou anti-division-par-zéro numérique même avec les float64.

**Pourquoi un seul `select` plutôt qu'une boucle `withColumn`** : même raison qu'au 5.4.6 — 16 features × re-planification Catalyst coûterait cher.

Enfin, on ajoute la colonne de poids :

```python
return df_norm.withColumn(
    'weight',
    F.when(F.col('task_label') == 'T0', t0_weight).otherwise(1.0)
)
```

T0 pèse 0.5, T1/T2 pèsent 1.0. Sans ça, le modèle prédit T0 tout le temps (50% d'accuracy sans rien apprendre).

### 5.5 `training.py` — MLlib

#### 5.5.1 `split_by_subject` — split par personne, pas par ligne

[scripts/eeg/training.py:14](scripts/eeg/training.py#L14)

```python
all_subjects = sorted([
    r.subject_id for r in df.select('subject_id').distinct().collect()
])
n_train = int(len(all_subjects) * train_ratio)
train_subjects = all_subjects[:n_train]
test_subjects = all_subjects[n_train:]

train_df = df.filter(F.col('subject_id').isin(train_subjects))
test_df = df.filter(F.col('subject_id').isin(test_subjects))
```

**LE choix méthodologique le plus important du projet.**

Split **aléatoire sur les epochs** (naïf) :
- S001 a 500 epochs, répartis 400 train / 100 test.
- Le modèle apprend le « style EEG » de S001 (forme du crâne, placement d'électrodes).
- Au test il retrouve ce style → il a l'air de bien classifier.
- En production sur un nouveau sujet → il s'effondre. C'est de la **mémorisation**, pas de la généralisation.

Split **par sujet** (ici) :
- 80% des **sujets** (≈53 personnes) vont en train.
- 20% des **sujets** (≈13 personnes) vont en test.
- Au test, le modèle voit des **personnes jamais vues**.
- C'est le **vrai critère** : peut-il généraliser ?

**Pourquoi `.distinct().collect()` plutôt que de hardcoder `['S001' ... 'S109']`** : si quelques fichiers sont corrompus (ça arrive), la range hardcodée inclurait des sujets fantômes. `.distinct()` donne la liste **réelle** présente dans le DataFrame.

#### 5.5.2 `build_pipeline` — le Pipeline MLlib

[scripts/eeg/training.py:38](scripts/eeg/training.py#L38)

```python
indexer = StringIndexer(inputCol='task_label', outputCol='label')
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
rf = RandomForestClassifier(
    featuresCol='features',
    labelCol='label',
    weightCol='weight',
    seed=seed,
)
return Pipeline(stages=[indexer, assembler, rf])
```

Trois stages :

1. **`StringIndexer`** : `"T0"→0.0`, `"T1"→1.0`, `"T2"→2.0`. MLlib veut des labels numériques.
2. **`VectorAssembler`** : concatène les 16 colonnes features en **une seule colonne** `features` de type `Vector`. MLlib veut un vecteur unique.
3. **`RandomForestClassifier`** : le modèle lui-même.
   - `weightCol='weight'` — MLlib utilise la colonne `weight` pour pondérer les exemples pendant l'entraînement.

**Pourquoi un Pipeline plutôt que des transformations séparées** :

Sans Pipeline, on ferait :
```python
indexed = indexer.fit(df_all).transform(df_all)   # ← indexer voit TOUT
assembled = assembler.transform(indexed)
train, test = assembled.randomSplit([0.8, 0.2])
model = rf.fit(train)
```

Problème : `indexer.fit(df_all)` apprend les mappings sur **tout** le dataset, test inclus. C'est du **data leakage** — le test influence l'entraînement.

Avec Pipeline, `CrossValidator` fait `pipeline.fit(train_fold)` — l'indexer apprend **seulement** sur le fold d'entraînement, à chaque fold. Zéro leakage.

**Pourquoi RandomForest plutôt que SVM ou régression logistique** :
- **Robuste au bruit** (l'EEG est ultra-bruité).
- **Pas d'hypothèse linéaire** (les relations sont non linéaires).
- **Gratuit : la feature importance** (quels `diff_alpha`/`C3_beta` pèsent le plus).

#### 5.5.3 `train_with_cv` — CrossValidator + grille d'hyperparamètres

[scripts/eeg/training.py:63](scripts/eeg/training.py#L63)

```python
param_grid = (ParamGridBuilder()
    .addGrid(rf.numTrees, [50, 100])
    .addGrid(rf.maxDepth, [5, 10])
    .build())

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,
    seed=seed,
)
return cv.fit(train_df)
```

**Ce qui se passe concrètement** :

```
ParamGrid : 2 numTrees × 2 maxDepth = 4 combinaisons
Folds : 3

Pour chaque combinaison (numTrees, maxDepth) :
    Pour chaque fold (1, 2, 3) :
        Entraîner sur 2/3 du train
        Évaluer sur 1/3 du train
    Moyenne des 3 accuracies

→ On garde la combinaison qui a la meilleure moyenne
→ On réentraîne sur TOUT le train avec cette combinaison
→ C'est cv.bestModel
```

**Total : 4 × 3 + 1 = 13 entraînements**. Coûteux mais on obtient une estimation fiable + les meilleurs hyperparamètres.

#### 5.5.4 `evaluate_model` — 4 métriques

[scripts/eeg/training.py:104](scripts/eeg/training.py#L104)

```python
for name, metric in [
    ('accuracy', 'accuracy'),
    ('f1', 'f1'),
    ('precision_weighted', 'weightedPrecision'),
    ('recall_weighted', 'weightedRecall'),
]:
    evaluator = MulticlassClassificationEvaluator(
        labelCol='label', predictionCol='prediction', metricName=metric
    )
    metrics[name] = evaluator.evaluate(predictions)
```

**Pourquoi pas juste accuracy ?**

Sur ce dataset, T0 représente ~50% des epochs. Un modèle qui prédit **toujours T0** aurait 50% d'accuracy **sans rien apprendre**. Accuracy = trompeur.

- **Precision** (par classe) : parmi ceux prédits T1, combien étaient vraiment T1 ?
- **Recall** (par classe) : parmi les vrais T1, combien ont été retrouvés ?
- **F1** : moyenne harmonique precision/recall — pénalise l'ignorance des classes minoritaires.
- **Weighted** : pondère chaque classe par sa fréquence. F1 weighted bas = le modèle rate les classes rares.

### 5.6 `export.py` — Parquet pour le dashboard

[scripts/eeg/export.py:16](scripts/eeg/export.py#L16)

On exporte **4 fichiers** :

```python
(df_raw
    .filter((F.col('subject_id') == sample_subject) & (F.col('run_id') == sample_run))
    .select('time', 'Cz', 'task_label')
    .orderBy('time')
    .limit(3200)
    .toPandas()
    .to_parquet(f'{output_dir}/signal_sample.parquet', index=False))
```

1. **`signal_sample.parquet`** — 3200 samples (20 s) d'un sujet/run pour afficher le signal brut sur le dashboard.
2. **`features.parquet`** — les 48k epochs × 16 features (visualisations, analyses).
3. **`predictions.parquet`** — prédictions du test set avec subject_id.
4. **`feature_importance.parquet`** — l'importance de chaque feature selon le RF gagnant.

**Point critique** : on n'appelle `.toPandas()` **qu'à la fin** et **seulement sur des extraits**. Tout le pipeline reste dans Spark (distribué). Si on avait fait `.toPandas()` sur les 15M lignes brutes, le **driver aurait crashé** (OOM).

**`cv_model.bestModel.stages[-1]`** — on extrait le RandomForest du dernier stage du meilleur Pipeline, pour accéder à `.featureImportances`.

---

## Partie 6 — Les optimisations Spark expliquées

### 6.1 Arrow : `spark.sql.execution.arrow.pyspark.enabled`

**Sans Arrow** : `toPandas()` sérialise chaque ligne en pickle, envoie sur le socket Python, désérialise. Pour 48k lignes × 16 colonnes, c'est ~30 secondes.

**Avec Arrow** : mémoire partagée colonnaire, zéro-copie. ~3 secondes.

Gain x10. Activé **partout** dans le projet.

### 6.2 Pandas UDF vs UDF classique

Déjà expliqué en 5.4.3. Gain typique : **x50 à x100** sur les opérations vectorisables (FFT, calculs numpy).

### 6.3 `shuffle_partitions=32` au lieu de 200

Après chaque shuffle (groupBy, join, window), Spark répartit les données sur **N partitions**. Par défaut N=200.

Avec 66 sujets, un `groupBy('subject_id')` ne produit au mieux que 66 groupes — **134 partitions seraient vides**. Chaque partition vide = un task scheduler pour rien.

On règle à 32 — assez pour du parallélisme, pas de vides.

### 6.4 `withColumns` (dict) vs boucle `withColumn`

```python
# Mauvais : 16 re-analyses Catalyst
for fc in feature_cols:
    df = df.withColumn(fc, expr)

# Bon : 1 re-analyse
df = df.withColumns({fc: expr for fc in feature_cols})
```

Ou encore mieux : **un seul `select(...)`** avec toutes les expressions, ce qu'on fait dans `normalize_by_subject` (scripts/eeg/features.py:149).

### 6.5 `Window.partitionBy` vs `groupBy`

| groupBy | Window |
|---------|--------|
| Réduit les lignes (agrège) | Garde toutes les lignes, ajoute des colonnes calculées par groupe |
| Pour des features globales | Pour des normalisations, rangs, cumuls |

Pour le z-score **par sujet**, on veut garder chaque epoch mais avec les stats de son sujet. Window est le bon outil.

### 6.6 Parquet colonnaire

`.select('time', 'Cz', 'task_label')` sur du **CSV** charge toutes les colonnes puis filtre.
Sur du **Parquet** ça ne lit **physiquement que ces 3 colonnes** du disque. Gain proportionnel au nombre de colonnes non utilisées (ici 61/64 ignorées = ~95% d'I/O économisée).

### 6.7 Split par sujet (pas par epoch) — anti data leakage #1

Expliqué en 5.5.1. **Sans ce choix, toutes les métriques sont fausses** (surestimées). Aucun gain de performance machine — c'est une décision méthodologique. Mais c'est elle qui rend le 45% final **honnête**.

### 6.8 Pipeline MLlib — anti data leakage #2

Sans Pipeline, le StringIndexer / l'assembler / la normalisation risquent d'être fit sur tout le dataset avant le split. Le Pipeline **encapsule** tout et garantit que les transformations sont apprises **uniquement** sur les folds d'entraînement dans la CV.

### 6.9 Pondération de classe T0

```yaml
class_weights:
  t0_weight: 0.5
```

T0 pèse 0.5, T1/T2 pèsent 1.0. Le RF utilise ces poids pour ajuster son critère de split. Sans pondération, il convergerait vers « toujours T0 » (50% gratuit). Avec, il est poussé à vraiment différencier.

---

## Partie 7 — Résultats et interprétation

### 7.1 Métriques observées (sur 66 sujets complets)

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| Accuracy | ~45% | Sur 3 classes déséquilibrées, le hasard serait 33% (uniforme) ou 50% (toujours T0). |
| F1 weighted | ~44% | Cohérent avec l'accuracy → pas de tricherie « toujours T0 ». |
| Precision weighted | ~45% | |
| Recall weighted | ~45% | |

### 7.2 Pourquoi pas 90% ?

3 obstacles **documentés** et **inhérents au dataset** :

1. **Ambiguïté des labels T1/T2 selon le run** (cf. 2.4). Dans R04 c'est « main gauche », dans R06 c'est « deux poings ». Le modèle apprend un mélange incohérent.
2. **EEG de surface** — signal énormément bruité par les muscles, les yeux, le secteur 50 Hz. Même les meilleurs papiers scientifiques plafonnent à 70-75% sur ce dataset avec des modèles deep learning.
3. **Split par sujet** — on teste sur des gens **jamais vus**. La littérature souvent rapporte 85-90% avec un split par epoch (data leakage inclus). Notre 45% est l'équivalent honnête de leur 85%.

### 7.3 Ce que le modèle prédit effectivement

Matrice de confusion typique :

```
              T0_pred   T1_pred   T2_pred
T0_réel        52%       24%       24%
T1_réel        35%       38%       27%
T2_réel        35%       26%       39%
```

- Le modèle prédit **les 3 classes** (pas de collapse vers T0).
- T1 et T2 sont mieux reconnus qu'au hasard (38-39% > 33%).
- La confusion T1↔T2 reste forte (26-27%) — c'est là que la science ferait encore la différence.

### 7.4 Feature importance (RF)

Top attendu :

1. `diff_alpha` — la feature reine de la latéralisation
2. `C3_alpha`, `C4_alpha` — désynchronisation
3. `diff_beta` — rebond post-mouvement
4. Les autres suivent

Si `diff_alpha` n'est **pas** dans le top 3, c'est un drapeau rouge : soit un bug dans l'ordre du signal (5.4.4), soit la normalisation a gommé le signal utile.

---

## Partie 8 — Comment lancer le projet

### 8.1 Pré-requis

- Docker Desktop (Windows/Mac) ou Docker + Compose sur Linux
- ~5 Go d'espace disque (dataset + Parquet + checkpoints Spark)

### 8.2 Première fois (setup complet)

```bash
# 1. Lancer le cluster Spark + Jupyter
cd "Analyse Cerveau"
docker compose up -d

# 2. Télécharger le dataset PhysioNet (~550 Mo, ~3 min)
docker exec spark-master python /opt/spark/scripts/download_eeg.py

# 3. Ouvrir Jupyter
# → http://localhost:8889
# → Naviguer vers warehouse/etape1_poc/poc_eeg.ipynb
# → Run → Run All Cells
```

### 8.3 Runs suivants (tout est en cache)

```bash
docker compose up -d
# → Ouvrir http://localhost:8889
# → Le notebook saute les étapes d'ingestion (Parquet déjà générés)
```

### 8.4 UIs utiles pendant l'exécution

| URL | Quoi |
|-----|------|
| http://localhost:8889 | Jupyter Lab |
| http://localhost:8080 | Spark Master UI (workers, applications) |
| http://localhost:4040 | Spark Application UI (jobs, stages, tasks) |
| http://localhost:18080 | Spark History Server (runs passés) |
| http://localhost:8050 | Dashboard Dash/Plotly |

### 8.5 Lancer le dashboard

```bash
# Une fois le notebook exécuté (export.py a écrit les 4 Parquet)
docker exec spark-master python /opt/spark/dashboard.py
# → http://localhost:8050
```

### 8.6 Changer un hyperparamètre

**Zéro code Python à modifier.** Juste `scripts/config.yaml` :

```yaml
model:
  num_trees_grid: [100, 200, 500]   # tester des forêts plus grandes
  max_depth_grid: [5, 10, 20]
  num_folds: 5                       # CV plus stricte

frequency_bands:
  alpha: [7, 14]                     # bande alpha élargie
```

Relancer le notebook. C'est **tout**.

### 8.7 Déboguer

- **Spark UI (4040)** — onglet « Stages » montre quel job est lent. Onglet « SQL » montre le plan Catalyst.
- **Logs** — passer `logging.level: DEBUG` dans `config.yaml` pour tout voir.
- **Un sujet seulement** — dans un test, filtrer `df.filter(F.col('subject_id') == 'S001')` pour itérer vite.

---

## Annexe — Glossaire rapide

| Terme | Définition |
|-------|-----------|
| **Epoch** | Fenêtre temporelle découpée dans le signal (ici 2 s = 320 samples) |
| **ERD** | Event-Related Desynchronization : baisse de puissance alpha pendant l'imagerie |
| **ERS** | Event-Related Synchronization : rebond beta après le mouvement |
| **FFT** | Fast Fourier Transform — décompose un signal en fréquences |
| **Pandas UDF** | User-Defined Function vectorisé via Apache Arrow |
| **Shuffle** | Redistribution des données entre partitions après un `groupBy`/`join` |
| **Catalyst** | L'optimiseur de plans de requêtes de Spark SQL |
| **Window** | Agrégation qui garde toutes les lignes et ajoute des colonnes calculées par groupe |
| **Data leakage** | Fuite d'information du test vers le train qui surestime les performances |
| **Contralatéralité** | Hémisphère cérébral gauche ↔ côté droit du corps (et inversement) |

---

**Fin du guide.** Si tu as lu jusqu'ici, tu sais exactement ce que fait chaque ligne du projet et pourquoi.
