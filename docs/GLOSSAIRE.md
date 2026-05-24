# Glossaire — à relire avant l'oral

## 1 · Vocabulaire EEG

| Terme       | Définition courte |
|-------------|-------------------|
| **EEG**     | Électroencéphalogramme : signal électrique du cortex via électrodes posées sur le crâne |
| **64 électrodes** | Système international 10-20, positions standardisées (C3, Cz, C4, etc.) |
| **Epoch**   | Fenêtre de signal de durée fixe (ici 2 secondes = 320 samples @ 160 Hz) |
| **FFT**     | Fast Fourier Transform : décompose un signal temporel en fréquences |
| **Bandes**  | Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-80 Hz) |
| **Band power** | Énergie du signal dans une bande de fréquences donnée |
| **C3 / Cz / C4** | Électrodes au-dessus du cortex moteur (gauche / centre / droite) |

## 2 · Vocabulaire BCI (Brain-Computer Interface)

| Terme       | Définition courte |
|-------------|-------------------|
| **BCI**     | Interface qui décode l'activité cérébrale pour commander un appareil |
| **Motor imagery** | Imagerie motrice : imaginer un mouvement sans le faire. Active le cortex moteur. |
| **T0 / T1 / T2** | Labels PhysioNet : Repos / Mouvement réel / Mouvement imaginé |
| **ERD** | *Event-Related Desynchronization* : baisse d'alpha (8-13 Hz) quand on imagine bouger |
| **ERS** | *Event-Related Synchronization* : augmentation d'alpha au repos |
| **Latéralisation** | Asymétrie hémisphérique : C3 (gauche) vs C4 (droit). Imaginer main droite → ERD sur C3 |

## 3 · Le truc fondamental à comprendre

**Quand tu imagines bouger ta main droite, l'alpha *diminue* sur l'électrode C3 (cortex moteur gauche).**

Ce phénomène (l'**ERD**) est ultra reproductible et c'est ce qui permet aux BCI de décoder l'intention motrice. Notre dashboard topographique le montre **en direct** : tu vois le côté gauche du crâne « bleuir » (= alpha bas) quand le sujet imagine bouger sa main droite.

---

## 4 · Le dataset PhysioNet Motor Imagery

- 109 sujets, 64 électrodes EEG @ **160 Hz**
- Schalk et al. 2004 — référence historique en BCI
- 14 runs par sujet : R01-R02 = baseline, **R03-R14 = motor imagery tasks**
- Chaque run dure ~125 s, contient 30 epochs (15 actives + 15 repos)

**Types de runs utilisés** :
- R03/R07/R11 : mouvement réel des poings
- **R04/R08/R12** : mouvement imaginé des poings (notre cible)
- R05/R09/R13 : mouvement réel pieds vs poings
- R06/R10/R14 : mouvement imaginé pieds vs poings

---

## 5 · Pipeline EEG complet (de la donnée à la prédiction)

```
EDF PhysioNet (160 Hz, 64 channels)
      │  kedro run --pipeline ingestion (batch, déjà fait)
      ▼
data/parquet/  (1 fichier par sujet × run)
      │  scripts/stream_producer_eeg.py
      │  (1 epoch = 2s = 320 samples → 1 fichier Parquet)
      ▼
data/stream_eeg/input/
      │  Spark Structured Streaming
      ▼
foreachBatch (Python pur) :
   1. compute_features    → 16 features FFT (C3/Cz/C4 × 4 bandes + 4 lat.)
   2. sklearn RF predict  → T0 / T1 / T2  +  probabilités
   3. compute_alpha_topomap → puissance alpha sur 64 électrodes
      │
      ▼
data/stream_eeg/output/predictions/  +  topomap/
      │  polling 1.5 s (Dash dcc.Interval)
      ▼
Dashboard sur :8052
   - topographie cérébrale (Plotly scatter + positions MNE 10-20)
   - bar chart probas T0/T1/T2
   - accuracy glissante
```

---

## 6 · Vocabulaire Big Data utilisé

| Terme | Définition |
|-------|------------|
| **Spark Structured Streaming** | Moteur de traitement de flux d'Apache Spark. Même API DataFrame que le batch. |
| **File source** | Source qui surveille un répertoire ; chaque nouveau fichier devient un micro-batch. |
| **Micro-batch** | Petit paquet de données traité atomiquement (pas event-par-event). |
| **Parquet** | Format colonnaire compressé. Standard Big Data. |
| **`foreachBatch`** | « Pour chaque micro-batch, fais ce que tu veux » → on écrit du Python pur (FFT + sklearn). |
| **`checkpointLocation`** | Répertoire où Spark sauve l'état pour reprendre après crash. |
| **`uirevision`** | Astuce Plotly pour ne pas redessiner toute la figure à chaque update → pas de flicker. |
| **Kedro** | Framework Python pour organiser ses pipelines de data science. |
| **MLflow** | Tracking d'expériences ML (utilisé pour le pipeline batch). |
| **DVC** | Git pour la donnée (Data Version Control). |

---

## 7 · Phrases-types pour le jury

### Sur le streaming
> « C'est du **vrai streaming**, pas un cron déguisé. Spark `readStream` ouvre une boucle de micro-batches qui scrute le filesystem ; si je coupe le producer, Spark reste à l'écoute. Si je le relance, il reprend là où il s'était arrêté grâce au **checkpoint**. »

### Sur le choix sklearn vs Spark.ml
> « J'utilise sklearn dans `foreachBatch` plutôt que le CrossValidatorModel Spark, pour une raison simple : la **latence**. Prédire 1 epoch via Spark coûte ~500 ms d'overhead JVM, sklearn fait ça en 10 ms. En BCI temps réel, chaque milliseconde compte. »

### Sur le pipeline ML
> « Le sklearn est entraîné automatiquement au démarrage du streaming sur 60 fichiers Parquet (~3700 epochs). C'est une **pipeline log10 + StandardScaler + RandomForest** — la log-transformation est cruciale parce que les FFT band powers s'étalent sur 6 ordres de grandeur. »

### Sur la topographie
> « Vous voyez le crâne avec ses 64 électrodes ? Les positions viennent du **standard 10-20** via MNE — c'est le système clinique de référence. Quand le sujet imagine sa main droite, l'alpha baisse sur C3 (cortex moteur gauche) — c'est l'**ERD** documenté en clinique depuis Pfurtscheller 1999. »

### Sur l'accuracy (honnêteté scientifique)
> « J'observe ~100 % sur ce sujet, mais c'est S001 qui est dans le train set. Honnêtement, sur un sujet inconnu, on serait plutôt à 50-60 % — ce qui est le **standard de la littérature BCI motor imagery** : c'est notoirement difficile parce qu'il y a une grande variabilité inter-sujets. Pour faire mieux, il faudrait du Deep Learning (EEGNet, ATCNet) ou des features spatiales (CSP). »

### Sur l'architecture
> « Le file source de Spark **est** une vraie API streaming, avec les mêmes garanties exactly-once via checkpoint + file metadata. Si on voulait passer à Kafka en prod, on change **uniquement** le bloc `readStream` ; le reste de la pipeline est inchangé. C'est la force de Spark Structured Streaming : source-agnostique. »
