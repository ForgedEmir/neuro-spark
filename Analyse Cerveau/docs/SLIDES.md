---
marp: true
theme: default
paginate: true
header: 'NeuroSpark · EEG BCI Live'
footer: 'UE28 Big Data · 2026'
style: |
  section { background: #f5f1e8; color: #2b2a26; font-family: Inter, system-ui, sans-serif; }
  h1, h2 { font-family: 'Fraunces', Georgia, serif; color: #2b2a26; }
  h1 { color: #c44536; }
  code { background: #f0eadf; padding: 2px 6px; border-radius: 4px; }
  pre { background: #fbf8f2; border-left: 4px solid #c44536; }
  .bleu { color: #5b8baf; }
  .rouge { color: #c44536; }
---

<!-- _class: lead -->
# NeuroSpark
## EEG Brain-Computer Interface Live

**Lire l'intention motrice en temps réel** à partir d'un EEG 64 électrodes.

Bonus à WESAD — démontre la **portée multi-domaine** du même stack streaming.

---

## 1 · La promesse BCI

Quand un sujet **imagine** bouger sa main droite :
- son cortex moteur gauche (électrode **C3**) montre une **baisse d'alpha** (ERD)
- on peut **détecter cette intention sans qu'il bouge**

→ C'est le principe des **interfaces cerveau-machine** (prothèses neurales, contrôle de fauteuils, jeux vidéo « pensés »).

**Notre défi** : décoder ce signal **en temps réel** via Spark Streaming.

---

## 2 · Dataset PhysioNet Motor Imagery

- 109 sujets, 64 électrodes EEG @ **160 Hz**
- Sujets imaginent bouger main (T1/T2) ou se reposent (T0)
- Classique de la **BCI** (Schalk et al., 2004)

| Label | Tâche                |
|-------|----------------------|
| **T0** | Repos               |
| **T1** | Mouvement réel      |
| **T2** | Mouvement imaginé   |

→ Notre modèle prédit **lequel des trois** depuis le signal EEG seul.

---

## 3 · Architecture

```
EDF PhysioNet ──► Kedro ingestion (batch) ──► Parquet
                                                 │
                                                 ▼
                                  stream_producer_eeg.py
                                  (1 epoch = 2s = 320 samples)
                                                 │
                                                 ▼
                                  data/stream_eeg/input/
                                                 │
   ┌─────────────────────────────────────────────┴───┐
   │  Spark Structured Streaming + foreachBatch       │
   │   • compute FFT band powers (C3/Cz/C4)           │
   │   • sklearn RF predict (log + scaler)            │
   │   • compute alpha power 64 electrodes            │
   └────────┬───────────────────────────┬─────────────┘
            ▼                           ▼
   predictions/                  topomap/
            │                           │
            └────────┬──────────────────┘
                     ▼
              Dashboard :8052
              • topographie cérébrale (Plotly + MNE 10-20)
              • bar chart probas
              • accuracy live
```

---

## 4 · Choix techniques notables

| Choix              | Pourquoi                                       |
|--------------------|------------------------------------------------|
| **sklearn live**   | 10 ms vs 500 ms (Spark.ml) → critique en BCI   |
| **log + StandardScaler** | FFT powers sur 6+ ordres → normalisation indispensable |
| **MNE 10-20**      | Positions cliniques standard pour le topomap   |
| **Plotly scatter** | Topomap interactif < latence MNE matplotlib    |
| **foreachBatch**   | Python pur pour la logique métier (FFT + ML)   |
| **Epoch 2 s**      | Compromis latence ↔ qualité spectrale FFT      |

---

## 5 · DÉMO LIVE

🎬 Une commande :

```bash
make demo-eeg
```

→ Dashboard sur `http://localhost:8052`

Ce qu'on observe :
1. La **topographie** se peint en temps réel selon les puissances alpha
2. La **prédiction** change à chaque epoch (toutes les 2s)
3. L'**accuracy glissante** se stabilise

---

## 6 · Ce que vous voyez (1/2)

### Au démarrage (~60s)

- Spark log : « Training sklearn RF on ~3700 epochs »
- Dashboard : « en attente du flux »

**Pourquoi ce délai ?** Le sklearn s'entraîne **automatiquement** sur les Parquet existants au démarrage (60 fichiers de sujets différents, ~3700 epochs).

C'est plus honnête qu'un modèle pré-entraîné statique — la pipeline est entièrement reproductible.

---

## 6 · Ce que vous voyez (2/2)

### En régime nominal

- **Topographie cérébrale** : le crâne avec 64 points colorés (rouge = alpha élevé, bleu = alpha faible = ERD)
- **Bar chart** : probabilités T0/T1/T2 en pourcentage
- **Accuracy live** : courbe rouge qui se stabilise vers 80-100%

### Phrase à dire au moment clé

> « Vous voyez ce point bleu sur **C3** (côté gauche du crâne) ? C'est l'**ERD** — la signature neurophysiologique de l'intention motrice. **Documenté depuis Pfurtscheller 1999**, et Spark le détecte ici en moins de 2 secondes. »

---

## 7 · Honnêteté scientifique

### L'accuracy 100 % a un caveat

- Le sklearn RF est entraîné sur 60 fichiers Parquet
- Si on stream **S001/R04**, S001 est dans le train set
- → accuracy artificiellement élevée

### Réalité BCI (cross-sujet)
- ~50-60 % d'accuracy sur un sujet inconnu
- C'est le **standard de la littérature** (BCI motor imagery est notoirement difficile)
- Pour faire mieux : Deep Learning (EEGNet, ATCNet) ou plus de données par sujet

À assumer dans la démo.

---

## 8 · Limites + futur

**Limites** :
- Entraînement intra-sujet → optimiste
- 16 features uniquement (3 canaux × 4 bandes + 4 lat.)
- Pas de filtrage spatial (CSP, Riemannian geometry)

**Si on continuait** :
- 🧠 **EEGNet** ou **ATCNet** (deep learning BCI state-of-the-art)
- 🌐 **CSP** (Common Spatial Patterns) → +10-15% d'accuracy
- 🎮 Brancher sur un avatar / jeu vidéo en live
- 🎯 Online learning : le modèle s'adapte au sujet en cours

---

<!-- _class: lead -->
## Punchline EEG

> Vous venez de voir un cerveau qui parle, et un système Big Data qui l'écoute.
>
> **Pas de mouvement, pas de mot, pas de toucher.**
> Juste 64 électrodes, Spark, et 2 secondes de latence.
>
> Le futur des interfaces homme-machine passe par là.

**Questions ?**
