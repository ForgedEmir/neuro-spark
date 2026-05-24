# Oral HELMo · Contenu de la démo (20 min)

> Support qui suit **exactement** les 4 points de l'énoncé. Lis-le à voix haute
> en répétant 2-3 fois. Les métriques sont **réelles** (mesurées sur ta machine).

---

## 1 · Démo de votre fonctionnalité streaming [5 min]

**Ce que tu fais à l'écran** :

```bash
make demo                 # Spark + dashboard sur http://localhost:8052
```

**Déroulé parlé** (5 min) — scénario en 3 actes :

0. **Contexte (30 s)** :
   > « Mon système est une interface cerveau-machine en temps réel. Je stream
   >   des signaux EEG (64 électrodes, 160 Hz) dans Spark, qui décode l'intention
   >   motrice du sujet : repos, mouvement réel, ou mouvement imaginé. »

**ACTE 1 — Le modèle lit dans les pensées (S001, défaut) [1.5 min]**
   Clique les 3 boutons (sujet S001 chargé par défaut) :
   - 🧘 **Repos** → zone moteur centrale rouge (alpha élevé) → prédit Repos ✓
   - ✋ **Mouvement réel** → zone moteur bleue (ERD) → prédit Mouvement réel ✓
   - 🧠 **Mouvement imaginé** → même signature bleue **sans bouger** → prédit T2 ✓
   > « Le sujet imagine, ne bouge pas, et Spark décode son intention. Regardez
   >   le pipeline en haut : epoch → Spark → FFT → ML → résultat, ~1 seconde. »

**ACTE 2 — Mais c'est trop beau (révélation) [1 min]**
   > « 100 % d'accuracy, c'est suspect. En réalité S001 fait partie du jeu
   >   d'entraînement — c'est du **data leakage**. Voyons un sujet jamais vu. »
   - Bascule le dropdown sur **S010 (hors-train)**

**ACTE 3 — Le flux continu + fenêtre glissante Spark (2 min)**
   - Coche **▶ Démarrer le flux continu** → un epoch arrive toutes les 3 s
   > « Là, plus d'intervention : c'est un flux continu, comme un vrai capteur. »
   - Pointe le **moniteur temps réel** (timeline) :
   > « Chaque barre est un epoch placé dans le temps — gris repos, bleu mouvement,
   >   rouge imaginé. On voit le flux défiler. »
   - Pointe la **fenêtre glissante Spark** (courbe du bas) :
   > « Et voici LE concept central du streaming : Spark agrège en **fenêtre
   >   glissante** — fenêtre de 8 secondes, qui glisse toutes les 2 secondes,
   >   avec un **watermark** de 5 s pour les événements en retard. La courbe
   >   montre le % du temps où le sujet a une intention motrice. C'est une
   >   2e query Spark qui lit la sortie de la 1re : un **chaînage de streams**. »

**(optionnel) Bascule sur S010 (hors-train) [30 s]**
   > « Si je change pour un sujet jamais vu, l'accuracy tombe à 50 % — c'est la
   >   vraie performance BCI cross-sujet. Data leakage vs généralisation. »

> ⭐ La fenêtre glissante (window + watermark) est **impossible en batch** :
>   c'est ce qui prouve que c'est du vrai streaming.

---

## 2 · Explication technique [5 min]

### a) Choix faits

| Choix | Pourquoi |
|-------|----------|
| **Spark Structured Streaming** (file source) | Vraie API streaming, mêmes garanties exactly-once que Kafka, sans broker à gérer. Si on voulait Kafka, on changerait juste le bloc `readStream`. |
| **`foreachBatch`** | Permet d'exécuter du Python pur (FFT + sklearn) sur chaque micro-batch, ce qu'un sink natif ne permet pas. |
| **sklearn (pas Spark.ml) pour la prédiction** | Latence : prédire 1 epoch via Spark.ml coûte ~500 ms d'overhead JVM ; sklearn fait ça en <10 ms. Critique en temps réel. |
| **2 queries chaînées** (prediction → windowed) | La 1re query prédit ; la 2e lit les prédictions en stream et agrège en fenêtre glissante. Pattern réaliste de pipeline streaming multi-étapes. |
| **`window` + `watermark`** | Cœur de Spark Structured Streaming : agrégation sur fenêtres temporelles glissantes, avec tolérance aux retards. **Impossible en batch.** |
| **Epoch de 2 s** | Compromis : assez long pour une FFT fiable (résolution fréquentielle), assez court pour de la réactivité. |
| **Z-score par électrode** | Normalise la topographie : chaque électrode comparée à sa propre baseline → pas de biais des électrodes marginales. |
| **`checkpointLocation`** | Tolérance aux pannes : si Spark crashe, il reprend où il s'était arrêté. |

### b) Métriques (CHIFFRES RÉELS mesurés)

| Métrique | Valeur mesurée | Commentaire |
|----------|----------------|-------------|
| **Latence end-to-end** | **~1 s** (médiane 0.97 s) | du moment où l'epoch est écrit au moment où la prédiction est disponible |
| **Latence cold-start** | ~4.5 s (1er epoch) | démarrage du premier micro-batch Spark |
| **Throughput** | **~0.9 epoch/s** | = 2 s de signal EEG traité par seconde (temps réel) |
| **Taille d'un micro-batch** | **121 Ko** (Parquet compressé) | 1 epoch = 320 samples × 64 électrodes = 2 s de signal |
| **Données brutes / batch** | 160 Ko (avant compression) | compression Snappy ≈ 25 % |
| **Trigger** | `processingTime = 1 s` | Spark vérifie le répertoire chaque seconde |
| **`maxFilesPerTrigger`** | 1 | un fichier = un micro-batch (déterministe) |
| **Fenêtre glissante** | window 8 s / slide 2 s | 2e query Spark, agrège l'intention motrice |
| **Watermark** | 5 s | tolérance aux événements en retard |

> **Phrase à dire** :
> « Ma latence end-to-end est d'environ 1 seconde, mon throughput de ~0.9 epoch
>   par seconde, soit du temps réel strict puisqu'un epoch représente 2 secondes
>   de signal. Chaque micro-batch fait 121 Ko. »

### c) Limites constatées

1. **Throughput bridé volontairement** : `maxFilesPerTrigger=1` limite à ~1 epoch/s. C'est un choix pour la démo (1 événement = 1 traitement visible). En production, on l'augmenterait pour scaler.
2. **Accuracy modeste cross-sujet (~50 %)** : sur un sujet jamais vu, le modèle est à 50 % (vs 33 % au hasard). C'est le **standard de la littérature BCI** — la variabilité inter-sujets est énorme. Un sujet du train donne 100 % (data leakage que je montre exprès).
3. **Modèle simple** : RandomForest sur 16 features FFT. Le state-of-the-art (EEGNet, ATCNet, CSP) ferait +10-20 %, mais demande du deep learning.
4. **Mono-nœud** : `local[2]`. La pipeline scale à un cluster sans changement de code, mais je ne l'ai pas testé en distribué.
5. **Cold start** : le premier epoch met ~4.5 s (initialisation du micro-batch). Négligeable ensuite.

---

## 3 · Appréciation des technologies [5 min]

> Pour chaque techno : ce que j'en ai pensé + si je la réutiliserais.

### Apache Spark (Structured Streaming)
- **Apprécié** : l'API DataFrame est identique en batch et en streaming → un seul code à apprendre. Le file source m'a permis de faire du vrai streaming sans monter un Kafka.
- **Moins apprécié** : l'overhead JVM est lourd pour de petites tâches (d'où mon choix sklearn). Les messages d'erreur sont verbeux et durs à lire.
- **Réutiliser ?** ✅ Oui pour du gros volume. Pour des micro-tâches temps réel, je regarderais des alternatives plus légères.

### Dash / Plotly
- **Apprécié** : très rapide pour faire un dashboard interactif en pur Python, sans toucher au JS. Les callbacks sont intuitifs, `dcc.Interval` rend le live facile.
- **Moins apprécié** : le `uirevision` et la gestion du flicker demandent du tâtonnement. Pas idéal pour du très haute fréquence.
- **Réutiliser ?** ✅ Absolument, pour tout prototype de visualisation de données.

### Data engineering / scaffolding : Kedro
- **Apprécié** : structure le projet en pipelines clairs (ingestion → features → training). Le catalog déclaratif (`catalog.yml`) sépare le code de la config. Très propre.
- **Moins apprécié** : courbe d'apprentissage au début, beaucoup de conventions à intégrer.
- **Réutiliser ?** ✅ Oui pour tout projet data sérieux — ça force de bonnes pratiques.

### MLOps : MLflow + DVC
- **MLflow** : tracking des expériences (modèles, métriques) — pratique pour comparer les runs. Réutiliser ✅.
- **DVC** : versioning des données hors Git. Utile mais un peu lourd pour un projet solo. Réutiliser ⚠️ selon la taille du projet.

### Streaming : file source Spark
- **Apprécié** : zéro infra à gérer, parfait pour apprendre les concepts (watermark, checkpoint, micro-batch).
- **Limite** : pas adapté à de la vraie production multi-source (là il faudrait Kafka).
- **Réutiliser ?** ✅ Pour prototyper du streaming. Pour la prod, je passerais à Kafka.

---

## 4 · 2 à 5 choses que vous avez apprises [5 min] — NON TECHNIQUES

> ⚠️ L'énoncé insiste : **apprentissages non techniques**. Analyse réflexive.
> Voici des pistes — **personnalise-les avec TES vraies expériences** sur ce projet.

### Piste 1 · Sur ma méthode de travail
> « J'ai appris à **itérer par petits incréments testés** plutôt que de tout
>   construire d'un coup. Chaque fois que j'ai voulu faire trop à la fois
>   (ex: combiner deux datasets), je me suis perdu. Avancer brique par brique,
>   en testant à chaque étape, m'a fait gagner du temps. »

### Piste 2 · Sur mes réflexes / résolution de problèmes
> « J'ai appris à **lire les messages d'erreur en entier** au lieu de paniquer.
>   Mes bugs (type mismatch Parquet, normalisation, data leakage) venaient
>   tous d'une cause précise que l'erreur ou les données expliquaient.
>   Diagnostiquer avant de coder une solution. »

### Piste 3 · Sur mon autonomie
> « J'ai appris à **chercher la doc et les bonnes pratiques** par moi-même
>   (docs Spark, forums Dash) plutôt que d'attendre qu'on me donne la réponse.
>   Savoir formuler la bonne question est devenu un réflexe. »

### Piste 4 · Sur ma capacité à gérer les difficultés
> « Quand mon accuracy était à 100 % partout, j'ai d'abord cru que c'était
>   bon — puis j'ai compris que c'était un **data leakage**. J'ai appris à
>   **me méfier des résultats trop beaux** et à les remettre en question
>   plutôt que de les accepter. L'esprit critique sur ses propres résultats. »

### Piste 5 · Sur ma gestion du stress / du temps
> « Avec une deadline serrée, j'ai appris à **prioriser ce qui marche** plutôt
>   que de chercher la perfection. Garder un système fonctionnel à chaque étape
>   (un filet de sécurité) plutôt que de tout casser pour une amélioration
>   risquée. Mieux vaut une démo simple qui marche qu'une démo ambitieuse qui plante. »

> **Choisis-en 2 à 4 et raconte une anecdote concrète du projet pour chacune.**
> Le jury veut du vécu, pas des généralités.

---

## Récap timing

| Bloc | Durée | Support |
|------|-------|---------|
| 1. Démo streaming | 5 min | dashboard `:8052` |
| 2. Technique (choix/métriques/limites) | 5 min | ce doc + code |
| 3. Appréciation technos | 5 min | ce doc |
| 4. Apprentissages non techniques | 5 min | tes anecdotes |
| **Total** | **20 min** | |

## Métriques à connaître par cœur (point 2)

- **Latence : ~1 seconde** end-to-end
- **Throughput : ~0.9 epoch/s** = 2 s de signal EEG/s (temps réel)
- **Micro-batch : 121 Ko** (320 samples × 64 électrodes = 2 s)
- **Trigger : 1 s** · **maxFilesPerTrigger : 1**
