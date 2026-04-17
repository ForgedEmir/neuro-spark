# Rapport Réflexif — PoC Analyse EEG avec Apache Spark
**UE28 Big Data — HELMo 2025-2026**  
**Étudiant :** Emir Mahemas  
**Date de remise :** 19 avril 2026  
**Projet :** Analyse d'imagerie motrice EEG en environnement distribué

---

## 1. Introduction et contexte

Ce projet de Proof of Concept (PoC) s'inscrit dans le cadre du cours de Big Data. L'objectif était de construire, de bout en bout, un pipeline de traitement de données massives appliqué à des signaux EEG (électroencéphalogramme). Le dataset utilisé est le **EEG Motor Movement/Imagery Dataset** de PhysioNet, constitué de 109 sujets enregistrés avec 64 électrodes à 160 Hz, soit environ 15 millions de lignes une fois converti en Parquet.

Le défi du projet n'était pas uniquement technique. Il consistait aussi à comprendre pourquoi on utilise chaque outil, et pas juste comment l'utiliser — une distinction que j'ai réalisée en cours de route.

---

## 2. Architecture et environnement technique

L'environnement de travail reposait sur un cluster Spark distribué orchestré par Docker :

| Composant | Rôle |
|---|---|
| `spark-master` | Coordinateur du cluster + Jupyter (port 8889) |
| `spark-worker` | Nœud d'exécution (8 GB RAM alloués) |
| `spark-history` | Interface historique des jobs |
| Volume `/opt/spark/data/` | Stockage partagé EDF + Parquet |

**Stack logicielle :** Python 3, PySpark 3.5.5, MNE, NumPy, Pandas, Plotly, MLlib  
**Format de données :** EDF (raw) → Parquet (traitement distribué)

---

## 3. Travail réalisé — pipeline complet

### Partie A — Conversion EDF → Parquet

La première étape était de rendre les données compatibles avec Spark. Les fichiers `.edf` (format médical) ne sont pas lisibles directement par Spark, j'ai donc utilisé la librairie MNE pour les lire et les convertir en Parquet.

Chaque ligne du Parquet résultant correspond à **un échantillon temporel** (1/160 seconde), avec comme colonnes : `subject_id`, `run_id`, `time`, `task_label`, et les 64 canaux EEG. La fonction `edf_to_dataframe()` associe également chaque sample à son label de tâche (T0/T1/T2) en utilisant les annotations temporelles du fichier EDF.

**Résultat :** 783 fichiers Parquet créés, 0 erreur, couvrant 66 sujets et 12 runs moteurs chacun (R03 à R14).

### Partie B — Exploration avec Spark

La SparkSession est le point d'entrée du cluster. J'ai appris une notion importante ici : la **lazy evaluation**. Quand on écrit `df = spark.read.parquet(...)`, Spark ne fait rien. Il construit un plan d'exécution. C'est seulement quand on appelle une action (`.count()`, `.show()`) que le calcul se déclenche réellement.

**Résultat exploration :**  
- 15 264 320 lignes, 68 colonnes  
- Distribution des tâches : T0 (repos) = 50%, T1 (imagerie main gauche) = 25%, T2 (imagerie main droite) = 25%  
- Ce déséquilibre allait causer des problèmes en classification — j'y reviens plus loin.

### Partie C — Feature Engineering (FFT spectrale)

C'est la partie la plus complexe du projet. L'idée était de ne pas donner le signal brut au modèle (trop bruité, trop long), mais d'en extraire des **caractéristiques fréquentielles** : la puissance dans les bandes Alpha (8–13 Hz), Beta (13–30 Hz) et Gamma (30–80 Hz).

Le processus en trois étapes :
1. **Découpage en epochs** — fenêtres de 2 secondes (320 samples) : `epoch_id = floor(time / 2)`
2. **Collecte des samples par epoch** — avec `collect_list()` qui regroupe les 320 valeurs d'un canal dans une liste
3. **Application de la FFT** — via une UDF Python qui calcule la puissance spectrale dans chaque bande

La **FFT (Transformée de Fourier Rapide)** décompose un signal temporel en composantes fréquentielles. `np.fft.rfft()` retourne les amplitudes pour chaque fréquence ; on élève au carré pour obtenir la puissance, et on moyenne les puissances dans la plage voulue (masque booléen sur le tableau de fréquences).

**Résultat :** 48 069 epochs, chacune décrite par 192 features (64 canaux × 3 bandes).

### Partie D — Classification MLlib (RandomForest)

Le pipeline MLlib comprenait trois étapes enchaînées :
- `StringIndexer` : convertit T0/T1/T2 en entiers 0/1/2
- `VectorAssembler` : fusionne les colonnes numériques en un vecteur dense
- `RandomForestClassifier` : 100 arbres, profondeur maximale 10

**Premier essai (192 features, sans pondération) :**  
Accuracy : ~51,78%. En apparence acceptable, mais la matrice de confusion révélait que le modèle prédisait T0 pour presque tout. L'accuracy était trompeuse — 50% des données étant T0, prédire toujours T0 donne mécaniquement 50% de précision.

**Deuxième essai (avec pondération T0=0.33) :**  
Accuracy : 30,93% — pire que le hasard (33%). En réduisant trop le poids de T0, le modèle l'a complètement ignoré. L'inverse du problème précédent.

**Troisième essai (canaux moteurs C3/Cz/C4 uniquement, T0=0.5) :**  
Réduction de 192 à 9 features, en gardant uniquement les canaux du cortex moteur. Le bruit des 61 autres canaux (zones frontale, occipitale, temporale) masquait le signal utile. Les résultats de cette version sont en cours d'évaluation.

### Partie E — Dashboard Plotly

Trois visualisations ont été produites et sauvegardées en HTML (accessible hors notebook) :
- **E1** — Signal brut du canal Cz au cours du temps, coloré par tâche
- **E2** — Matrice de confusion interactive (T0/T1/T2 prédit vs réel)
- **E3** — Puissance Alpha moyenne par canal moteur (C3/Cz/C4) selon la tâche

Le graphique E3 est scientifiquement le plus intéressant : en neurosciences, le phénomène d'**Event-Related Desynchronization (ERD)** prédit que lors d'une imagerie motrice, la puissance Alpha diminue sur le cortex controlatéral (imaginer la main droite → C3 gauche s'active, C4 droit se désynchronise).

---

## 4. Difficultés rencontrées et solutions

### 4.1 Noms de colonnes avec points (Spark AnalysisException)

Les fichiers EDF originaux utilisent des noms de canaux comme `C3..` ou `Cz..`. Dans Spark, le point (`.`) est un séparateur de namespace, interprété comme "colonne imbriquée". Résultat : toute opération sur ces colonnes levait une `AnalysisException`.

**Solution :** Renommage systématique de toutes les colonnes au début de la Partie C :
```
C3..  → C3
Cz..  → Cz
Fc5.  → Fc5
```
**Leçon retenue :** Toujours inspecter les noms de colonnes avant de lancer des transformations Spark. Les caractères spéciaux (`.`, ` `, `-`) sont des pièges courants.

### 4.2 Le déséquilibre de classes (class imbalance)

T0 représente 50% du dataset — deux fois plus que T1 ou T2. Un RandomForest non pondéré "apprend" qu'il a intérêt à prédire T0 par défaut, car ça maximise son score sur le jeu d'entraînement.

J'ai d'abord sur-corrigé avec `weight=0.33` pour T0, ce qui a fait l'inverse : le modèle fuyait T0. La bonne approche est une pondération proportionnelle à l'inverse de la fréquence (`0.5` pour T0, `1.0` pour T1/T2).

**Leçon retenue :** L'accuracy seule ne suffit pas à évaluer un modèle sur des classes déséquilibrées. La matrice de confusion est indispensable.

### 4.3 Jupyter non actif dans Docker

Lors d'une reconnexion VS Code, le serveur Jupyter n'était plus actif dans le conteneur. La commande `jupyter notebook list` retournait vide. Il a fallu relancer manuellement le serveur avec les bonnes options (`--allow-root --ip=0.0.0.0`), puis remplacer le hostname du conteneur par `localhost` dans l'URL.

**Leçon retenue :** Dans un environnement Docker, ne pas supposer que les processus persistent entre les sessions. Vérifier l'état avant de travailler.

### 4.4 "Spark natif" mal interprété

J'ai cru que "Spark natif" voulait dire "sans aucune bibliothèque externe (NumPy, Pandas)". Ce n'est pas le cas. Spark natif signifie utiliser l'API DataFrame/MLlib de Spark — pas du Pandas pour traiter 15M de lignes. NumPy à l'intérieur d'une UDF Spark est tout à fait standard et attendu.

---

## 5. Ce que j'ai compris (concepts clés)

**Lazy evaluation Spark :** Spark ne calcule rien immédiatement. Il construit un plan d'exécution logique (DAG), optimise, puis exécute uniquement quand une action est appelée. C'est ce qui permet de distribuer efficacement le travail.

**Parquet vs CSV :** Le format Parquet est columnar — les données d'une colonne sont stockées ensemble sur disque. Pour Spark, qui sélectionne souvent quelques colonnes sur des millions de lignes, c'est beaucoup plus efficace que CSV (row-based).

**FFT et bandes EEG :** La Transformée de Fourier décompose un signal temporel en fréquences. Pour un signal EEG, certaines plages de fréquences ont une signification physiologique (Alpha = relaxation, Beta = concentration, Gamma = traitement cognitif).

**Pipeline MLlib :** Un Pipeline enchaîne des transformateurs et estimateurs. L'avantage est qu'on peut appliquer exactement la même chaîne sur le jeu de test que sur le jeu d'entraînement, sans risque de fuite de données (data leakage).

**Motor cortex et imagerie motrice :** Les canaux C3 et C4 correspondent respectivement au cortex moteur gauche et droit. Imaginer un mouvement de la main droite active C3 (côté gauche du cerveau contrôle le côté droit du corps). C'est ce qui rend T1 et T2 potentiellement distinguishables par l'EEG.

---

## 6. Résultats et analyse critique

| Version | Features | Pondération T0 | Accuracy | Remarque |
|---|---|---|---|---|
| Baseline (prédire T0 toujours) | — | — | ~50% | Triche — ne prédit que T0 |
| RF naïf | 192 (64 canaux) | Aucune | 51,78% | Même problème, matrice révélatrice |
| RF pondéré agressif | 192 (64 canaux) | 0.33 | 30,93% | Sous le hasard — T0 fui |
| **RF moteur pondéré** | **9 (C3/Cz/C4)** | **0.5** | **41,82%** | **Bat le hasard, 3 classes prédites** |
| Baseline aléatoire théorique | — | — | 33,33% | Référence |

**Performance par classe (version finale) :**

| Classe | Rappel | Précision | Interprétation |
|---|---|---|---|
| T0 (repos) | 54,2% | 55,7% | Bien reconnu — signal de fond distinct |
| T1 (imagerie main gauche) | 37,9% | 27,9% | Difficile — confondu avec T0 |
| T2 (imagerie main droite) | 19,9% | 28,4% | Le plus dur — très proche de T1 |

Les deux premiers essais illustrent un problème classique sur données déséquilibrées. L'accuracy de 51,78% semblait correcte mais cachait un modèle inutile : la matrice de confusion montrait que T1 et T2 n'étaient presque jamais prédits. Réduire le poids de T0 à 0.33 a provoqué l'effet inverse (accuracy 30,93%). La valeur 0.5 — inversement proportionnelle à la fréquence de T0 (~50%) — s'est avérée correcte.

La réduction aux 9 features du cortex moteur est justifiée scientifiquement : les 61 autres canaux (zones frontale, occipitale, pariétale) n'apportent pas d'information discriminante pour distinguer l'imagerie main gauche / main droite. Ils ajoutaient du bruit qui noyait le signal utile.

La confusion persistante entre T1 et T2 est attendue : les deux correspondent à une imagerie motrice unilatérale, et leurs patterns EEG sont très proches avec seulement 3 canaux et de la puissance spectrale brute comme features.

Des recherches publiées sur ce dataset atteignent 70–85% d'accuracy avec des méthodes plus avancées (CSP — Common Spatial Patterns, filtrage par sujet, réseaux LSTM). Dans le cadre d'un PoC, valider le pipeline complet et comprendre les limites méthodologiques est l'objectif premier.

---

## 7. Perspectives et améliorations possibles

**Normalisation par sujet (StandardScaler)** — Les signaux EEG varient fortement d'une personne à l'autre (amplitude, bruit). Normaliser les features par sujet avant d'entraîner améliorerait la généralisation du modèle.

**Cross-validation par sujet** — Le split 80/20 aléatoire mélange les epochs d'un même sujet entre train et test. Idéalement, il faudrait entraîner sur 52 sujets et tester sur 14 (stratification sujet-indépendante).

**Features temporelles complémentaires** — En plus de la puissance spectrale, la variance du signal, la corrélation entre C3 et C4, ou les coefficients de cohérence pourraient enrichir les features.

**Fokus sur les runs correspondants** — Les runs R04, R08, R12 correspondent spécifiquement à l'imagerie des deux mains (pas les poings). Filtrer sur ces runs uniquement améliorerait la cohérence des labels T1/T2.

---

## 8. Conclusion

Ce PoC m'a permis de construire un pipeline Big Data complet sur des données réelles et volumineuses. Au-delà du code, j'ai surtout compris les raisons derrière chaque choix technique : pourquoi Parquet plutôt que CSV, pourquoi le pipeline MLlib plutôt qu'un script séquentiel, pourquoi la matrice de confusion plutôt que l'accuracy seule.

Les difficultés rencontrées — noms de colonnes, déséquilibre de classes, environnement Docker — sont représentatives des problèmes réels en production. Savoir les diagnostiquer est aussi important que savoir les résoudre.

L'objectif de l'étape suivante (Data Engineering, mai 2026) sera de rendre ce pipeline robuste, paramétrable et déployable, plutôt qu'un notebook exploratoire.

---

*Rapport généré le 19 avril 2026 — UE28 Big Data, HELMo Liège*
