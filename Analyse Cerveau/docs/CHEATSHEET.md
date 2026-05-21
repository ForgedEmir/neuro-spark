# Cheatsheet Démo · à imprimer sur 1 page

## Avant l'oral (chez toi, la veille)

```bash
# Vérification que tout fonctionne
make demo        # attendre 60s warmup puis dashboard sur :8052
# Ctrl+C pour arrêter quand t'as vérifié visuellement
```

Le premier lancement entraîne un sklearn RF sur ~3700 epochs (~60 s).
Les suivants sont instantanés (modèle caché dans `data/models/sklearn_rf.pkl`).

## Jour J · Ordre des actions

| # | Action                                          | Quand                     |
|---|-------------------------------------------------|---------------------------|
| 1 | Ouvrir 2 terminaux côte à côte                  | Avant de monter           |
| 2 | Dans le 1er : `make demo`                       | Au début de la phase B    |
| 3 | Attendre « Dashboard EEG sur :8052 »            | ~15 s (cache présent)     |
| 4 | Ouvrir navigateur sur `http://localhost:8052`   | Dès que prêt              |
| 5 | Côté 2e terminal : `tail -f data/stream_eeg/spark.log` | Pour montrer Spark live |
| 6 | **Cliquer sur les boutons** dans l'ordre pédagogique : Repos → Mouvement réel → Mouvement imaginé | Pendant 5-7 min |
| 7 | Ctrl+C dans le 1er terminal → tout se ferme     | À la fin                  |

## 🎮 Deux modes au choix

### Mode INTERACTIF (par défaut) — pour expliquer
**3 gros boutons** dans le dashboard :
- 🧘 **Repos (T0)** : envoie un epoch de repos
- ✋ **Mouvement réel (T1)** : envoie un epoch de mouvement
- 🧠 **Mouvement imaginé (T2)** : envoie un epoch d'imagerie motrice

À chaque clic, un epoch est envoyé dans `data/stream_eeg/input/`. Spark le détecte (<2 s), prédit la classe, calcule la topographie alpha, et le dashboard se met à jour.

### Mode AUTO (case à cocher) — pour défendre le streaming
**Coche la checkbox « Mode automatique »** → le dashboard envoie un epoch toutes les 3 secondes en boucle (T0 → T1 → T2 → T0 → T2 → T1 → ...). C'est du **vrai streaming continu** : Spark voit défiler les epochs sans que tu touches à rien.

**Si un examinateur dit « clique-bouton c'est pas du streaming »** → tu coches la case, et il voit le flux continu qui arrive en boucle.

### Ordre suggéré pendant l'oral

**Phase 1 — montrer que c'est du streaming continu (mode AUTO, 30 s)** :
1. Coche **Mode automatique**
2. Laisse tourner 30 s : la topomap se met à jour toutes les 3 s, l'accuracy se construit
3. Dis : *« Là le système traite un flux continu, sans intervention humaine. C'est du Spark Structured Streaming pur. »*

**Phase 2 — décortiquer chaque classe (mode INTERACTIF, 3-4 min)** :
4. Décoche la case
5. Clique 🧘 **Repos** → topographie rouge (alpha haut)
6. Clique ✋ **Mouvement réel** → C3/C4 bleuissent
7. Clique 🧠 **Mouvement imaginé** → ⭐ même pattern → **ERD**
8. Clique 5-10 fois en variant pour montrer la prédiction live

## Timing oral (20 min)

| Min   | Bloc                                | Slides    |
|-------|-------------------------------------|-----------|
| 0–2   | Pitch & promesse BCI                | 1, 2      |
| 2–4   | Dataset PhysioNet Motor Imagery     | 2         |
| 4–7   | Architecture du pipeline            | 3, 4      |
| 7–14  | **DÉMO LIVE EEG BCI**               | 5, 6      |
| 14–18 | Décryptage technique + caveat       | 7         |
| 18–20 | Limites + futur + Q&A               | 8, 9      |

## Phrases-clés à dire

**Au démarrage de la démo** :
> « Là, Spark écoute. Il ne traite rien tant qu'aucun epoch n'arrive.
>   C'est du vrai event-driven, pas un cron déguisé. Pendant le warmup,
>   un sklearn RandomForest s'entraîne sur 3700 epochs de signaux EEG. »

**Pendant que tu cliques sur Repos** :
> « Là je dis à Spark : "voici un epoch où le sujet ne fait rien".
>   Regardez la topographie : alpha haut partout (rouge), le cerveau
>   est en mode veille. Le modèle prédit T0 — Repos. »

**Pendant que tu cliques sur Mouvement réel** :
> « Maintenant je lui envoie un epoch où le sujet bouge sa main. Regardez
>   les électrodes C3 et Cz et C4 (en noir, le motor cortex) — l'alpha
>   baisse, ça bleuit. Le modèle prédit T1. »

**Pendant que tu cliques sur Mouvement imaginé** ⭐️ :
> « Et voilà le punchline : le sujet n'a pas bougé. Il a **imaginé**.
>   Et pourtant la topographie montre la même signature : alpha bas sur
>   le cortex moteur. C'est l'**ERD** — Event-Related Desynchronization.
>   Documenté depuis Pfurtscheller 1999. Et Spark le détecte en 2 secondes. »

**Sur l'accuracy 100 %** (honnêteté scientifique) :
> « J'obtiens ~100 % sur S001 — mais S001 est dans le train set du sklearn,
>   donc c'est optimiste. Cross-sujet, on serait à 50-60 %, ce qui est
>   **le standard de la littérature BCI motor imagery** (très difficile
>   à cause de la variabilité inter-individus). »

## Si quelque chose plante

| Problème                        | Réponse                                                   |
|---------------------------------|-----------------------------------------------------------|
| Dashboard reste « en attente »  | `ls data/stream_eeg/output/predictions/` doit être >0     |
| Spark crashed                    | `tail data/stream_eeg/spark.log` puis `make demo` à nouveau |
| Port 8052 déjà pris              | `pkill -f dashboard_eeg` puis relancer                    |
| Warmup trop long (>3 min)        | `rm data/models/sklearn_rf.pkl` et relance                |
| Tout à zéro                      | `make eeg-reset && make demo`                             |

## Questions probables du jury

| Q                                          | R                                                              |
|--------------------------------------------|----------------------------------------------------------------|
| C'est vraiment du streaming ?              | Oui : `readStream` file source + watermark + checkpoint. Le dashboard a aussi un mode AUTO qui envoie en continu — coche la case pour montrer. |
| Le mode bouton, c'est pas batch déguisé ?  | Non. Source streaming = répertoire dynamique surveillé par Spark. Chaque epoch = un événement. C'est l'équivalent d'un capteur médical qui envoie une trame quand le sujet déclenche une tâche. |
| Pourquoi pas Kafka ?                       | File source = même API streaming, même garanties exactly-once. Pas de broker à gérer. Swap = 5 lignes. |
| Pourquoi sklearn et pas Spark.ml en live ? | Latence : 500 ms d'overhead JVM par epoch vs 10 ms sklearn. En BCI temps réel, chaque ms compte. |
| Comment ça scale ?                         | File source partitionnable, executors distribuent les fichiers. Code identique en cluster Spark. |
| 100 % accuracy, vraiment ?                 | Honnête : S001 est dans le train. Cross-sujet ~50-60 % (standard littérature BCI). |
| Pourquoi PhysioNet et pas plus moderne ?   | Classique reconnu (Schalk 2004). Permet de comparer avec d'autres travaux. EEGNet/ATCNet seraient le step suivant. |
| Pourquoi pas Flink ?                       | Spark déjà dans le stack (pipeline batch). Même API DataFrame batch ↔ stream → un seul code base. |
