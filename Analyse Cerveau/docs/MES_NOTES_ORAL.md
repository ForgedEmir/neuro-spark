# Mes notes pour l'oral (20 min)

> Simple mais complet. Je relis, je dis avec mes mots. 4 blocs × 5 min.

---

## 🎬 1. Démo de ma fonctionnalité streaming [5 min]

**Ce que je dis en lançant `make demo` :**

> « J'ai construit une interface cerveau-machine en temps réel. Des signaux
>   EEG — l'activité électrique du cerveau, 64 capteurs — arrivent en continu
>   dans mon système, et celui-ci devine en direct ce que fait le sujet :
>   repos, mouvement réel, ou mouvement imaginé. »

**Ce que je montre, dans l'ordre :**

1. Je clique sur **Repos** → le cortex moteur (zone centrale) devient rouge.
2. Je clique sur **Mouvement imaginé** → la zone devient bleue.
   « Le sujet n'a pas bougé, il a juste *imaginé* — et le système le détecte. »
3. Je pointe l'**animation du pipeline** en haut.
   « On voit le trajet : le signal arrive, il est traité, le modèle prédit,
   le résultat s'affiche — en 1 seconde. »
4. Je coche **le flux continu** → les signaux arrivent en boucle.
   « Là c'est du streaming continu, sans que je touche à rien. »
5. Je pointe le **moniteur temps réel** en bas → les prédictions défilent.

---

## ⚙️ 2. Explication technique [5 min]

### Mes choix (ce que j'ai vraiment décidé)

> « J'ai utilisé **Spark Structured Streaming** : un outil qui traite les
>   données au fur et à mesure qu'elles arrivent, pas en bloc à la fin. »

> « Pour faire arriver les données, j'ai choisi la méthode la plus simple :
>   je dépose des petits fichiers dans un dossier, et Spark les détecte tout
>   seul. J'ai préféré ça à une grosse solution comme Kafka — mêmes résultats,
>   mais rien à installer. »

> « Pour deviner l'intention, j'utilise un **modèle entraîné** (un classifieur).
>   Je l'ai fait tourner en dehors de Spark, parce que pour un seul petit signal,
>   Spark est trop lent à démarrer. Mon modèle, lui, répond en quelques
>   millisecondes. »

### Mes métriques (affichées en live sur le dashboard)

> Pendant la démo, je pointe le **panneau de métriques** en haut, qui se met
> à jour à chaque clic :

- ⏱️ **Latence** : ~1 à 3 secondes selon la charge — le temps entre le signal
  qui arrive et le résultat affiché.
- 🚀 **Débit** : autour de 0.5 à 1 signal par seconde. Et comme chaque signal
  fait 2 secondes de cerveau, je traite **aussi vite que ça arrive** :
  c'est du **temps réel**.
- 📦 **Taille d'un paquet** : ~110 Ko — un paquet = 2 secondes de signal sur 64 capteurs.
- 🔢 **Signaux traités** : compteur en direct.

### Mes limites (ce que j'assume)

> « J'ai bridé volontairement le débit à 1 signal par seconde, pour que la démo
>   soit claire. En vrai on pourrait aller plus vite. »

> « Sur un sujet que le modèle connaît, j'ai ~100 % de réussite — mais c'est
>   trompeur, ce sujet a servi à l'entraîner. Sur un sujet **inconnu**, je tombe
>   à ~50 %. Et c'est normal : chaque cerveau est différent, c'est la grande
>   difficulté de ce domaine. »

> « J'ai aussi essayé d'ajouter une fenêtre glissante (agréger les résultats sur
>   le temps), mais ça ne marchait pas bien, donc je l'ai retirée pour garder
>   quelque chose de stable. »

---

## 💬 3. Mon appréciation des technologies [5 min]

> Mon projet avait déjà une base (sur la branche `develop`) : un pipeline qui
> entraînait un modèle EEG **en mode batch** (tout d'un coup). Ma contribution
> a été d'**ajouter le streaming** par-dessus.

> « **Spark** : il y avait déjà un traitement batch dans le projet. J'ai aimé
>   pouvoir réutiliser la même logique pour faire du temps réel — pas besoin
>   d'apprendre deux outils. Ce qui m'a un peu freiné : Spark est lourd, il met
>   du temps à démarrer. Je le réutiliserais quand même, surtout pour de gros
>   volumes. »

> « **Dash (le dashboard)** : j'ai créé une page web interactive entièrement en
>   Python, sans avoir à coder du JavaScript. Très rapide à mettre en place.
>   Je le réutiliserais sans hésiter pour tout prototype. »

> « **Kedro (l'organisation du projet)** : grâce à sa structure, j'ai pu
>   ajouter ma nouvelle partie streaming proprement, à côté de l'existant, sans
>   tout casser. Un peu déroutant au début, mais ça force à être organisé. Je
>   garde. »

> « **MLflow et DVC (le côté MLOps)** : ils étaient déjà dans le projet pour
>   suivre les expériences et versionner les données. Utile, même si DVC est un
>   peu lourd pour un projet tout seul. »

> « Pendant le projet, j'ai aussi **enlevé des choses** : j'avais d'abord essayé
>   un autre dataset puis une fenêtre glissante, mais comme ça compliquait sans
>   apporter, j'ai préféré simplifier. »

> « Globalement, la stack Spark + Kedro + Dash, je la reprendrais avec plaisir. »

---

## 🧠 4. Ce que j'ai appris (non technique) [5 min]

> « **1. Travailler brique par brique.** À chaque fois que j'ai voulu tout faire
>   d'un coup, je me suis planté. J'ai appris à avancer petit à petit, en testant
>   chaque morceau avant de passer au suivant. Ça m'a fait gagner du temps. »

> « **2. Privilégier ce qui fonctionne.** J'ai appris à garder ce qui marche
>   plutôt que de tout casser pour une idée ambitieuse. Quand ma fenêtre
>   glissante posait problème, je l'ai retirée pour rester avec une démo stable.
>   Mieux vaut quelque chose de simple qui marche que de compliqué qui plante. »

> « **3. Me méfier des résultats trop beaux.** À un moment mon modèle avait
>   100 % de réussite, j'étais content — puis j'ai compris que je testais sur
>   des données qu'il avait déjà vues. J'ai appris à remettre en question mes
>   propres résultats avant de les accepter. »

> **(Je choisis ces 3-là et je raconte avec mon vécu.)**

---

## ⏱️ Timing

| Bloc | Durée |
|------|-------|
| 1. Démo | 5 min |
| 2. Technique | 5 min |
| 3. Appréciation | 5 min |
| 4. Apprentissages | 5 min |

## Mes métriques à pointer du doigt

Le **panneau métriques en haut du dashboard** affiche tout en direct :
**Latence · Débit · Taille paquet · Signaux traités**.
Je n'ai pas à les apprendre par cœur — il suffit de les lire à l'écran.
