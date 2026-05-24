---
marp: true
theme: default
paginate: true
header: 'NeuroSpark · EEG BCI Streaming'
footer: 'UE28 Big Data · 2026'
style: |
  section { background: #f5f1e8; color: #2b2a26; font-family: Inter, system-ui, sans-serif; font-size: 26px; }
  h1, h2 { font-family: 'Fraunces', Georgia, serif; color: #2b2a26; }
  h1 { color: #c44536; font-size: 46px; }
  code { background: #f0eadf; padding: 2px 8px; border-radius: 5px; }
  strong { color: #c44536; }
  .big { font-size: 32px; line-height: 1.7; }
  .corail { color: #c44536; }
  .sauge { color: #7a9b76; }
  .ocean { color: #5b8baf; }
---

<!-- _class: lead -->

# NeuroSpark
## Lire l'intention motrice en temps réel

Un pipeline streaming qui décode les signaux du cerveau
au fur et à mesure qu'ils arrivent.

Emir Mahemas — UE28 Big Data

---

## 1 · La question

> Un sujet est devant un ordinateur.
> Il imagine bouger sa main, sans la bouger.
>
> **Est-ce qu'on peut détecter cette intention,
>   en direct, depuis l'extérieur du crâne ?**

→ Oui, et c'est ce que mon système fait, en **~2 secondes**.

---

## 2 · Le dataset

**PhysioNet Motor Imagery** (Schalk et al., 2004)
classique de la BCI (Brain-Computer Interface).

<div class="big">

- 109 sujets, **64 électrodes EEG**
- Signal échantillonné à **160 Hz**
- 3 états mesurés : Repos · Mouvement réel · Mouvement imaginé

</div>

Mon système prédit en direct lequel des trois.

---

## 3 · Architecture (en surface)

<div class="big">

```
🧠 Signal (2 s, 64 capteurs)
        ▼
📁 Fichier déposé dans un dossier
        ▼
👁️ Spark détecte et lit le fichier
        ▼
🤖 Modèle prédit l'état mental
        ▼
📊 Dashboard affiche le résultat

         tout ça en ~1 seconde
```

</div>

---

## 4 · DÉMO LIVE

🎬 Une commande :

```bash
make demo
```

Trois choses à regarder :
1. La **topographie cérébrale** (le « cerveau » sur le dashboard)
2. La **prédiction** + sa confiance
3. Le **panneau métriques** (latence, débit, taille, signaux traités)

→ je vais cliquer / observer / commenter.

---

## 5 · Mes choix techniques

<div class="big">

🔹 **Spark Structured Streaming** → traite les données en flux,
   pas en bloc. Même outil que pour le batch, juste un mode différent.

🔹 **Pas de Kafka** → je dépose des fichiers dans un dossier,
   Spark les détecte. Plus simple, mêmes garanties.

🔹 **Un modèle hors Spark pour la prédiction** → Spark est lourd
   à démarrer pour un seul signal. Mon modèle répond en ~10 ms.

</div>

---

## 6 · Mes métriques (en live sur le dashboard)

<div class="big">

| | |
|---|---|
| ⏱️ **Latence** | ~1 à 3 s (signal → résultat) |
| 🚀 **Débit** | ~1 signal/s = **temps réel** |
| 📦 **Taille paquet** | ~110 Ko (2 s × 64 capteurs) |
| 🔢 **Signaux traités** | compteur live |

</div>

→ pas besoin de mémoriser, le dashboard affiche tout.

---

## 7 · Mes limites (assumées)

<div class="big">

🔸 **Sujet connu** → ~100 % (mais c'est du *data leakage* :
   ce sujet a servi à entraîner le modèle, je le montre exprès).

🔸 **Sujet inconnu** → ~50 % seulement.
   Normal en BCI : chaque cerveau est différent.

🔸 **Modèle simple** (RandomForest) → un réseau de neurones ferait
   mieux, mais demande beaucoup plus de données.

</div>

---

## 8 · Appréciation des technologies

<div class="big">

🟠 **Spark** : puissant mais lourd à démarrer.
   À reprendre pour du gros volume.

🟢 **Dash + Plotly** : super rapide pour un dashboard interactif
   en pur Python. Je le reprends sans hésiter.

🟡 **Kedro** : structure le projet en pipelines clairs.
   Courbe d'apprentissage, mais bonne hygiène.

🔵 **MLflow / DVC** : utiles pour suivre les expériences
   et versionner les données.

</div>

---

## 9 · Ce que j'ai appris (non technique)

<div class="big">

**Travailler brique par brique.** À chaque fois que j'ai voulu tout
faire d'un coup, je me suis planté. Petit + testé > grand + cassé.

**Privilégier ce qui fonctionne.** J'ai essayé d'ajouter une fenêtre
glissante, ça posait problème, je l'ai retirée. Mieux vaut simple et
stable qu'ambitieux et cassé.

**Me méfier des résultats trop beaux.** À un moment j'avais 100 %
partout — c'était en fait un data leakage. J'ai appris à remettre
en question mes propres résultats.

</div>

---

<!-- _class: lead -->

## Punchline

> Pas de mouvement, pas de mot, pas de toucher.
>
> Juste **64 électrodes, Spark, et 2 secondes** —
> et le système devine ce que pense le sujet.

**Merci · Questions ?**
