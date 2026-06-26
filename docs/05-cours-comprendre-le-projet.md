# Comprendre le projet de A à Z — cours pour la soutenance

> Public : tu sais coder (Python, dev full-stack web) mais tu pars de **zéro en
> Machine Learning**. Ce cours construit toute la logique pas à pas, avec des
> analogies de développeur et des renvois vers **ton propre code et tes résultats**.
> Objectif : pouvoir **expliquer et défendre** chaque choix à l'oral.

---

## Partie 0 — Le projet en 3 phrases (à savoir réciter)

1. On prédit **quels clients vont résilier leur abonnement** (le *churn*) à partir de
   leurs données comportementales.
2. On a comparé **4 modèles**, choisi le meilleur (**Random Forest**), géré le
   **déséquilibre** des données et réglé le **seuil de décision**.
3. On a rendu le tout **utilisable** : une **API** qui prédit, et un **dashboard**
   pour le responsable marketing.

Si tu ne retiens que ça, tu as déjà la colonne vertébrale de la présentation.

---

## Partie 1 — C'est quoi le Machine Learning ?

### L'idée fondamentale
En **développement classique**, *tu écris les règles* :
```python
if paiements_rates >= 2 and satisfaction < 2:
    risque = "élevé"
```
Problème : dans la vraie vie, il y a 30 variables et des milliers de combinaisons. Tu
ne peux pas écrire toutes les règles à la main.

En **Machine Learning**, *tu ne écris pas les règles* : tu donnes à la machine
**des exemples déjà étiquetés** (10 000 clients dont on sait s'ils sont partis ou non),
et **elle déduit les règles toute seule**.

> 🧠 **Analogie dev :** au lieu de coder un filtre anti-spam avec des `if`, tu montres
> 10 000 e-mails marqués « spam / pas spam » et l'algorithme apprend à reconnaître le
> spam. Le « code » des règles est **généré par l'entraînement**, pas écrit par toi.

### Vocabulaire de base (à maîtriser)
| Terme | Ce que c'est | Analogie dev |
|---|---|---|
| **Feature** (variable) | une donnée d'entrée (âge, NPS, paiements ratés…) | un champ d'un formulaire / une colonne |
| **Label** (cible) | la réponse à prédire (`churn` : 0 ou 1) | la valeur attendue d'un test |
| **Modèle** | une fonction `f(features) → prédiction` | une fonction… dont le corps est **appris**, pas écrit |
| **Entraînement** (*fit*) | ajuster le modèle sur les exemples | « compiler » le modèle à partir des données |
| **Prédiction** (*predict*) | appliquer le modèle à un nouveau client | appeler la fonction |

📂 Dans ton code : `model.fit(X_train, y_train)` (entraînement) puis
`model.predict_proba(X)` (prédiction) — fichier `src/churn/train.py`.

---

## Partie 2 — Apprentissage supervisé & notre problème

**Supervisé** = on apprend à partir d'exemples dont **on connaît déjà la réponse**
(les anciens clients dont on sait s'ils ont churné). C'est notre cas.

Deux grandes familles de tâches supervisées :
- **Classification** : prédire une **catégorie** (churn : oui/non). ← **notre projet**
- **Régression** : prédire un **nombre** (ex. le revenu à risque en €).

Notre tâche : **classification binaire** → « ce client va-t-il partir : oui (1) ou non (0) ? »
Le modèle renvoie en réalité une **probabilité** entre 0 et 1 (ex. 0,82 = 82 % de risque).

---

## Partie 3 — Les données (ton dataset)

- **10 000 clients**, **30 variables** (features) + 1 cible (`churn`).
- Variables **numériques** (âge, ancienneté, NPS, nb de paiements ratés…) et
  **catégorielles** (genre, type de contrat, moyen de paiement…).
- La cible `churn` vaut **0** (resté) ou **1** (parti).

### Le point crucial : le déséquilibre
Seulement **10,2 %** des clients ont churné (1 021 sur 10 000). Donc **8,8 clients
fidèles pour 1 churner** (ratio 8,8:1). **Retiens ce chiffre**, c'est le cœur du projet
(voir Partie 7).

📂 Vérifiable dans l'EDA et affiché par `train.py` : *« Taux de churn : 10.2% »*.

---

## Partie 4 — Préparer les données

Un modèle ne « comprend » que des **nombres propres**. Il faut donc transformer les
données brutes. C'est le **preprocessing**.

### 4 opérations
1. **Valeurs manquantes** : certaines cases sont vides (ex. `complaint_type`). On les
   **remplit** (imputation) au lieu de jeter la ligne.
2. **Encodage des catégorielles** : un modèle ne sait pas lire `"Monthly"`. On
   transforme en colonnes 0/1 (**one-hot** : une colonne par catégorie).
   > 🧠 Analogie : comme transformer un `enum` en plusieurs booléens.
3. **Normalisation (scaling)** : mettre toutes les variables numériques à la même
   échelle. Sinon `total_revenue` (des milliers) écrase `csat_score` (1 à 5).
4. **Découpage train / test** (voir ci-dessous).

📂 Fichier `src/churn/preprocessing.py` : un `ColumnTransformer` qui fait tout ça.

### Train / Test : la règle d'or
On coupe les données en deux :
- **Train (80 %)** : pour **entraîner** le modèle.
- **Test (20 %)** : **mis de côté**, jamais vu à l'entraînement, pour **mesurer la
  vraie performance**.

> 🧠 **Analogie dev :** tu ne valides pas ton appli uniquement sur les données avec
> lesquelles tu l'as développée. Tu gardes un jeu « de prod » pour voir si ça marche
> sur du nouveau.

### Le piège n°1 : la fuite de données (*data leakage*)
Si tu calcules la normalisation (la moyenne, etc.) sur **toutes** les données avant de
couper, le modèle « voit » indirectement le test → les scores deviennent **trop beaux
et faux**. La règle : **on apprend les transformations sur le train uniquement**, puis
on les applique au test.

📂 C'est garanti par le **`Pipeline`** : preprocessing + modèle dans un seul objet,
`fit` sur le train seulement. **Argument fort à l'oral.**

---

## Partie 5 — Les modèles (les 4 qu'on a comparés)

On va du plus simple au plus complexe. **Principe pédagogique : commencer simple**
(une *baseline*) pour avoir un point de référence, puis voir si du plus complexe fait
réellement mieux.

### 1. Régression logistique — la *baseline*
Une formule mathématique simple qui combine les variables pour sortir une probabilité.
Rapide, **interprétable**, mais ne capte pas les relations complexes. Sert de référence.

### 2. Arbre de décision → Random Forest ⭐ (notre gagnant)
Un **arbre de décision**, c'est… une cascade de `if/else` **apprise** sur les données :
```
si csat_score < 2  → si paiements_ratés >= 2 → risque élevé
                   → sinon → risque moyen
```
> 🧠 Pour un dev : un arbre = un gros `if/else` imbriqué, mais c'est l'algorithme qui
> trouve les conditions et les seuils.

Un seul arbre se trompe facilement. Le **Random Forest** = **des centaines d'arbres**
entraînés sur des variantes des données, qui **votent**. La moyenne des votes est bien
plus robuste. C'est notre **modèle final**.

### 3. Gradient Boosting
Aussi des arbres, mais construits **en séquence** : chaque nouvel arbre corrige les
erreurs du précédent. Souvent très performant (chez nous : 2ᵉ, juste derrière le RF).

### 4. MLP — le réseau de neurones (Deep Learning)
Des couches de « neurones » (sommes pondérées + transformations non linéaires) qui
apprennent des représentations complexes. Puissant **mais** gourmand en données et
sensible au réglage.

> ⚠️ **Résultat clé de ton projet : le MLP est le PIRE des 4.** C'est volontairement
> mis en avant : sur des **données tabulaires** (des colonnes, comme ici), les
> **arbres battent souvent le Deep Learning**, pour un coût bien moindre. → *« le Deep
> Learning n'est pas toujours supérieur »* (un des objectifs pédagogiques de l'énoncé).

📂 Fichier `src/churn/models.py`.

---

## Partie 6 — Évaluer un modèle (le nerf de la guerre)

### Pourquoi l'accuracy (taux de bonnes réponses) est un PIÈGE ici
90 % des clients restent. Un modèle qui dit **bêtement « personne ne part »** a
**90 % d'accuracy**… et est **totalement inutile** (il rate 100 % des churners !).
Donc on **n'utilise pas l'accuracy** comme métrique principale.

### La matrice de confusion (à savoir dessiner)
On classe les prédictions en 4 cases :

| | Prédit : reste | Prédit : churn |
|---|---|---|
| **Réel : reste** | Vrai Négatif (VN) | Faux Positif (FP) |
| **Réel : churn** | **Faux Négatif (FN)** ⚠️ | Vrai Positif (VP) |

- **FN = churner non détecté** = client perdu = **le plus coûteux** pour nous.
- FP = on alerte sur un client qui serait resté = juste une sollicitation marketing.

### Les bonnes métriques
- **Recall (rappel)** = des churners réels, **combien on en attrape** = VP / (VP+FN).
  → la plus importante ici (on veut rater le moins de churners possible).
- **Precision** = parmi nos alertes, **combien sont justes** = VP / (VP+FP).
- **F1-score** = équilibre entre precision et recall.
- **ROC-AUC** et surtout **PR-AUC** = qualité du **classement** des clients par risque,
  **indépendamment du seuil**. La **PR-AUC** est la plus adaptée quand les positifs
  sont rares (notre cas).

### Cross-validation (validation croisée)
Au lieu d'un seul découpage train/test, on découpe le train en **5 morceaux (plis)** et
on entraîne/teste 5 fois en tournant. On **moyenne** → résultat plus **fiable**, moins
dépendant de la chance du découpage. (« stratifiée » = chaque pli garde les 10 % de
churners.)

### Overfitting / underfitting (à connaître)
- **Overfitting (sur-apprentissage)** : le modèle **mémorise** le train (même le bruit)
  et généralise mal. > 🧠 Analogie : une fonction qui ne marche que sur tes données de
  seed/test, codées en dur.
- **Underfitting** : modèle trop simple, il rate les patterns.
- On cherche **l'équilibre** (compromis **biais / variance**).

📂 Métriques calculées dans `src/churn/evaluate.py`, comparées dans `train.py`.

---

## Partie 7 — Le déséquilibre & le seuil ⭐ (le sommet de ta soutenance)

C'est **l'histoire la plus forte** du projet. Maîtrise-la parfaitement.

### Le problème
Avec 10 % de churners, le modèle « voit » surtout des clients qui restent. Livré tel
quel, il a tendance à **sous-détecter** les churners.

### Les 2 leviers qu'on a utilisés

**A. Au niveau des données / de l'apprentissage**
- **SMOTE** : crée des **exemples synthétiques** de churners pour rééquilibrer le train.
- **class_weight = "balanced"** : on dit au modèle « une erreur sur un churner coûte
  plus cher » → il y fait plus attention.
- *(On a comparé plusieurs stratégies ; elles se valent ici — honnêteté méthodo.)*

**B. Le seuil de décision (LE point qui claque à l'oral)**
Le modèle sort une **probabilité** (ex. 0,30). Pour décider « churn ou pas », on
applique un **seuil**. Par défaut c'est **0,5** — mais **rien n'oblige** à 0,5 !

Chez nous, **au seuil 0,5 → recall = 0** : le modèle ne déclenche **aucune** alerte
(les probabilités des churners dépassent rarement 0,5) → inutile malgré ~90 % d'accuracy.
En **abaissant le seuil à 0,19**, on détecte **78 % des churners** (recall 0,78), au
prix d'une précision plus faible (24 %).

> 💬 **Phrase à dire :** *« Régler le seuil, c'est une décision business : un faux
> négatif nous coûte un client perdu, un faux positif juste un e-mail. On accepte donc
> plus de fausses alertes pour ne rater presque aucun churner. »*

📂 `find_best_threshold()` dans `evaluate.py` ; seuil stocké dans `models/metadata.json`.

---

## Partie 8 — Interpréter le modèle (l'expliquer)

Un modèle qui prédit sans qu'on sache **pourquoi** a peu de valeur en entreprise. On a
donc analysé **quelles variables comptent**.

- **Feature importance / Permutation importance** : vision **globale** — quelles
  variables pèsent le plus *en général*.
- **SHAP** : vision **locale** — pourquoi **ce client précis** est à risque.

**Résultats chez toi (à citer) :** les facteurs dominants sont
**la satisfaction (CSAT)**, puis **les échecs de paiement**, puis **l'ancienneté** et
**l'engagement**. Cohérent avec le bon sens métier → le modèle est crédible.

📂 `src/churn/explain.py` ; figures dans `reports/figures/`.

---

## Partie 9 — L'architecture logicielle (ta zone de confort 💪)

Là, tu parles **dev** — appuie-toi dessus, c'est un point fort.

```
  Dashboard (Streamlit)  ──HTTP/JSON──►  API REST (FastAPI)  ──►  Modèle (.joblib)
   - saisie d'un client                   POST /predict            pipeline entraîné
   - jauge de risque                      GET  /health             + seuil de décision
   - simulation                           validation Pydantic
```

- **Le modèle entraîné est sérialisé** (sauvegardé sur disque avec `joblib`) avec son
  pipeline de preprocessing → on garantit que la transformation à la prédiction est
  **identique** à celle de l'entraînement.
  > 🧠 Analogie : comme un **build artifact** que tu déploies.
- **L'API (FastAPI)** = service d'inférence. `POST /predict` reçoit un client en JSON,
  renvoie la probabilité + la classe. `GET /health` = healthcheck. Validation des
  entrées avec **Pydantic** (comme un DTO/validation de body).
- **Le dashboard (Streamlit)** = la couche de présentation, qui **appelle l'API**
  (architecture **Front → API → Modèle**, réaliste).
- **Reproductibilité** : `Makefile`, `requirements.txt`, tests `pytest`, dépôt Git.

📂 `api/` (main, schemas, service) et `dashboard/app.py`.

---

## Partie 10 — Glossaire express (révision flash)

- **Feature** : variable d'entrée. **Label/cible** : ce qu'on prédit (`churn`).
- **Classification** : prédire une catégorie. **Régression** : prédire un nombre.
- **Train/Test** : entraînement / évaluation sur données jamais vues.
- **Data leakage** : fuite du test vers le train → scores faussés. Évité par le Pipeline.
- **Overfitting** : mémorise au lieu de généraliser.
- **Cross-validation** : moyenne sur plusieurs découpages → fiabilité.
- **Déséquilibre** : classes très inégales (10 % churn).
- **SMOTE / class_weight** : techniques pour gérer le déséquilibre.
- **Seuil** : probabilité-limite pour décider (0,19 chez nous).
- **Matrice de confusion** : VP/FP/FN/VN.
- **Recall** : % de churners attrapés. **Precision** : % d'alertes justes. **F1** : équilibre.
- **PR-AUC / ROC-AUC** : qualité du classement (indépendant du seuil).
- **Random Forest** : forêt d'arbres qui votent. **MLP** : réseau de neurones.
- **Feature importance / SHAP** : expliquer les décisions du modèle.

---

## Partie 11 — Préparation à l'oral

### Déroulé conseillé (≈ 6-8 min), calé sur tes 11 slides
1. **Titre** → te présenter, annoncer le sujet (rétention / churn).
2. **Contexte** → le churn coûte cher, on veut le prédire pour agir.
3. **Données** → 10 000 clients, 30 variables, **et surtout : 10 % de churn → déséquilibre**.
4. **Méthodo** → pipeline anti-fuite, cross-validation, 4 modèles, sélection sur PR-AUC.
5. **Comparaison** → Random Forest gagne ; **le MLP perd → le DL n'est pas toujours mieux**.
6. **Seuil** → *ton moment fort* : 0,5 → recall 0 ; 0,19 → recall 78 %. Décision business.
7. **Interprétabilité** → CSAT et paiements ratés = facteurs n°1 (cohérent métier).
8. **Démo** → montrer le dashboard en live (jauge, simulation).
9. **Architecture** → Front → API → Modèle (ton point fort de dev).
10. **Limites** → données synthétiques, précision 24 %, dérive temporelle.
11. **Conclusion** → 2 enseignements : RF > DL sur tabulaire ; le seuil compte autant que le modèle.

### Questions probables du jury (+ réponses)
- **« Pourquoi pas l'accuracy ? »** → Déséquilibre : 90 % d'accuracy en ne détectant
  aucun churner. On vise Recall / F1 / PR-AUC.
- **« Pourquoi le Random Forest et pas le Deep Learning ? »** → Sur données tabulaires
  les arbres captent mieux les interactions, plus robustes, moins coûteux ; nos
  résultats le montrent (MLP dernier).
- **« C'est quoi le data leakage et comment vous l'évitez ? »** → Fuite du test vers le
  train ; évité en apprenant le preprocessing sur le train seul, via un Pipeline.
- **« Pourquoi un seuil à 0,19 ? »** → Optimisé sur le F1 en validation ; choix business
  (rater un churner coûte plus qu'une fausse alerte).
- **« Votre modèle est-il bon ? »** → Honnêteté : PR-AUC ~0,27 (vs 0,10 au hasard),
  ROC-AUC 0,79 — signal réel mais modéré (données synthétiques) ; la **démarche** est
  rigoureuse et défendable.
- **« Comment l'améliorer ? »** → Feature engineering, suivi de dérive, plus de données.

### Règle d'or à l'oral
Si tu ne sais pas : **ne bluffe pas**. Dis *« le modèle l'apprend à partir des données,
je peux montrer où c'est implémenté »*. Le jury évalue ta **compréhension de la
démarche**, pas ta capacité à réciter des formules.

---

## Pour t'entraîner
1. Lis ce cours **2 fois**.
2. Relis le **rapport** (`docs/03-rapport.pdf`) — il dit la même chose en condensé.
3. **Lance la démo** (`make api` + `make dashboard`) et joue avec la jauge.
4. Demande à quelqu'un (ou à moi) de te poser les questions de la Partie 11.
```
