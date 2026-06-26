# Spécifications Fonctionnelles — Système de Prédiction du Churn

> Le *QUOI* : ce que le système fait, pour qui, et comment l'utilisateur s'en sert.
> (Les choix techniques sont traités dans `02-specifications-techniques.md`.)

---

## 1. Vision & objectif

Construire un système d'aide à la décision qui :

1. **prédit la probabilité qu'un client résilie son abonnement** (*churn*),
2. **explique** les raisons de ce risque,
3. met le tout à disposition d'un utilisateur métier via un **dashboard interactif**
   et une **API REST**.

L'objectif métier : permettre à une entreprise par abonnement (SaaS, télécom,
e-commerce) d'**agir sur les clients à risque avant qu'ils ne partent**, et donc de
réduire la perte de revenu liée au churn.

---

## 2. Acteurs (utilisateurs)

| Acteur | Ce qu'il attend du système |
|---|---|
| **Responsable marketing / CRM** | Identifier les clients à risque, comprendre pourquoi, prioriser les actions de rétention |
| **Data Scientist** (auteur du projet) | Analyser les données, comparer les modèles, justifier les choix (via le rapport) |
| **Système tiers** | Appeler l'API pour scorer un client automatiquement (ex : un CRM, une plateforme d'emailing) |

---

## 3. Périmètre

### Inclus
- **Tâche : classification binaire** → prédire `churn` (0 = reste / 1 = part).
  C'est la tâche « naturelle » du dataset et la plus solide.
- **Gestion du déséquilibre des classes** (les churners sont minoritaires).
- **Comparaison d'au moins 3 modèles**, dont un **modèle Deep Learning (MLP)**.
- **Dashboard interactif** orienté décideur métier.
- **API REST** d'inférence ; le dashboard consomme l'API.
- **Interprétabilité** des prédictions (feature importance / SHAP).

### Hors périmètre
- Autres tâches de prédiction (revenu à risque, CLV) — non retenues.
- Déploiement en production réel, ingestion de données temps réel live.
- Authentification / gestion d'utilisateurs.
- Réentraînement automatique du modèle.

---

## 4. Exigences fonctionnelles

### EF1 — Acquisition & préparation des données
Le système :
- charge le dataset `customer_churn.csv` ;
- nettoie les données (valeurs manquantes, incohérences, doublons) ;
- encode les variables catégorielles ;
- normalise / standardise les variables numériques si nécessaire ;
- fournit une **analyse exploratoire (EDA) documentée** (distributions, corrélations,
  taux de churn par segment).

**Garantie :** aucune fuite de données — le preprocessing est appris sur le jeu
d'entraînement uniquement, puis appliqué au jeu de test.

### EF2 — Modélisation multi-algorithmes
Le système :
- entraîne un **modèle baseline simple** (ex. régression logistique) pour référence ;
- entraîne des **modèles avancés** (ex. Random Forest, Gradient Boosting) ;
- entraîne **un modèle Deep Learning (MLP)** ;
- compare l'ensemble de manière structurée (≥ 3 modèles au total) ;
- **gère le déséquilibre des classes** : compare au moins 2 techniques de rééquilibrage
  (ex. SMOTE, over/under-sampling, `class_weight`) et **ajuste le seuil de décision** ;
- **désigne un modèle final justifié** (performance / robustesse / interprétabilité /
  coût / cohérence métier).

### EF3 — Système d'évaluation
Le système produit :
- des **métriques adaptées au déséquilibre** : Recall, F1-score, ROC-AUC, PR-AUC
  (*l'accuracy seule est insuffisante et trompeuse ici*) ;
- des **tableaux et graphiques comparatifs** entre modèles ;
- une **analyse des erreurs** : matrice de confusion, exemples de clients mal prédits,
  hypothèses explicatives ;
- une **validation robuste** (validation croisée stratifiée).

### EF4 — Interprétabilité
Le système permet de :
- identifier **quelles variables influencent le plus** le modèle (importance globale) ;
- expliquer **pourquoi un client donné** est jugé à risque (explication locale,
  ex. SHAP) ;
- produire des explications **cohérentes avec la logique métier** (ex. paiements ratés,
  NPS faible, baisse d'usage → risque accru).

### EF5 — Dashboard interactif *(obligatoire)*
Une application web orientée utilisateur métier permettant de :
- **saisir ou charger un profil client** et obtenir sa **probabilité de churn** ;
- **visualiser l'explication** de la prédiction (variables qui augmentent / réduisent
  le risque) ;
- **explorer les KPI** : taux de churn global, nombre / liste de clients à risque,
  segments les plus exposés ;
- **comparer les performances** des modèles testés ;
- **simuler un scénario** (ex. « si la satisfaction de ce client augmentait ? »).

Le dashboard est **autonome et utilisable par un non-technicien**. Il obtient ses
prédictions **en appelant l'API** (et non en chargeant directement le modèle).

> Note : les visualisations d'analyse scientifique (EDA, comparaison des modèles) du
> rapport sont distinctes de l'interface décisionnelle du dashboard.

### EF6 — API REST *(incluse)*
Le système expose un service d'inférence :
- **`POST /predict`** : reçoit un profil client en JSON → renvoie la **probabilité** et
  la **classe** prédite (churn oui/non) ;
- **`GET /health`** : vérifie que le service est actif et que le modèle est chargé ;
- **gestion d'erreurs** : champs manquants, types incorrects, valeurs hors plage →
  message clair + code HTTP approprié ;
- **documentation minimale** du format d'entrée/sortie (README ou doc auto type Swagger).

---

## 5. Parcours utilisateur type (dashboard)

1. Le responsable marketing ouvre le dashboard.
2. Il consulte le **taux de churn global** et la **liste des clients à risque**.
3. Il sélectionne un client → voit *« risque de départ : 82 %, principalement causé par :
   3 paiements ratés + NPS faible + baisse d'usage récente »*.
4. Il simule une action (ex. amélioration de l'engagement) → observe l'impact sur le risque.
5. Il décide de l'action de rétention à déclencher.

---

## 6. Critères d'acceptation (« c'est réussi si… »)

- [ ] Le modèle **détecte réellement les churners** (Recall / F1 / PR-AUC corrects,
      pas une accuracy trompeuse).
- [ ] Chaque prédiction est **explicable** (globale + locale).
- [ ] Le **déséquilibre des classes** est traité et justifié.
- [ ] Le **dashboard fonctionne** pour un utilisateur non technique et appelle l'API.
- [ ] L'**API** répond correctement et gère les erreurs.
- [ ] Le projet est **reproductible** : code structuré en modules, versionné (Git),
      environnement documenté.

---

## 7. Exigences non fonctionnelles

| Type | Exigence |
|---|---|
| **Reproductibilité** | Scripts d'exécution (CLI / Makefile), `requirements.txt`, artefacts versionnés |
| **Maintenabilité** | Code organisé en modules (data / modeling / evaluation / api / dashboard), pas de notebook monolithique |
| **Robustesse** | API testée indépendamment du dashboard, gestion d'erreurs |
| **Traçabilité** | Logs / suivi des expériences (métriques, figures) |
| **Lisibilité** | Démarche scientifique documentée dans le rapport (choix justifiés) |

---

## 8. Lien avec la grille de notation

| Domaine (barème /20) | Exigences fonctionnelles couvertes |
|---|---|
| Préparation & qualité des données (4) | EF1 |
| Modèles & évaluation (5) | EF2, EF3 |
| Reproductibilité & traçabilité (3) | Non fonctionnelles §7 |
| Exposition POC (3) | EF6 (API), CLI |
| Rapport & visualisations (3) | EF3, EF4 |
| Qualité logicielle (2) | Non fonctionnelles §7 |
