# Spécifications Techniques — Système de Prédiction du Churn

> Le *COMMENT* : architecture, stack, structure du code, contrats d'API, et choix
> d'implémentation. Référence le *QUOI* défini dans `01-specifications-fonctionnelles.md`.

---

## 1. Vue d'ensemble de l'architecture

Architecture en couches, réaliste (Front / API / Modèle) :

```
┌──────────────────┐      HTTP/JSON       ┌──────────────────┐
│   Dashboard      │  ───────────────►    │   API REST       │
│   (Streamlit)    │   POST /predict      │   (FastAPI)      │
│                  │  ◄───────────────    │                  │
│  - saisie client │   proba + classe     │  - validation    │
│  - KPIs / SHAP   │                      │  - preprocessing │
│  - simulation    │                      │  - inférence     │
└──────────────────┘                      └────────┬─────────┘
                                                    │ charge
                                                    ▼
                                         ┌──────────────────────┐
                                         │  Artefacts (joblib)   │
                                         │  - pipeline préproc.  │
                                         │  - modèle final       │
                                         │  - seuil de décision  │
                                         └──────────────────────┘
```

Principe clé : **le modèle n'est jamais chargé directement par le dashboard**.
Le dashboard est une couche de présentation qui consomme l'API. Le pipeline de
préprocessing est **sérialisé avec le modèle** pour garantir une transformation
identique entre l'entraînement et l'inférence (et éviter toute fuite de données).

---

## 2. Stack technique

| Brique | Technologie | Justification |
|---|---|---|
| Langage | **Python 3.11+** | Standard data science |
| Manipulation données | **pandas**, **numpy** | — |
| ML classique | **scikit-learn** | Pipelines, modèles, métriques, CV |
| Déséquilibre | **imbalanced-learn** | SMOTE, sampling, `Pipeline` compatible sklearn |
| Deep Learning | **TensorFlow / Keras** | MLP « vrai » réseau de neurones (cf. §7, note) |
| Interprétabilité | **shap**, feature importance native | Explications globale + locale |
| API | **FastAPI** + **uvicorn** | Validation Pydantic, doc Swagger auto |
| Validation schéma | **Pydantic v2** | Contrats d'entrée/sortie typés |
| Dashboard | **Streamlit** | Couche de présentation rapide |
| Visualisation | **plotly**, **matplotlib/seaborn** | Graphiques interactifs (dashboard) + statiques (rapport) |
| Sérialisation | **joblib** | Artefacts modèle + pipeline |
| Suivi d'expériences | **CSV/JSON loggés** (MLflow en option bonus) | Traçabilité métriques |
| Tests | **pytest** | Tests unitaires (preprocessing, API) |
| Env / reproductibilité | **requirements.txt** + **Makefile** | Installation et exécution reproductibles |
| Versioning | **Git / GitHub** | Historique, collaboration |

---

## 3. Arborescence du projet

```
projet-m1-churn/
├── README.md
├── requirements.txt
├── Makefile                      # cibles : setup, eda, train, api, dashboard, test
├── config.yaml                   # chemins, hyperparamètres, seed, seuil
│
├── data/
│   ├── raw/customer_churn.csv    # dataset brut (non versionné si volumineux)
│   └── processed/                # éventuels exports intermédiaires
│
├── docs/
│   ├── 01-specifications-fonctionnelles.md
│   └── 02-specifications-techniques.md
│
├── notebooks/
│   └── 01_eda.ipynb              # EDA et expérimentation (pas le pipeline final)
│
├── src/churn/
│   ├── __init__.py
│   ├── config.py                 # chargement config.yaml, constantes
│   ├── data.py                   # chargement + split train/test stratifié
│   ├── preprocessing.py          # ColumnTransformer (num + cat), pas de fuite
│   ├── balancing.py              # stratégies de rééquilibrage comparées
│   ├── models.py                 # définition des modèles (sklearn + Keras)
│   ├── train.py                  # entraînement, CV, sélection du modèle final
│   ├── evaluate.py               # métriques, matrice de confusion, tuning du seuil
│   ├── explain.py                # SHAP + feature importance
│   └── persistence.py            # save/load des artefacts (joblib)
│
├── api/
│   ├── main.py                   # app FastAPI, endpoints
│   ├── schemas.py                # modèles Pydantic (request/response)
│   └── service.py                # chargement artefacts + logique d'inférence
│
├── dashboard/
│   └── app.py                    # application Streamlit (consomme l'API)
│
├── models/                       # artefacts sérialisés (.joblib, .keras)
│   ├── final_model.joblib
│   ├── preprocessor.joblib
│   └── metadata.json             # seuil, version, métriques, features attendues
│
├── reports/
│   ├── figures/                  # graphiques pour le rapport
│   └── metrics/                  # tableaux comparatifs (CSV)
│
└── tests/
    ├── test_preprocessing.py
    └── test_api.py
```

---

## 4. Données

### 4.1 Schéma des variables (cible : `churn`)

| Variable | Type | Rôle |
|---|---|---|
| `age` | numérique | feature |
| `gender` | catégorielle | feature |
| `tenure` | numérique | feature (ancienneté) |
| `contract_type` | catégorielle | feature |
| `monthly_charges` | numérique | feature |
| `total_revenue` | numérique | feature |
| `payment_failures` | numérique | feature |
| `support_tickets` | numérique | feature |
| `session_duration` | numérique | feature |
| `login_frequency` | numérique | feature |
| `nps_score` | numérique | feature |
| **`churn`** | binaire (0/1) | **cible** |

> Le schéma exact sera confirmé après l'EDA (types réels, cardinalité des catégorielles).

### 4.2 Découpage
- **Split stratifié** train/test (ex. 80/20) sur `churn` pour préserver le ratio de classes.
- **Validation croisée stratifiée** (`StratifiedKFold`, k=5) pour la sélection de modèle.
- `random_state` fixe (depuis `config.yaml`) → reproductibilité.

### 4.3 Pipeline de préprocessing (anti-fuite)
Implémenté via `ColumnTransformer` dans un `Pipeline` sklearn :
- **Numériques** : imputation (médiane) → `StandardScaler`.
- **Catégorielles** : imputation (mode) → `OneHotEncoder(handle_unknown="ignore")`.
- Le pipeline est **`fit` sur le train uniquement**, puis appliqué au test et à l'inférence.
- Le préprocesseur est **sérialisé avec le modèle** (cohérence train/prod).

---

## 5. Gestion du déséquilibre des classes

Étapes (EF2/EF3) :
1. **Diagnostic** : ratio classe majoritaire/minoritaire, matrice de confusion d'un baseline.
2. **Comparaison d'au moins 2 techniques de rééquilibrage**, intégrées dans un
   `imblearn.Pipeline` (appliquées **dans la CV**, sur le train fold uniquement) :
   - SMOTE (sur-échantillonnage synthétique),
   - Random Under-Sampling **ou** Random Over-Sampling.
3. **Approche au niveau du modèle** : `class_weight="balanced"` (cost-sensitive).
4. **Ajustement du seuil de décision** : optimisation sur le train/validation pour
   maximiser le F1 ou le Recall (le seuil 0.5 par défaut n'est pas optimal).
   → Le seuil retenu est stocké dans `metadata.json` et utilisé par l'API.

---

## 6. Modélisation

| Modèle | Rôle | Bibliothèque |
|---|---|---|
| Régression logistique | **baseline** (référence, interprétable) | scikit-learn |
| Random Forest | modèle avancé (non-linéaire) | scikit-learn |
| Gradient Boosting (ou XGBoost) | modèle avancé (souvent meilleur) | scikit-learn / xgboost |
| **MLP (réseau de neurones)** | **Deep Learning** | TensorFlow/Keras |

- **Hyperparamètres** : `GridSearchCV` ou `RandomizedSearchCV` sur des plages réalistes
  et justifiées (pas de tuning « au maximum »).
- **Sélection du modèle final** : sur des critères multiples (PR-AUC / Recall, robustesse
  en CV, interprétabilité, coût, cohérence métier) — pas uniquement le meilleur score.
- Le choix est **argumenté dans le rapport** ; le modèle final est sérialisé.

---

## 7. Modèle Deep Learning (MLP)

- **Architecture** : MLP simple — entrées (features encodées) → 1-2 couches denses
  (ReLU) + dropout → couche sortie sigmoïde (probabilité de churn).
- **Entraînement** : `binary_crossentropy`, optimiseur Adam, early stopping sur la
  validation, gestion du déséquilibre via `class_weight`.
- **Analyse critique attendue** (objectif pédagogique du sujet) : comparer le MLP aux
  modèles classiques et discuter *« le Deep Learning n'est pas toujours supérieur »*
  (biais/variance, overfitting, coût de calcul, gain réel vs complexité).

> **Note d'implémentation :** si l'on veut éviter la dépendance TensorFlow et rester
> 100 % dans le pipeline sklearn, `sklearn.neural_network.MLPClassifier` est une
> alternative valable et plus simple. Choix par défaut retenu : **Keras** (réseau de
> neurones « explicite », plus défendable comme composante Deep Learning).

---

## 8. Évaluation

- **Métriques** (adaptées au déséquilibre) : Recall, Precision, F1-score, ROC-AUC,
  **PR-AUC** (prioritaire ici), + accuracy à titre indicatif.
- **Artefacts produits** : tableau comparatif des modèles (`reports/metrics/*.csv`),
  courbes ROC et Precision-Recall, matrices de confusion.
- **Analyse des erreurs** : exemples de faux négatifs (churners ratés — coûteux),
  hypothèses explicatives.
- **Validation** : `StratifiedKFold` (k=5), moyennes ± écarts-types.

---

## 9. Interprétabilité

- **Globale** : importance native (modèles à arbres) + **permutation importance**
  (agnostique) → quelles variables pèsent le plus.
- **Locale** : **SHAP** sur le modèle final → expliquer une prédiction individuelle
  (« ce client est à risque à cause de X, Y, Z »).
- Sorties exploitées par le dashboard (EF5) et le rapport (EF4).

---

## 10. Sérialisation & artefacts

`models/` contient :
- `preprocessor.joblib` — le `ColumnTransformer` fitté ;
- `final_model.joblib` (ou `.keras` pour le MLP) — le modèle retenu ;
- `metadata.json` — version, seuil de décision, liste ordonnée des features attendues,
  métriques de référence.

L'API charge ces artefacts au démarrage (et non à chaque requête).

---

## 11. Spécification de l'API (FastAPI)

### 11.1 Endpoints

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/health` | Statut du service + modèle chargé |
| `POST` | `/predict` | Prédiction du churn pour un client |
| `GET` | `/model-info` *(option)* | Métadonnées du modèle (version, features, seuil) |
| `GET` | `/docs` | Documentation Swagger auto (FastAPI) |

### 11.2 `POST /predict` — contrat

**Requête** (`application/json`) :
```json
{
  "age": 42,
  "gender": "Female",
  "tenure": 8,
  "contract_type": "Month-to-month",
  "monthly_charges": 79.9,
  "total_revenue": 639.2,
  "payment_failures": 3,
  "support_tickets": 5,
  "session_duration": 12.4,
  "login_frequency": 2,
  "nps_score": 3
}
```

**Réponse** `200 OK` :
```json
{
  "churn_probability": 0.82,
  "churn_prediction": 1,
  "threshold": 0.41,
  "model_version": "1.0.0"
}
```

### 11.3 Validation & gestion d'erreurs

- Validation via **Pydantic** (types, bornes : ex. `nps_score` 0-10, valeurs ≥ 0).
- Codes HTTP :
  - `200` — succès ;
  - `422` — payload invalide (champ manquant, type incorrect) — géré nativement par FastAPI ;
  - `400` — valeur métier hors plage ;
  - `503` — modèle non chargé (`/health` le reflète).
- Messages d'erreur clairs et structurés (`{"detail": "..."}`).

### 11.4 Logique d'inférence (`api/service.py`)
1. Charger artefacts au démarrage (préprocesseur + modèle + seuil).
2. Convertir le JSON validé en DataFrame (ordre des features depuis `metadata.json`).
3. Appliquer le préprocesseur → prédire la probabilité.
4. Appliquer le **seuil** pour la classe → renvoyer la réponse.

---

## 12. Spécification du dashboard (Streamlit)

Le dashboard **consomme l'API** via `requests` (pas de modèle chargé localement).

### Sections (mapping EF5)
1. **Vue d'ensemble (KPIs)** : taux de churn global, nb de clients à risque,
   segments les plus exposés (graphiques Plotly).
2. **Scoring d'un client** : formulaire de saisie → appel `POST /predict` →
   affichage proba + classe (`st.metric`, jauge).
3. **Explication (SHAP)** : variables qui poussent le risque à la hausse/baisse pour
   le client saisi.
4. **Comparaison des modèles** : tableau/graphes des performances (lecture des
   artefacts `reports/metrics/`).
5. **Simulation de scénario** : modification d'un paramètre (ex. `nps_score`,
   `login_frequency`) → re-appel API → impact sur le risque.

### Points techniques
- Gestion d'état : `st.session_state` ; mise en cache : `st.cache_data` pour les
  données statiques (KPIs, métriques).
- URL de l'API configurable (variable d'environnement / `config.yaml`).
- Gestion des erreurs réseau (API indisponible → message clair).

---

## 13. Reproductibilité & exécution

### Makefile (cibles principales)
```makefile
setup:      # pip install -r requirements.txt
eda:        # exécute / exporte l'EDA
train:      # entraîne, compare, sélectionne, sérialise le modèle final
api:        # uvicorn api.main:app --reload
dashboard:  # streamlit run dashboard/app.py
test:       # pytest
```

### Garanties
- **Seed** fixe (`config.yaml`) sur split, CV, modèles.
- **`requirements.txt`** avec versions épinglées.
- Pipeline final **exécutable hors notebook** (les notebooks ne servent qu'à l'EDA).
- Artefacts et métriques **versionnés/loggés** (traçabilité).

---

## 14. Tests (pytest)

| Test | Objet |
|---|---|
| `test_preprocessing.py` | Le préprocesseur fit/transform sans fuite, gère valeurs manquantes |
| `test_api.py` | `/health` répond, `/predict` renvoie le bon schéma, `422` sur payload invalide |

> Tester l'API **indépendamment du dashboard** (Postman / curl / `TestClient` FastAPI)
> avant l'intégration — point de défaillance classique des projets.

---

## 15. Ordre de mise en œuvre recommandé

1. EDA (`notebooks/01_eda.ipynb`) → comprendre les données, valider le schéma §4.1.
2. Preprocessing + split (`src/churn/`) → pipeline anti-fuite.
3. Baseline + gestion du déséquilibre → métriques de référence.
4. Modèles avancés + MLP → comparaison, tuning, sélection finale.
5. Évaluation + interprétabilité (SHAP) → artefacts et figures.
6. Sérialisation → `models/`.
7. API FastAPI → testée seule (curl/Postman/pytest).
8. Dashboard Streamlit → branché sur l'API.
9. Rapport (6 p.) + support de présentation + dépôt Git propre.
```
