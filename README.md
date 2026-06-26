# Projet M1 — Système Intelligent de Rétention Client 

## Objectif

Construire un système d'aide à la décision qui prédit la probabilité qu'un client
résilie son abonnement (*churn*), explique ce risque, et le met à disposition d'un
utilisateur métier via un dashboard interactif et une API REST.

## Périmètre retenu

| Décision | Choix |
|---|---|
| Tâche de prédiction | **Churn — classification binaire** (cible : `churn`) |
| API REST | **Incluse** (le dashboard appelle l'API) |
| Deep Learning | **Inclus** — un MLP en complément des modèles classiques (optionnel dans le sujet) |

## Dataset

- Source : [Kaggle — customer-churn-prediction-business-dataset](https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset)
- Fichier : `customer_churn.csv`
- ~10 000 clients, variables numériques + catégorielles, cible `churn` (0/1)

## Documentation

- [`docs/01-specifications-fonctionnelles.md`](docs/01-specifications-fonctionnelles.md) — le *QUOI*
- [`docs/02-specifications-techniques.md`](docs/02-specifications-techniques.md) — le *COMMENT*
