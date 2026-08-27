# Anomaly Detection — Supply Chain

Détection non-supervisée d'anomalies de demande sur données publiques **M5 Walmart**.
Pendant public/synthétique d'un projet de détection d'anomalies S&OP.

> ⚠️ **Données** : dataset public M5 uniquement. Aucune donnée d'entreprise n'est
> utilisée ni versionnée dans ce dépôt.

## Problème

Repérer automatiquement les signaux de demande anormaux (chocs de ventes,
ruptures de tendance, incohérences) au grain **SKU × Store × semaine**, pour
prioriser l'attention du planner avant que l'anomalie ne cascade en rupture ou
surstock.

## Approche

| Étape | Choix |
|---|---|
| Signal | Ventes brutes, données au modèle sous forme d'**écarts** (jamais de niveau absolu) |
| Grille | SKU × Store × semaine |
| Modèles | Isolation Forest (tabulaire) → LSTM Autoencoder (séquence) → consensus |
| Évaluation | **Injection synthétique** d'anomalies labellisées |
| Métrique | **PR-AUC** (déséquilibre extrême) + courbe recall vs sévérité |

## Points méthodologiques

- **La saisonnalité est portée par le feature engineering** : z-scores glissants,
  déviations saisonnières, flags calendaires (holiday / SNAP / event) — le modèle
  apprend à distinguer « haut *parce que* décembre » de « haut anormalement ».
- **Intermittence M5 traitée explicitement** : l'évaluation ne porte que sur les
  séries denses (z-score/σ mal définis sur les séries clairsemées).
- **Seuil = arbitrage coût métier** (rupture ratée ≠ fausse alerte), pas un
  hyperparamètre au doigt mouillé.

## Stack

Python · PySpark (local mode) · scikit-learn · MLflow · FastAPI · Docker · GCP Cloud Run

## Structure

```
src/
  data/         agrégation quotidien -> hebdo (PySpark)
  features/     feature engineering contextuel
  evaluation/   injection synthétique + métriques
  models/       Isolation Forest (+ LSTM AE en V2)
  serving/      API FastAPI
```

## Statut

🚧 En construction — MVP (Isolation Forest + injection + API) en cours.

## Roadmap

- [ ] Phase 1 — EDA + agrégation hebdo + segmentation densité
- [ ] Phase 2 — Feature engineering contextuel
- [ ] Phase 3 — Injection synthétique + harnais de métriques
- [ ] Phase 4 — Isolation Forest + évaluation
- [ ] Phase 5 — API FastAPI + Cloud Run *(jalon MVP publiable)*
- [ ] Phase 6 — LSTM Autoencoder
- [ ] Phase 7 — Couche de consensus
- [ ] Phase 8 — Simulation de flux Pub/Sub
