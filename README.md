# Anomaly Detection — Supply Chain

Unsupervised demand anomaly detection on the public **M5 Walmart** dataset.


> ⚠️ **Data**: public M5 dataset only. No company data is used or versioned in
> this repository.

## Problem

Automatically surface abnormal demand signals (sales shocks, trend breaks,
inconsistencies) at the **SKU × Store × week** grain, to prioritize planner
attention before an anomaly cascades into a stockout or overstock.

## Approach

| Step | Choice |
|---|---|
| Signal | Raw sales, fed to the model as **deviations** (never absolute levels) |
| Grid | SKU × Store × week |
| Models | Isolation Forest (tabular) → LSTM Autoencoder (sequence) → consensus |
| Evaluation | **Synthetic injection** of labeled anomalies |
| Metric | **PR-AUC** (extreme imbalance) + recall-vs-severity curve |

## Methodological notes

- **Seasonality is carried by feature engineering**: rolling z-scores, seasonal
  deviations, calendar flags (holiday / SNAP / event) — the model learns to tell
  "high *because* December" from "abnormally high".
- **M5 intermittency handled explicitly**: evaluation only covers dense series
  (z-score/σ are ill-defined on sparse series).
- **Threshold = business-cost trade-off** (missed stockout ≠ false alarm), not a
  hand-waved hyperparameter.

## Stack

Python · PySpark (local mode) · scikit-learn · MLflow · FastAPI · Docker · GCP Cloud Run

## Structure

```
src/
  data/         daily -> weekly aggregation (PySpark)
  features/     contextual feature engineering
  evaluation/   synthetic injection + metrics
  models/       Isolation Forest (+ LSTM AE in V2)
  serving/      FastAPI app
```

## Status

🚧 Work in progress — MVP (Isolation Forest + injection + API) underway.

## Roadmap

- [ ] Phase 1 — EDA + weekly aggregation + density segmentation
- [ ] Phase 2 — Contextual feature engineering
- [ ] Phase 3 — Synthetic injection + metrics harness
- [ ] Phase 4 — Isolation Forest + evaluation
- [ ] Phase 5 — FastAPI + Cloud Run *(publishable MVP milestone)*
- [ ] Phase 6 — LSTM Autoencoder
- [ ] Phase 7 — Consensus layer
- [ ] Phase 8 — Pub/Sub stream simulation
