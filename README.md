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

## Key EDA findings

The exploratory analysis drove every feature decision — each is backed by a
number, not intuition. Full analysis in `notebooks/01_eda.ipynb`.

- **Weekly aggregation already tames intermittency.** Zero-sales share is 39.7%
  at the weekly grain (vs the daily grain M5 is known for). After trimming
  cold-start zeros (weeks before a product's first sale — 21% of the raw grid),
  true demand intermittency is a moderate **23.9%**.
- **Density is a continuum, not two populations.** No bimodality, so the
  dense/sparse threshold is set by *evaluation need* (a rolling z-score needs a
  well-defined local variance), not by a natural gap. **Dense pool = density ≥
  0.8 AND life ≥ 52 weeks → 15,509 series** across all three categories.
- **Seasonal deviation is computed per series, year-over-year.** The global
  seasonal profile mixes trend and fiscal-calendar effects (Walmart's fiscal
  year starts in February); a per-series N-1 comparison is robust to this by
  construction — which is what the 52-week life floor guarantees.
- **SNAP matters, but only for food.** FOODS sales rise ~19% with SNAP days;
  HOBBIES and HOUSEHOLD are flat (not SNAP-eligible). `snap_days` is kept as a
  feature with `cat_id`, letting the models learn the interaction rather than
  hand-crafting it.
- **Only sporting events move demand.** Of four event types, only Sporting
  (Super Bowl) shows a lift (+6.4% weekly); Cultural, Religious and National are
  flat or negative. The raw event count is dropped in favour of a single
  `is_sporting_week` flag — a data-justified subtractive choice.

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

- [x] Phase 1 — EDA + weekly aggregation + density segmentation
- [ ] Phase 2 — Contextual feature engineering
- [ ] Phase 3 — Synthetic injection + metrics harness
- [ ] Phase 4 — Isolation Forest + evaluation
- [ ] Phase 5 — FastAPI + Cloud Run *(publishable MVP milestone)*
- [ ] Phase 6 — LSTM Autoencoder
- [ ] Phase 7 — Consensus layer
- [ ] Phase 8 — Pub/Sub stream simulation
