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
- **No event feature survives scrutiny.** Super Bowl lift is +0.5% (noise). NBA
  Finals show +5.2% raw but only **+3.3% once de-confounded from June
  seasonality** (compared to adjacent June weeks, not the global baseline) —
  below any anomaly-detection threshold. Conclusion: no M5 event shifts weekly
  food demand enough to warrant a feature; the YoY seasonal signal already
  captures the expected seasonal level. A fully data-justified rejection.

## Feature Engineering

- **Dense pool is fully continuous:** all 15,509 series span the same 277 weeks
  with no gaps (density ≥ 0.8 selected products present since d_1). Rolling and
  YoY windows are calendar-exact by row position. Scope note: evaluation covers
  mature, established series — not launch-phase products.

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
