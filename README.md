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

## Injection

- **Drop injection is volume-restricted.** On low-volume series, mean - k*sigma
  collapses below zero (82% of series saturate at k=3), degrading the drop into a
  binary zeroing. Drops are therefore injected only where mean - k*sigma > 0
  (621/3000 series at k=3, fewer at higher k). Consequence: drop recall is
  measured on high-volume, low-relative-variance series and is optimistic
  vs the full dense pool — a documented scope limit.
- **Injected anomaly types exercise distinct features (validated).** Spikes
  surface in the local z-score (mean recomputed z ≈ 3 for k=3). Contextual
  anomalies stay hidden from it (mean |z| = 1.47, below any threshold) but
  explode in the YoY ratio (mean |yoy_ratio - 1| = 1.09). Confirms each feature
  earns its place: a z-score-only detector would miss contextual anomalies
  entirely.
- **Level-shift injection reproduces rolling-detector absorption (validated).**
  Mean recomputed z-score across the 8-week plateau decays monotonically:
  3.01 → 2.17 → 1.76 → 1.45 → 1.23 → 1.07 → 0.89 → 0.79. A local-deviation
  detector flags the shift onset but goes blind to its persistence as the rolling
  mean absorbs the new level (~13-week window). This is the mechanistic
  justification for the sequence-based LSTM autoencoder, which reconstructs
  waveform shape and does not absorb sustained regime change.


## Isolation Forest: results

### Evaluation protocol

The model is unsupervised, so performance is measured against **synthetically
injected anomalies** (see `02_injection`): the model trains on all dense series
(labels never enter training) and is scored on a labelled eval set of four
anomaly types across severities k ∈ {2, 2.5, 3, 3.5, 4}, plus clearly-normal
negatives.

**Metric choice.** We report **recall at a fixed false-positive budget**
(flagging the top 5% / top 1% most anomalous weeks), not PR-AUC as the headline.
On this problem the positive:negative ratio is ~1:150, where PR-AUC is
mechanically low and hard to read (a strong detector still scores ~0.3). Recall
at a fixed FP rate speaks the operational language — *"within the alert budget a
planner can triage, what share of true anomalies do we catch?"* — and is the
metric a demand planner actually cares about. PR-AUC is kept as a secondary
diagnostic.

**Negatives are restricted to clearly-normal weeks (|z| < 2).** M5 is a real
retail dataset containing unlabeled natural anomalies (2.7% of weeks have
|z| ≥ 3). Without this filter, the model is penalised for correctly flagging
real-but-unlabeled anomalies, collapsing every metric. The trade-off, stated
honestly: we measure separation from *clearly normal* weeks, not from natural
anomalies.

### Detection profile (recall at fixed FP budget)

| Anomaly type | Severity | recall @ 5% FP | recall @ 1% FP |
|---|---|---|---|
| **spike** | k=2 | 26% | 10% |
| **spike** | k=3 | 54% | 23% |
| **spike** | k=4 | **81%** | 44% |
| **drop** | k=2 | 5% | 0.2% |
| **drop** | k=3 | 13% | 0.3% |
| **drop** | k=4 | 19% | 0.9% |
| **contextual** | — | 13% | 5% |
| **level_shift** | k=2 | 13% | 4% |
| **level_shift** | k=3 | 19% | 7% |
| **level_shift** | k=4 | 26% | 10% |

### Reading the profile

- **Spikes — strong, and scale cleanly with severity** (26% → 54% → 81%). The
  core detector works: at k=4, four of five spikes are caught within a 5% alert
  budget. This is the reference signal, carried directly by the rolling z-score.

- **Level shifts — the result that motivates the autoencoder.** Even at k=4,
  recall is only 26%, far below spikes at equal severity. The cause is measured,
  not assumed: a rolling detector **absorbs** a sustained shift. The mean
  recomputed z-score across the 8-week plateau decays 3.01 → 2.17 → 1.76 → 1.45
  → 1.23 → 1.07 → 0.89 → 0.79 — the IF flags the onset and goes blind to the
  persistence, so it catches only the first 2–3 plateau weeks. A sequence model
  (LSTM autoencoder, Phase 6) that reconstructs waveform shape should cover this
  blind spot. **This is the quantified case for a second model.**

- **Contextual — weak (13%).** Contextual anomalies are plausible in absolute
  terms and surface only in the YoY feature; a tree-based IF struggles to isolate
  a point abnormal on a single dimension while normal on the other five. A
  structural limit of the IF, and a second target for the autoencoder.

- **Drops — weak and asymmetric (19% at k=4).** The detector is markedly better
  at upward shocks than downward ones: many weeks have naturally low sales, so a
  drop is harder to isolate than a spike. Drops are also volume-restricted by
  construction (on low-volume series, mean − k·σ saturates at zero). A relevant
  supply-chain caveat — missing a demand collapse can be as costly as missing a
  spike.

### Summary

The Isolation Forest is a strong detector of **upward point shocks**, mediocre on
**sustained regime changes** and **seasonal/contextual** anomalies, and weak on
**downward shocks**. This is a clear, defensible profile that defines the brief
for the LSTM autoencoder: cover the IF's blind spots on sequence-shaped and
context-dependent anomalies.

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
