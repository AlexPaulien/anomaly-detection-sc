# Anomaly Detection — Supply Chain

Unsupervised demand anomaly detection on the public **M5 Walmart** dataset.
Spikes, drops, trend breaks and seasonal inconsistencies are surfaced at the
SKU × Store × week grain so a planner can act before they cascade into a
stockout or overstock.

> ⚠️ **Data:** public M5 dataset only. No company data is used or versioned in
> this repository.

**Live API (Cloud Run):** `https://anomaly-detection-sc-117719476260.europe-west1.run.app`
— `/health`, `POST /score`, interactive docs at `/docs`. See
[Serving & deployment](#serving--deployment).

---

## Problem

Demand planners cannot eyeball tens of thousands of SKU × store series every
week. The goal is to automatically flag the weeks whose demand behaviour is
abnormal — and to do it with an honest, measurable notion of "abnormal" — so
review effort goes where it matters.

The twist: anomaly detection is **unsupervised** (no labels), and M5 is a real,
noisy retail dataset. Most of the engineering below is about making detection
*measurable* and *honest* under those constraints.

## Approach

| Step | Choice |
|---|---|
| Signal | Raw sales, fed to the model as **deviations** (never absolute levels) |
| Grid | SKU × Store × week |
| Models | Isolation Forest (shipped) → LSTM Autoencoder → consensus (roadmap) |
| Evaluation | **Synthetic injection** of labeled anomalies |
| Primary metric | **Recall at a fixed false-positive budget** (PR-AUC secondary) |

## Pipeline

```
src/
  data/         daily -> weekly aggregation (PySpark)
  features/     contextual feature engineering
  evaluation/   synthetic injection + metrics
  models/       Isolation Forest (+ LSTM AE in V2)
  serving/      FastAPI app

notebooks/
  01_eda.ipynb              EDA + density segmentation
  02_injection.ipynb        synthetic anomaly injection
  03_isolation_forest.ipynb training, evaluation, threshold, MLflow
```

Daily M5 sales (wide, ~59M rows after melt) → weekly grid via PySpark →
contextual features → Isolation Forest → synthetic-injection evaluation →
MLflow tracking → standalone bundle → FastAPI on Cloud Run.

---

## Key EDA findings

Every feature decision is backed by a number, not intuition. Full analysis in
`notebooks/01_eda.ipynb`.

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
  feature alongside `cat_code`, letting the models learn the interaction rather
  than hand-crafting it.
- **No event feature survives scrutiny.** Super Bowl lift is +0.5% (noise). NBA
  Finals show +5.2% raw but only **+3.3% once de-confounded from June
  seasonality** (compared to adjacent June weeks, not the global baseline) —
  below any anomaly-detection threshold. No M5 event shifts weekly food demand
  enough to warrant a feature; the YoY signal already captures the expected
  seasonal level. A fully data-justified rejection.

## Feature engineering

**Golden rule: no absolute levels reach the model.** Raw sales are used only to
*compute* features; the model sees deviations, so a high-volume store is not
flagged just for being large.

Final feature set (six columns):

| Feature | Meaning |
|---|---|
| `sales_zscore` | Rolling 13-week z-score, **shifted** (prior weeks only) so an anomaly can't mask itself by inflating its own window stats |
| `yoy_ratio` | Sales vs the same week one year earlier, per series (robust to fiscal-calendar / trend artifacts) |
| `wow_delta` | Week-over-week relative change |
| `snap_days` | SNAP days in the week (0–7) |
| `cat_code` | Encoded product category |
| `month` | Fiscal month (coarse residual seasonality) |

**Scope note:** the dense pool is fully continuous — series have variable lengths; validation and demo set showcase series with at least 52 weeks of historical date
so rolling and YoY windows are calendar-exact by row position. Evaluation
therefore covers mature, established series — not launch-phase products.

## Evaluation: synthetic injection

Because the task is unsupervised, performance is measured against anomalies we
inject ourselves (type and position known), then score and compare — the
analogue of an honest holdout. The model trains on all dense series; **labels
never enter training**, they exist only to measure. Four types across severities
k ∈ {2, 2.5, 3, 3.5, 4}. Full walkthrough in `notebooks/02_injection.ipynb`.

Three validations confirm the injection mechanism behaves as intended:

- **Types exercise distinct features.** Spikes surface in the local z-score
  (mean recomputed z ≈ 3 at k=3). Contextual anomalies stay hidden from it
  (mean |z| = 1.47, below any threshold) but explode in the YoY ratio
  (mean |yoy_ratio − 1| = 1.09). A z-score-only detector would miss contextual
  anomalies entirely — each feature earns its place.
- **Level shifts reproduce rolling-detector absorption.** Across the 8-week
  plateau, the mean recomputed z-score decays monotonically:
  3.01 → 2.17 → 1.76 → 1.45 → 1.23 → 1.07 → 0.89 → 0.79. A local-deviation
  detector flags the shift onset but goes blind to its persistence as the
  rolling mean absorbs the new level (~13-week window). This is the mechanistic
  case for a sequence model (Phase 6).
- **Drop injection is volume-restricted.** On low-volume series, mean − k·σ
  collapses below zero (82% saturate at k=3), degrading a drop into binary
  zeroing. Drops are injected only where mean − k·σ > 0 (621/3000 series at k=3,
  fewer at higher k). Consequence: drop recall is measured on high-volume,
  low-variance series and is optimistic vs the full pool — a documented limit.

**Metric choice — recall at a fixed false-positive budget**, not PR-AUC as the
headline. At ~1:150 positive:negative imbalance, PR-AUC is mechanically low and
hard to read (a strong detector still scores ~0.3). Recall at a fixed FP rate
speaks the operational language — *"within the alert budget a planner can
triage, what share of true anomalies do we catch?"*. PR-AUC is kept as a
secondary diagnostic.

**Negatives are restricted to clearly-normal weeks (|z| < 2).** M5 contains
unlabeled natural anomalies (2.7% of weeks have |z| ≥ 3); without this filter the
model is penalised for correctly flagging them, collapsing every metric. The
trade-off, stated honestly: we measure separation from *clearly normal* weeks,
not from natural anomalies.

---

## Results: Isolation Forest

Detection profile — recall at a fixed FP budget:

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

- **Spikes — strong, scaling cleanly with severity** (26% → 54% → 81%). At k=4,
  four of five spikes are caught within a 5% alert budget. The reference signal,
  carried directly by the rolling z-score.
- **Level shifts — the case for the autoencoder.** At a tenable 5% budget the IF
  catches only 20%; recall recovers (44% at a 20% budget) but only at impractical
  alert volumes (~775 alerts/week already at 5% on the dense pool). The cause is
  the measured absorption above — not detection the IF can't reach at any budget,
  but detection it can't reach *efficiently*. The AE targets better recall at
  constant budget.
- **Contextual — the IF's genuine weak point** (13%, flattest sensitivity curve).
  Contextual anomalies surface only in the YoY feature; a tree-based IF struggles
  to isolate a point abnormal on a single dimension. The second AE target.
- **Drops — weak and asymmetric** (19% at k=4). Better at upward shocks than
  downward ones: many weeks are naturally low-sales, so a drop is harder to
  isolate. A relevant supply-chain caveat — missing a demand collapse can be as
  costly as missing a spike.

**Operating threshold** is set from a 5% alert budget (≈775 alerts/week on the
dense pool), not an invented cost. A sensitivity curve (recall vs budget, per
type) exposes the trade-off rather than hiding it behind a single number.
Training, evaluation and threshold selection are tracked in MLflow with
recall-vs-severity logged by step. See `notebooks/03_isolation_forest.ipynb`.

**Summary:** the Isolation Forest is a strong detector of **upward point
shocks**, mediocre on **sustained regime changes**, weak on **seasonal /
contextual** and **downward** anomalies. A clear, defensible profile that defines
the brief for the LSTM autoencoder: cover the sequence-shaped and
context-dependent blind spots.

## Serving & deployment

The model is served as a FastAPI app on **Cloud Run** (europe-west1).

```bash
curl -X POST <url>/score -H "Content-Type: application/json" \
  -d '{"sales_zscore": 6.0, "yoy_ratio": 4.0, "wow_delta": 3.0,
       "snap_days": 2, "cat_code": 0, "month": 6}'
# -> {"anomaly_score": 0.63, "is_anomaly": true, "threshold": 0.55, ...}
```

- **Batch features / online scoring split.** The API scores a feature vector; it
  does not recompute features. A rolling z-score needs 13 weeks of history and
  the YoY ratio needs 52 — infeasible from a single request. Feature engineering
  stays in the batch pipeline (Spark); the API is the online scoring layer.
- **Standalone bundle, no MLflow at runtime.** The service loads `model.pkl` +
  `clip_bounds.json` + `threshold.json` from `models/isolation_forest/` — the
  three artifacts that must travel together to reproduce scoring. Verified:
  deployed scores match local exactly (train/serve parity).
- **Joint-profile decision, not a single-feature threshold.** A point abnormal on
  one feature but normal on the others scores milder than a coherent shock across
  several — the IF evaluates the full six-dimensional profile.
- **Scale-to-zero** keeps idle cost ~nil; the first request after idle has a
  cold-start delay (model load). An assumed trade-off, not a defect.

## Stack

Python · PySpark (local mode) · scikit-learn · MLflow · FastAPI · Docker · GCP Cloud Run

## Status & roadmap

**MVP shipped and deployed** — Isolation Forest, synthetic-injection evaluation,
and a live scoring API on Cloud Run. V2 (sequence model + consensus) extends it;
the MVP stands on its own.

- [x] Phase 1 — EDA + weekly aggregation + density segmentation
- [x] Phase 2 — Contextual feature engineering
- [x] Phase 3 — Synthetic injection + metrics harness
- [x] Phase 4 — Isolation Forest + evaluation + MLflow
- [x] Phase 5 — FastAPI + Cloud Run *(publishable MVP milestone)*
- [ ] Phase 6 — LSTM Autoencoder
- [ ] Phase 7 — Consensus layer
- [ ] Phase 8 — Pub/Sub stream simulation
