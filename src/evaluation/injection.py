"""Phase 3 — Synthetic anomaly injection (THE brick that carries evaluation).

Principle: models train unsupervised on all the data; we EVALUATE on anomalies
we injected, hence labeled. This is what yields precision/recall in an
unsupervised setting (analogous to the P1 honest holdout).

Prerequisite: work only on the clean DENSE series pool. Magnitude is ALWAYS
relative to local variance, never absolute.
"""

import numpy as np
from pyspark.sql import Window, functions as F

from src.config import DATA_DIR

INJECTION_SEED = 42



def build_injection_pool(features, n_series=3000, seed=INJECTION_SEED):
    """Pick (series, week) targets for injection.

    One target per series, drawn from the fully-valid zone (yoy_ratio present =>
    at least 52 weeks of history). Returns the chosen series with the target week
    and that week's local stats, needed to size anomalies relative to sigma.
    """
    # Eligible rows: features fully present (in particular yoy + rolling stats).
    eligible = features.filter(
        F.col("yoy_ratio").isNotNull()
        & F.col("roll_std").isNotNull()
        & (F.col("roll_std") > 0)
    )

    # Sample series, then one random week per sampled series.
    series = (
        eligible.select("id").distinct()
        .orderBy(F.rand(seed))
        .limit(n_series)
    )

    pool = (
        eligible.join(series, on="id", how="inner")
        .withColumn("rk", F.row_number().over(
            Window.partitionBy("id").orderBy(F.rand(seed))
        ))
        .filter(F.col("rk") == 1)
        .drop("rk")
    )
    return pool



def inject_spike(pool, k):
    """Inject an upward spike of k local sigmas on each target week.

    new_sales = roll_mean + k * roll_std, rounded to a non-negative integer.
    Returns the pool with injected value, the original, and label metadata.
    """
    return (
        pool
        .withColumn("sales_orig", F.col("sales"))
        .withColumn(
            "sales_injected",
            F.greatest(
                F.lit(0),
                F.round(F.col("roll_mean") + F.lit(k) * F.col("roll_std")),
            ).cast("int"),
        )
        .withColumn("anomaly_type", F.lit("spike"))
        .withColumn("severity_k", F.lit(float(k)))
        .withColumn("label", F.lit(1))
    )



def inject_drop(pool, k):
    """Inject a downward drop of k local sigmas.

    Restricted to series where mean - k*sigma stays > 0, so the drop is a graded
    anomaly rather than a saturated zero. Low-volume series (where any strong
    drop collapses to 0) are excluded from drop evaluation — a documented scope
    limit, analogous to the dense-pool restriction.
    """
    return (
        pool
        .filter(F.col("roll_mean") - F.lit(k) * F.col("roll_std") > 0)
        .withColumn("sales_orig", F.col("sales"))
        .withColumn(
            "sales_injected",
            F.round(F.col("roll_mean") - F.lit(k) * F.col("roll_std")).cast("int"),
        )
        .withColumn("anomaly_type", F.lit("drop"))
        .withColumn("severity_k", F.lit(float(k)))
        .withColumn("label", F.lit(1))
    )



def inject_contextual(pool, features, seed=INJECTION_SEED):
    """Inject a contextual anomaly: replace the target week's sales with the
    series' own value from an opposite-season week (~26 weeks away).

    Value stays plausible in absolute terms but is wrong for the season, so only
    the YoY feature can flag it. No k here — severity is intrinsic to the swap.
    """
    # For each target series, grab a value ~half a year from the target week.
    opposite = (
        features.alias("f")
        .join(pool.select("id", F.col("wm_yr_wk").alias("target_wk")), on="id")
        .withColumn("wk_gap", F.abs((F.col("wm_yr_wk") % 100) - (F.col("target_wk") % 100)))
        .filter((F.col("wk_gap") >= 24) & (F.col("wk_gap") <= 28))   # ~opposite season
        .withColumn("rk", F.row_number().over(
            Window.partitionBy("id").orderBy(F.rand(seed))))
        .filter(F.col("rk") == 1)
        .select("id", F.col("sales").alias("opp_sales"))
    )

    return (
        pool.join(opposite, on="id", how="inner")
        .withColumn("sales_orig", F.col("sales"))
        .withColumn("sales_injected", F.col("opp_sales").cast("int"))
        .withColumn("anomaly_type", F.lit("contextual"))
        .withColumn("severity_k", F.lit(None).cast("double"))
        .withColumn("label", F.lit(1))
        .drop("opp_sales")
    )



def inject_level_shift(pool, features, k, horizon=8, seed=INJECTION_SEED):
    """Inject a sustained level shift: from the target week onward (horizon
    weeks), sales jump by k local sigmas.

    Unlike a spike, this labels MULTIPLE consecutive weeks as anomalous. It's the
    regime-change case a rolling z-score gradually absorbs (after ~13 weeks the
    new level becomes 'normal') — the case that motivates the LSTM autoencoder.
    """
    shift_spec = pool.select(
        "id",
        F.col("wm_yr_wk").alias("shift_start"),
        (F.lit(k) * F.col("roll_std")).alias("shift_amount"),
    )

    shifted = (
        features.join(shift_spec, on="id", how="inner")
        .filter(F.col("wm_yr_wk") >= F.col("shift_start"))
        .withColumn("wk_since",
            F.row_number().over(Window.partitionBy("id").orderBy("wm_yr_wk")) - 1)
        .filter(F.col("wk_since") < horizon)
        .withColumn("sales_orig", F.col("sales"))
        .withColumn("sales_injected",
            F.greatest(F.lit(0),
                F.round(F.col("sales") + F.col("shift_amount"))).cast("int"))
        .withColumn("anomaly_type", F.lit("level_shift"))
        .withColumn("severity_k", F.lit(float(k)))
        .withColumn("label", F.lit(1))
    )
    return shifted



def recompute_features_simple(injected):
    """Recompute the features affected by a single-week injection.

    Only three features depend on the injected value; the rolling stats,
    sales_prev and sales_ly all come from untouched weeks, so no window replay is
    needed — pure arithmetic on existing columns.
      - sales_zscore : (injected - roll_mean) / roll_std   [prior-window stats]
      - wow_delta    : (injected - sales_prev) / sales_prev
      - yoy_ratio    : injected / sales_ly
    snap_days, cat_code, month are value-independent and stay as-is.
    """
    v = F.col("sales_injected")
    return (
        injected
        .withColumn("sales_zscore",
            F.when(F.col("roll_std") > 0,
                   (v - F.col("roll_mean")) / F.col("roll_std")))
        .withColumn("wow_delta",
            F.when(F.col("sales_prev") > 0,
                   (v - F.col("sales_prev")) / F.col("sales_prev")))
        .withColumn("yoy_ratio",
            F.when(F.col("sales_ly") > 0,
                   v / F.col("sales_ly")))
    )



def recompute_features_level_shift(shifted, features):
    """Recompute features for a level-shift injection (Approach A).

    1. Overlay injected values onto the full series (only the plateau weeks
       change; everything else keeps its original sales).
    2. Replay the rolling z-score window on the modified series, so the shift is
       progressively ABSORBED into the rolling mean — reproducing exactly the
       behaviour that makes a rolling detector myopic to sustained regime change.
    3. Recompute wow / yoy arithmetically and return only labelled plateau weeks.
    """
    # Series that received a shift, restricted to those series only (efficiency).
    shift_ids = shifted.select("id").distinct()
    base = features.join(shift_ids, on="id", how="inner")

    # Overlay: bring injected values in, coalesce onto original sales.
    overlay = (
        base.join(
            shifted.select("id", "wm_yr_wk",
                           F.col("sales_injected").alias("inj"),
                           F.col("label").alias("lbl"),
                           "anomaly_type", "severity_k"),
            on=["id", "wm_yr_wk"], how="left")
        .withColumn("sales_mod", F.coalesce(F.col("inj"), F.col("sales")))
    )

    # Replay the rolling window on sales_mod (the shift enters the mean).
    w = Window.partitionBy("id").orderBy("wm_yr_wk").rowsBetween(-13, -1)
    overlay = (
        overlay
        .withColumn("roll_mean_mod", F.mean("sales_mod").over(w))
        .withColumn("roll_std_mod", F.stddev("sales_mod").over(w))
        .withColumn("sales_zscore",
            F.when(F.col("roll_std_mod") > 0,
                   (F.col("sales_mod") - F.col("roll_mean_mod")) / F.col("roll_std_mod")))
    )

    # wow / yoy on modified series (single-lag windows on sales_mod).
    w_lag = Window.partitionBy("id").orderBy("wm_yr_wk").rowsBetween(-1, -1)
    w_ly = Window.partitionBy("id").orderBy("wm_yr_wk").rowsBetween(-52, -52)
    overlay = (
        overlay
        .withColumn("prev_mod", F.first("sales_mod").over(w_lag))
        .withColumn("ly_mod", F.first("sales_mod").over(w_ly))
        .withColumn("wow_delta",
            F.when(F.col("prev_mod") > 0,
                   (F.col("sales_mod") - F.col("prev_mod")) / F.col("prev_mod")))
        .withColumn("yoy_ratio",
            F.when(F.col("ly_mod") > 0, F.col("sales_mod") / F.col("ly_mod")))
    )

    # Keep only the labelled plateau weeks.
    return overlay.filter(F.col("lbl") == 1).withColumnRenamed("lbl", "label")



SEVERITY_GRID = [2.0, 2.5, 3.0, 3.5, 4.0]


def build_eval_set(features, pool, k_grid=SEVERITY_GRID, seed=INJECTION_SEED):
    """Assemble the labelled evaluation set: injected anomalies (label=1) at
    several severities + untouched normal weeks (label=0).

    Same series pool across all k values, so the recall-vs-severity curve varies
    only severity, not the underlying series.
    """
    cols = ["id", "wm_yr_wk", "sales_zscore", "yoy_ratio", "wow_delta",
            "snap_days", "cat_code", "month", "anomaly_type", "severity_k", "label"]

    parts = []

    # k-dependent types: spike, drop, level_shift — one copy per k.
    for k in k_grid:
        parts.append(recompute_features_simple(inject_spike(pool, k)).select(*cols))
        parts.append(recompute_features_simple(inject_drop(pool, k)).select(*cols))
        parts.append(recompute_features_level_shift(
            inject_level_shift(pool, features, k), features).select(*cols))

    # k-independent type: contextual — once.
    parts.append(recompute_features_simple(inject_contextual(pool, features)).select(*cols))

    positives = parts[0]
    for p in parts[1:]:
        positives = positives.unionByName(p)

    # Negatives: sample of normal weeks from the injected series, excluding the
    # exact (id, week) cells that were injected.
    injected_cells = pool.select("id", "wm_yr_wk").withColumn("is_pool", F.lit(1))
    negatives = (
        features.join(pool.select("id").distinct(), on="id", how="inner")
        .join(injected_cells, on=["id", "wm_yr_wk"], how="left")
        .filter(F.col("is_pool").isNull())          # exclude injected cells
        .filter(F.col("yoy_ratio").isNotNull() & (F.col("roll_std") > 0))
        .withColumn("anomaly_type", F.lit("normal"))
        .withColumn("severity_k", F.lit(None).cast("double"))
        .withColumn("label", F.lit(0))
        .select(*cols)
    )

    return positives.unionByName(negatives)