"""Phase 2 — CONTEXTUAL feature engineering.

Core of the project under the "raw sales" signal: the baseline no longer absorbs
seasonality, so the burden of "what is expected" migrates HERE.

GOLDEN RULE: never feed the raw absolute level to the Isolation Forest
(otherwise large stores show up as "anomalous"). We feed DEVIATIONS.
"""

from pyspark.sql import Window, functions as F

from src.config import DATA_DIR



ZSCORE_WINDOW = 13   # weeks; a quarter — stable sigma on dense series, still reactive
YOY_LAG = 52   # weeks; "same week last year" by row position (pool is gap-free)



def trim_cold_start(weekly):
    """Remove leading zero-sales weeks (before a product's first real sale).

    For each series, find the week of its first non-zero sale, then keep only
    weeks at or after it. Leading zeros are 'not yet stocked', not 'zero demand'
    — keeping them would understate density and inject false drops. See EDA-2.
    """
    first_sale = (
        weekly
        .filter(F.col("sales") > 0)
        .groupBy("id")
        .agg(F.min("wm_yr_wk").alias("first_sale_week"))
    )

    trimmed = (
        weekly.join(first_sale, on="id", how="inner")
        .filter(F.col("wm_yr_wk") >= F.col("first_sale_week"))
        .drop("first_sale_week")
    )
    return trimmed



def add_rolling_zscore(df):
    """Rolling z-score of weekly sales, SHIFTED to exclude the current week.

    mean/std computed over the 13 weeks strictly BEFORE the current one, so a
    real anomaly cannot mask itself by inflating its own window statistics.
    """
    # Ordered window, per series. rowsBetween(-13, -1): the 13 prior rows,
    # excluding the current one (that's the shift / leakage guard).
    w = (
        Window.partitionBy("id")
        .orderBy("wm_yr_wk")
        .rowsBetween(-ZSCORE_WINDOW, -1)
    )

    df = df.withColumn("roll_mean", F.mean("sales").over(w))
    df = df.withColumn("roll_std", F.stddev("sales").over(w))
    df = df.withColumn("roll_n", F.count("sales").over(w))   # points in window

    # z = (current - prior mean) / prior std. Guard against std = 0.
    df = df.withColumn(
        "sales_zscore",
        F.when(F.col("roll_std") > 0,
               (F.col("sales") - F.col("roll_mean")) / F.col("roll_std"))
         .otherwise(None),
    )
    return df



def add_yoy_deviation(df):
    """Year-over-year deviation: compare each week to the same week one year
    earlier (52 rows back — valid because the dense pool is gap-free).

    Robust to the global-profile / fiscal-calendar artifacts seen in EDA, since
    it compares each series to its own past, not to a cross-series average.
    """
    # Single-row window: exactly 52 rows back (both bounds = -52).
    w = (
        Window.partitionBy("id")
        .orderBy("wm_yr_wk")
        .rowsBetween(-YOY_LAG, -YOY_LAG)
    )

    df = df.withColumn("sales_ly", F.first("sales").over(w))

    # Relative deviation. Guard against division by zero (sales_ly = 0).
    df = df.withColumn(
        "yoy_ratio",
        F.when(F.col("sales_ly") > 0, F.col("sales") / F.col("sales_ly"))
         .otherwise(None),
    )
    return df



def add_context_features(df):
    """Week-over-week relative delta + encoded category.

    Event flags were tested and dropped: no M5 event shifts weekly food demand
    enough to matter for anomaly detection. Super Bowl +0.5% (noise); NBA Finals
    +3.3% once de-confounded from June seasonality — below any anomaly threshold.
    The YoY seasonal feature already captures the expected June level.
    """
    w_lag = Window.partitionBy("id").orderBy("wm_yr_wk").rowsBetween(-1, -1)
    df = df.withColumn("sales_prev", F.first("sales").over(w_lag))
    df = df.withColumn(
        "wow_delta",
        F.when(F.col("sales_prev") > 0,
               (F.col("sales") - F.col("sales_prev")) / F.col("sales_prev"))
         .otherwise(None),
    )

    cat_map = {"FOODS": 0, "HOUSEHOLD": 1, "HOBBIES": 2}
    mapping = F.create_map([F.lit(x) for kv in cat_map.items() for x in kv])
    df = df.withColumn("cat_code", mapping[F.col("cat_id")])

    return df



# Final feature set — every column justified by EDA, no absolute levels.
FEATURE_COLS = ["sales_zscore", "yoy_ratio", "wow_delta", "snap_days", "cat_code", "month"]

def build_feature_matrix(weekly, dense_ids):
    """Full Phase-2 pipeline: dense filter -> cold-start trim -> features.

    Returns the weekly grid with all feature columns attached. Keeps id/keys and
    raw sales for traceability; the model consumes FEATURE_COLS only.
    """
    df = weekly.join(dense_ids, on="id", how="inner")
    df = trim_cold_start(df)

    df = add_rolling_zscore(df)
    df = add_yoy_deviation(df)
    df = add_context_features(df)

    keep = ["id", "item_id", "store_id", "cat_id", "state_id",
        "wm_yr_wk", "month", "sales",
        "roll_mean", "roll_std", "sales_prev", "sales_ly"] + \
       ["sales_zscore", "yoy_ratio", "wow_delta", "snap_days", "cat_code"]
    return df.select(*dict.fromkeys(keep))   # dedup while preserving order



def write_features(features, out_name="features.parquet"):
    """Persist the feature matrix — Phase 2 checkpoint.

    Reload point for Phase 3 (injection) and Phase 4 (model), avoiding replay of
    the ordered window computations.
    """
    out_path = DATA_DIR / out_name
    features.write.mode("overwrite").parquet(str(out_path))
    return out_path