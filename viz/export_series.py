"""Export scored single-series data for the anomaly timeline viz / Cloud Run demo.

Reads the feature matrix, scores every valid (series, week) with the standalone
IF bundle, and writes a browseable set of series to viz/series_data.js:

    window.SERIES_DATA = {
      "models": [{"key": "if", "label": ..., "threshold": ..., "color": ...}],
      "series": [{"id", "cat", "store", "weeks": [{"wk", "sales", "scores": {"if": ...}}]}]
    }

Demo curation: only LONG series (>= MIN_SCORED scored weeks) are eligible, so the
timeline is mostly scorable rather than mostly in the no-YoY zone. Per category we
pick a spread from calmest to busiest by flag count, for readable variety.

Multi-model ready: when the autoencoder exists, score its series too, add an "ae"
entry to models + an "ae" key to each week's scores — the viz already loops.

Run from the repo root:  python -m viz.export_series
"""
import json
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR
from src.models.isolation_forest import load_bundle, score_rows

OUT = Path(__file__).parent / "series_data.js"
MIN_SCORED = 120          # a series needs this many scorable weeks to be demo-eligible
PER_CATEGORY = 4          # series shown per category (spread calm -> busy)


def _score_all():
    feats = pd.read_parquet(DATA_DIR / "features.parquet")
    model, bounds, thr, cols = load_bundle("isolation_forest")
    threshold = float(thr["threshold"])
    valid = feats.dropna(subset=cols).copy()
    valid["anomaly_score"] = score_rows(valid, model, bounds, threshold, cols)["anomaly_score"].values
    return feats, valid, threshold


def _pick_demo_series(valid, threshold, per_category=PER_CATEGORY, min_scored=MIN_SCORED):
    """Long series only; per category, a spread from fewest to most flagged weeks."""
    v = valid.copy()
    v["flag"] = v["anomaly_score"] >= threshold
    agg = (v.groupby(["cat_id", "id"])
             .agg(scored=("anomaly_score", "size"), flags=("flag", "sum"))
             .reset_index())
    agg = agg[agg["scored"] >= min_scored]

    picks = []
    for _, g in agg.groupby("cat_id"):
        g = g.sort_values("flags").reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue
        if n <= per_category:
            idx = list(range(n))
        else:                                   # evenly spaced: calm -> busy
            idx = [round(i * (n - 1) / (per_category - 1)) for i in range(per_category)]
        picks += list(g.loc[idx, "id"])
    return list(dict.fromkeys(picks))


def export(series_ids=None):
    feats, valid, threshold = _score_all()
    if series_ids is None:
        series_ids = _pick_demo_series(valid, threshold)

    out = {
        "models": [{"key": "if", "label": "Isolation Forest",
                    "threshold": round(threshold, 4), "color": "#e4572e"}],
        "series": [],
    }

    for sid in series_ids:
        s_all = feats[feats["id"] == sid].sort_values("wm_yr_wk")
        score_by_wk = (valid[valid["id"] == sid]
                       .set_index("wm_yr_wk")["anomaly_score"].to_dict())
        weeks = [{
            "wk": int(row["wm_yr_wk"]),
            "sales": int(row["sales"]),
            "scores": {"if": (None if int(row["wm_yr_wk"]) not in score_by_wk
                              else round(float(score_by_wk[int(row["wm_yr_wk"])]), 4))},
        } for _, row in s_all.iterrows()]
        meta = s_all.iloc[0]
        out["series"].append({"id": sid, "cat": str(meta["cat_id"]),
                              "store": str(meta["store_id"]), "weeks": weeks})

    OUT.write_text("window.SERIES_DATA = " + json.dumps(out) + ";\n")
    print(f"wrote {OUT} — {len(out['series'])} series, threshold={threshold:.4f}")


if __name__ == "__main__":
    export()
