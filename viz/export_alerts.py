"""Export the weekly alert queue for the triage dashboard -> viz/alerts_data.js

For each of the last N scorable weeks, every dense series is scored by both
models, each score is converted to its percentile rank against clearly-normal
weeks, and alerts are ranked by the consensus max(p_if, p_ae) at a 5% FP budget.
Each alert carries its agreement tier (both / if_only / ae_only).

The AE scores a 13-week window CENTRED on the week, so the most recent 6 weeks
are not scorable yet — the dashboard states this lag.

Run from the repo root:  python -m viz.export_alerts
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.models.isolation_forest import load_bundle, score_rows
from src.models.autoencoder import (
    load_ae, get_device, build_series_channels, reconstruction_error, WINDOW_L,
)

OUT = Path(__file__).parent / "alerts_data.js"
L, HALF = WINDOW_L, WINDOW_L // 2
FP = 0.05
HIST_BEFORE = 45            # weeks of history shown before the alert week


def export(n_weeks=12, top_n=150, ref_series=400, seed=0):
    feats = pd.read_parquet(DATA_DIR / "features.parquet")
    feats = feats.sort_values(["id", "wm_yr_wk"]).reset_index(drop=True)

    # ---- IF score for every row with complete features -----------------------
    if_model, bounds, thr, cols = load_bundle("isolation_forest")
    ok = feats[cols].notna().all(axis=1)
    feats["if_score"] = np.nan
    feats.loc[ok, "if_score"] = score_rows(
        feats.loc[ok], if_model, bounds, float(thr["threshold"]), cols)["anomaly_score"].values

    weeks_all = np.sort(feats["wm_yr_wk"].unique())
    sel = weeks_all[-(HALF + n_weeks):-HALF]          # last n AE-scorable weeks
    rng = np.random.default_rng(seed)
    ids = feats["id"].unique()
    ref_ids = set(rng.choice(ids, size=min(ref_series, len(ids)), replace=False))

    # ---- build AE windows: selected weeks for all series, all weeks for the
    #      reference sample (used to calibrate percentiles on normal weeks) -----
    wins, meta, store = [], [], {}
    for sid, s in feats.groupby("id", sort=False):
        n = len(s)
        if n < L:
            continue
        wks = s["wm_yr_wk"].to_numpy()
        chans, _, _ = build_series_channels(s)
        pos_sel = set(np.where(np.isin(wks, sel))[0])
        cand = range(HALF, n - HALF) if sid in ref_ids else sorted(pos_sel)
        z = s["sales_zscore"].to_numpy()
        ifs = s["if_score"].to_numpy()
        rm = s["roll_mean"].to_numpy()
        sales = s["sales"].to_numpy()
        kept = False
        for p in cand:
            if p < HALF or p >= n - HALF:
                continue
            wins.append(chans[p - HALF:p + HALF + 1])
            meta.append((sid, p, int(wks[p]), p in pos_sel, sid in ref_ids,
                         z[p], ifs[p], rm[p], int(sales[p])))
            kept = True
        if kept:
            store[sid] = (sales.astype(np.int32), str(s["cat_id"].iloc[0]), str(s["store_id"].iloc[0]))

    device = get_device()
    ae_model, _ = load_ae()
    ae = reconstruction_error(ae_model, np.asarray(wins, dtype=np.float32), device)
    m = pd.DataFrame(meta, columns=["id", "pos", "wk", "sel", "ref", "z", "if_score", "usual", "sales"])
    m["ae_score"] = ae
    m = m[m["if_score"].notna()]                       # both scores required

    # ---- percentile calibration on clearly-normal reference weeks ------------
    neg = m[m["ref"] & (m["z"].abs() < 2)]
    if_sorted, ae_sorted = np.sort(neg["if_score"].to_numpy()), np.sort(neg["ae_score"].to_numpy())
    rank = lambda arr, v: np.searchsorted(arr, v, side="right") / len(arr)
    c_thr = float(np.quantile(np.maximum(rank(if_sorted, neg["if_score"].to_numpy()),
                                         rank(ae_sorted, neg["ae_score"].to_numpy())), 1 - FP))

    a = m[m["sel"]].copy()
    a["pif"] = rank(if_sorted, a["if_score"].to_numpy())
    a["pae"] = rank(ae_sorted, a["ae_score"].to_numpy())
    a["c"] = np.maximum(a["pif"], a["pae"])
    f_if, f_ae = a["pif"] >= 1 - FP, a["pae"] >= 1 - FP
    a["tier"] = np.select([f_if & f_ae, f_if & ~f_ae, ~f_if & f_ae],
                          ["both", "if_only", "ae_only"], default="none")
    a["flag"] = (a["c"] >= c_thr) & (a["tier"] != "none")
    print(f"reference normal weeks: {len(neg):,} | consensus threshold: {c_thr:.4f}")

    out = {"meta": {"weeks": [int(w) for w in sel], "fp_budget": FP, "window": L,
                    "lag_weeks": HALF, "consensus_threshold": round(c_thr, 4)},
           "weeks": {}}
    wk_index = {int(w): i for i, w in enumerate(weeks_all)}
    for wk, g in a.groupby("wk"):
        fl = g[g["flag"]].copy()
        fl["s2"] = fl["pif"] + fl["pae"]
        fl["absz"] = fl["z"].abs()
        top = fl.sort_values(["c", "s2", "absz"], ascending=False).head(top_n)
        alerts = []
        for r in top.itertuples():
            sales, cat, st = store[r.id]
            lo = max(0, r.pos - HIST_BEFORE)
            alerts.append({
                "id": r.id.replace("_evaluation", ""), "cat": cat, "store": st,
                "sales": int(r.sales), "usual": int(round(r.usual)) if np.isfinite(r.usual) else 0,
                "pif": round(float(r.pif), 4), "pae": round(float(r.pae), 4),
                "c": round(float(r.c), 4), "tier": r.tier,
                "h": [int(v) for v in sales[lo:r.pos + HALF + 1]], "hi": int(r.pos - lo),
            })
        i = wk_index[int(wk)]
        vc = fl["tier"].value_counts()
        out["weeks"][int(wk)] = {
            "hw": [int(w) for w in weeks_all[max(0, i - HIST_BEFORE):i + HALF + 1]],
            "alerts": alerts,
            "counts": {"total": int(len(fl)), "scored": int(len(g)),
                       "both": int(vc.get("both", 0)), "if_only": int(vc.get("if_only", 0)),
                       "ae_only": int(vc.get("ae_only", 0))},
        }
        print(f"week {wk}: {len(fl):4d} flagged / {len(g):,} scored "
              f"(both {vc.get('both', 0)}, IF only {vc.get('if_only', 0)}, AE only {vc.get('ae_only', 0)})")

    OUT.write_text("window.ALERTS_DATA = " + json.dumps(out) + ";\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    export()
