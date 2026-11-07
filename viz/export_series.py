"""Export scored single-series data for the anomaly timeline viz / Cloud Run demo.

Both models (Isolation Forest + LSTM autoencoder) score every series. The dropdown
leads with a CONTROLLED demo: a calm real series with a clean level shift injected,
clearly labelled — the textbook IF-vs-AE picture (IF flags the onset then goes
quiet as its rolling window absorbs the shift; the AE flags the whole plateau).
Natural contrast series (IF/AE disagreement) follow.

Run from the repo root:  python -m viz.export_series
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.models.isolation_forest import load_bundle, score_rows, FEATURE_COLS
from src.models.autoencoder import (
    load_ae, get_device, build_series_channels, make_windows, reconstruction_error,
    SEASONAL_PERIOD,
)

OUT = Path(__file__).parent / "series_data.js"
MIN_SCORED = 120


# ------------------------------------------------------------------ IF ---------
def _score_all_if():
    feats = pd.read_parquet(DATA_DIR / "features.parquet")
    model, bounds, thr, cols = load_bundle("isolation_forest")
    threshold = float(thr["threshold"])
    valid = feats.dropna(subset=cols).copy()
    valid["anomaly_score"] = score_rows(valid, model, bounds, threshold, cols)["anomaly_score"].values
    return feats, valid, threshold, (model, bounds, cols)


def _recompute_if_scores(sales, base, if_bundle, threshold):
    """IF per-week scores on a (possibly injected) sales array, mirroring the Spark
    feature pipeline in pandas (rolling-13 shifted z-score, YoY, WoW)."""
    model, bounds, cols = if_bundle
    df = pd.DataFrame({"sales": np.asarray(sales, dtype=float)})
    prior = df["sales"].shift(1)
    df["roll_mean"] = prior.rolling(13).mean()
    df["roll_std"] = prior.rolling(13).std()
    df["sales_zscore"] = (df["sales"] - df["roll_mean"]) / df["roll_std"]
    pv = df["sales"].shift(1)
    df["wow_delta"] = (df["sales"] - pv) / pv
    df["yoy_ratio"] = df["sales"] / df["sales"].shift(52)
    for c in ("snap_days", "cat_code", "month"):
        df[c] = base[c].to_numpy()
    df["wk"] = base["wm_yr_wk"].to_numpy()
    for c in ("sales_zscore", "wow_delta", "yoy_ratio"):
        df.loc[~np.isfinite(df[c]), c] = np.nan
    valid = df.dropna(subset=cols)
    if len(valid) == 0:
        return {}
    sc = score_rows(valid, model, bounds, threshold, cols)["anomaly_score"].values
    return {int(w): float(s) for w, s in zip(valid["wk"].to_numpy(), sc)}


# ------------------------------------------------------------------ AE ---------
def _channels(sales, woy, med, iqr):
    normed = (sales - med) / iqr
    ang = 2 * np.pi * (woy % SEASONAL_PERIOD) / SEASONAL_PERIOD
    return np.stack([normed, np.sin(ang), np.cos(ang)], axis=1)


def _ae_scores_from(sales, woy, model, device, med, iqr):
    chans = _channels(np.asarray(sales, float), woy, med, iqr)
    wins, centers = make_windows(chans)
    if len(wins) == 0:
        return {}
    errs = reconstruction_error(model, wins.astype(np.float32), device)
    return {int(c): float(e) for c, e in zip(centers, errs)}   # keyed by POSITION


def _ae_scores_for_series(s, model, device):
    chans, _, _ = build_series_channels(s)
    wins, centers = make_windows(chans)
    if len(wins) == 0:
        return {}
    errs = reconstruction_error(model, wins.astype(np.float32), device)
    wks = s["wm_yr_wk"].to_numpy()
    return {int(wks[c]): float(e) for c, e in zip(centers, errs)}


def _ae_threshold(feats, model, device, n_series=200, fp=0.05, seed=0):
    ids = feats["id"].drop_duplicates()
    sample = ids.sample(min(n_series, len(ids)), random_state=seed)
    errs = []
    for _, s in feats[feats["id"].isin(sample)].sort_values("wm_yr_wk").groupby("id"):
        chans, _, _ = build_series_channels(s)
        wins, centers = make_windows(chans)
        if len(wins) == 0:
            continue
        z = s["sales_zscore"].to_numpy()
        e = reconstruction_error(model, wins.astype(np.float32), device)
        errs += [err for c, err in zip(centers, e) if np.isfinite(z[c]) and abs(z[c]) < 2]
    return float(np.quantile(errs, 1 - fp))


# ---------------------------------------------------- controlled demo ----------
def _make_injected_demo(feats, valid, if_bundle, if_thr, ae_model, device,
                        k=3.0, horizon=12):
    """A calm base series + a clean injected level shift. Returns a series dict."""
    v = valid.copy()
    v["flag"] = v["anomaly_score"] >= if_thr
    agg = (v.groupby("id")
             .agg(scored=("anomaly_score", "size"), flags=("flag", "sum"),
                  med=("sales", "median")).reset_index())
    agg = agg[(agg["scored"] >= 150) & (agg["med"].between(20, 120))]
    base_id = agg.sort_values("flags").iloc[0]["id"]      # calmest suitable series

    s = feats[feats["id"] == base_id].sort_values("wm_yr_wk").reset_index(drop=True)
    sales = s["sales"].to_numpy(dtype=float)
    woy = s["wm_yr_wk"].to_numpy() % 100
    med = float(np.median(sales))
    q1, q3 = np.percentile(sales, [25, 75])
    iqr = max(q3 - q1, 1.0)

    t0 = len(sales) // 2
    offset = round(k * iqr)
    inj = sales.copy()
    inj[t0:t0 + horizon] = inj[t0:t0 + horizon] + offset

    if_by_wk = _recompute_if_scores(inj, s, if_bundle, if_thr)
    ae_by_pos = _ae_scores_from(inj, woy, ae_model, device, med, iqr)
    wks = s["wm_yr_wk"].to_numpy()
    weeks = [{
        "wk": int(wks[i]),
        "sales": int(inj[i]),
        "scores": {
            "if": (None if int(wks[i]) not in if_by_wk else round(if_by_wk[int(wks[i])], 4)),
            "ae": (None if i not in ae_by_pos else round(ae_by_pos[i], 4)),
        },
    } for i in range(len(s))]
    return {"id": f"DEMO · injected level shift (+{offset} for {horizon}w)",
            "cat": "DEMO", "store": "", "weeks": weeks}


# ---------------------------------------------------- natural contrast ---------
def _longest_true_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _pick_contrast_series(feats, valid, if_thr, ae_model, device, ae_thr,
                          sample=600, n_ae=5, n_if=4, min_scored=MIN_SCORED, seed=0):
    counts = valid.groupby("id").size()
    eligible = counts[counts >= min_scored].index.to_numpy()
    rng = np.random.default_rng(seed)
    cand = set(rng.choice(eligible, size=min(sample, len(eligible)), replace=False))
    sub = feats[feats["id"].isin(cand)]
    if_scores_map = {sid: g.set_index("wm_yr_wk")["anomaly_score"].to_dict()
                     for sid, g in valid[valid["id"].isin(cand)].groupby("id")}
    rows = []
    for sid, s in sub.sort_values("wm_yr_wk").groupby("id"):
        ae = _ae_scores_for_series(s, ae_model, device)
        wks = s["wm_yr_wk"].to_numpy()
        ifs = if_scores_map.get(sid, {})
        ae_flag = np.array([ae.get(int(w), -1) >= ae_thr for w in wks])
        if_flag = np.array([ifs.get(int(w), -1) >= if_thr for w in wks])
        rows.append((sid, int(if_flag.sum()), int(ae_flag.sum()),
                     _longest_true_run(ae_flag & ~if_flag)))
    d = pd.DataFrame(rows, columns=["id", "if_flags", "ae_flags", "ae_only_run"])
    print("top AE-only runs (natural level shifts the IF misses):")
    print(d.sort_values("ae_only_run", ascending=False).head(8).to_string(index=False))
    ae_show = list(d.sort_values("ae_only_run", ascending=False).head(n_ae)["id"])
    if_show = list(d.sort_values("if_flags", ascending=False).head(n_if)["id"])
    return list(dict.fromkeys(ae_show + if_show))


# -------------------------------------------------------------- export ---------
def _series_dict(feats, valid, sid, ae_model, device):
    s = feats[feats["id"] == sid].sort_values("wm_yr_wk")
    if_by = valid[valid["id"] == sid].set_index("wm_yr_wk")["anomaly_score"].to_dict()
    ae_by = _ae_scores_for_series(s, ae_model, device)
    weeks = [{
        "wk": int(r["wm_yr_wk"]), "sales": int(r["sales"]),
        "scores": {
            "if": (None if int(r["wm_yr_wk"]) not in if_by else round(float(if_by[int(r["wm_yr_wk"])]), 4)),
            "ae": (None if int(r["wm_yr_wk"]) not in ae_by else round(float(ae_by[int(r["wm_yr_wk"])]), 4)),
        }} for _, r in s.iterrows()]
    m = s.iloc[0]
    return {"id": sid, "cat": str(m["cat_id"]), "store": str(m["store_id"]), "weeks": weeks}


def export():
    feats, valid, if_thr, if_bundle = _score_all_if()
    device = get_device()
    ae_model, _ = load_ae()
    ae_thr = _ae_threshold(feats, ae_model, device)
    print(f"IF threshold={if_thr:.4f} | AE threshold={ae_thr:.4f}")

    demo = _make_injected_demo(feats, valid, if_bundle, if_thr, ae_model, device)
    natural = _pick_contrast_series(feats, valid, if_thr, ae_model, device, ae_thr)

    out = {
        "models": [
            {"key": "if", "label": "Isolation Forest", "threshold": round(if_thr, 4), "color": "#e4572e"},
            {"key": "ae", "label": "LSTM Autoencoder", "threshold": round(ae_thr, 4), "color": "#4c6ef5"},
        ],
        "series": [demo] + [_series_dict(feats, valid, sid, ae_model, device) for sid in natural],
    }
    OUT.write_text("window.SERIES_DATA = " + json.dumps(out) + ";\n")
    print(f"wrote {OUT} — {len(out['series'])} series (1 demo + {len(natural)} natural)")


if __name__ == "__main__":
    export()
