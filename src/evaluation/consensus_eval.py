"""Unified IF + AE evaluation and consensus — src/evaluation/consensus_eval.py

Scores the SAME injected targets with BOTH models so they can be combined:
  - IF score: features recomputed on the injected sales (pandas), scored by the
    IF bundle.
  - AE score: reconstruction error of the window centred on the target week.

Consensus = max of each model's percentile rank against clean negatives (a single
ranked score with a controlled FP budget). The comparison shows the consensus
recovers the IF's spike strength AND the AE's level-shift strength.
"""
import numpy as np
import pandas as pd

from src.models.isolation_forest import recompute_if_scores_pandas
from src.models.autoencoder import (
    reconstruction_error, robust_scale, SEASONAL_PERIOD, WINDOW_L,
)

L = WINDOW_L
HALF = L // 2
HORIZON = 8


def _channels(sales, woy, med, iqr):
    normed = (sales - med) / iqr
    ang = 2 * np.pi * (woy % SEASONAL_PERIOD) / SEASONAL_PERIOD
    return np.stack([normed, np.sin(ang), np.cos(ang)], axis=1)


def _add_single(wins, meta, base, mod, woy, med, iqr, t, atype, k, if_bundle, if_thr):
    """One modified-week target (spike/drop/contextual): append its window + IF score."""
    n = len(mod)
    start = t - HALF
    if start < 0 or start + L > n:
        return
    ifsc = recompute_if_scores_pandas(mod, base, if_bundle, if_thr)
    wk = int(base["wm_yr_wk"].to_numpy()[t])
    if wk not in ifsc:
        return
    ch = _channels(mod, woy, med, iqr)
    wins.append(ch[start:start + L])
    meta.append((atype, k, 1, ifsc[wk]))


def build_consensus_eval(feats, valid, pool_pd, if_bundle, if_thr, ae_model, device,
                         k_grid=(2.0, 2.5, 3.0, 3.5, 4.0)):
    """Returns a DataFrame [anomaly_type, severity_k, label, if_score, ae_score]."""
    targets = dict(zip(pool_pd["id"], pool_pd["wm_yr_wk"]))
    if_neg_map = {sid: g.set_index("wm_yr_wk")["anomaly_score"].to_dict()
                  for sid, g in valid[valid["id"].isin(targets)].groupby("id")}
    groups = {sid: g for sid, g in feats[feats["id"].isin(targets)].groupby("id")}

    all_rows = []
    for sid, wk_t in targets.items():
        s = groups.get(sid)
        if s is None:
            continue
        s = s.sort_values("wm_yr_wk").reset_index(drop=True)
        n = len(s)
        if n < L:
            continue
        sales = s["sales"].to_numpy(dtype=float)
        woy = s["wm_yr_wk"].to_numpy() % 100
        wks = s["wm_yr_wk"].to_numpy()
        z = s["sales_zscore"].to_numpy()
        rmean = s["roll_mean"].to_numpy()
        rstd = s["roll_std"].to_numpy()
        _, med, iqr = robust_scale(sales)
        clean_ch = _channels(sales, woy, med, iqr)
        tp = np.where(wks == wk_t)[0]
        if len(tp) == 0:
            continue
        t = int(tp[0])

        wins, meta = [], []          # meta: (type, k, label, if_score)

        # ---- negatives: clean windows at |z| < 2 weeks (IF score from features.parquet) ----
        ifn = if_neg_map.get(sid, {})
        for pos in range(HALF, n - HALF):
            if pos == t or not np.isfinite(z[pos]) or abs(z[pos]) >= 2:
                continue
            wk = int(wks[pos])
            if wk in ifn:
                wins.append(clean_ch[pos - HALF:pos + HALF + 1])
                meta.append(("normal", np.nan, 0, float(ifn[wk])))

        rm, rs = rmean[t], rstd[t]
        if np.isfinite(rm) and np.isfinite(rs) and rs > 0:
            for k in k_grid:
                mod = sales.copy(); mod[t] = max(0.0, round(rm + k * rs))
                _add_single(wins, meta, s, mod, woy, med, iqr, t, "spike", k, if_bundle, if_thr)
                if rm - k * rs > 0:
                    mod = sales.copy(); mod[t] = round(rm - k * rs)
                    _add_single(wins, meta, s, mod, woy, med, iqr, t, "drop", k, if_bundle, if_thr)
                end = min(t + HORIZON, n)
                mod = sales.copy()
                mod[t:end] = np.maximum(0.0, np.round(sales[t:end] + k * rs))
                ifsc = recompute_if_scores_pandas(mod, s, if_bundle, if_thr)
                ch = _channels(mod, woy, med, iqr)
                for pos in range(t, end):
                    start = pos - HALF
                    wk = int(wks[pos])
                    if start >= 0 and start + L <= n and wk in ifsc:
                        wins.append(ch[start:start + L])
                        meta.append(("level_shift", k, 1, ifsc[wk]))

            gap = np.abs((woy % SEASONAL_PERIOD) - (woy[t] % SEASONAL_PERIOD))
            cand = np.where((gap >= 24) & (gap <= 28))[0]
            if len(cand):
                src = int(np.random.default_rng(t).choice(cand))
                mod = sales.copy(); mod[t] = sales[src]
                _add_single(wins, meta, s, mod, woy, med, iqr, t, "contextual", np.nan, if_bundle, if_thr)

        if not wins:
            continue
        errs = reconstruction_error(ae_model, np.asarray(wins, dtype=np.float32), device)
        for (atype, k, label, ifs), ae in zip(meta, errs):
            all_rows.append((atype, k, label, ifs, float(ae)))

    return pd.DataFrame(all_rows,
                        columns=["anomaly_type", "severity_k", "label", "if_score", "ae_score"])


def add_consensus(df):
    """Percentile-rank each model against negatives; consensus = max(p_if, p_ae)."""
    neg = df[df["label"] == 0]
    if_sorted = np.sort(neg["if_score"].to_numpy())
    ae_sorted = np.sort(neg["ae_score"].to_numpy())
    out = df.copy()
    out["p_if"] = np.searchsorted(if_sorted, out["if_score"].to_numpy(), side="right") / len(if_sorted)
    out["p_ae"] = np.searchsorted(ae_sorted, out["ae_score"].to_numpy(), side="right") / len(ae_sorted)
    out["consensus"] = np.maximum(out["p_if"], out["p_ae"])
    return out


def recall_table(df, fp=0.05):
    """Recall@fp per type for IF / AE / consensus, each at its own fp threshold."""
    df = add_consensus(df)
    neg = df[df["label"] == 0]
    thr = {m: neg[m].quantile(1 - fp) for m in ("if_score", "ae_score", "consensus")}
    print(f"=== recall @ {int(fp*100)}% FP  —  IF / AE / consensus ===")
    for atype in ["spike", "drop", "contextual", "level_shift"]:
        pt = df[(df["label"] == 1) & (df["anomaly_type"] == atype)]
        ks = sorted(pt["severity_k"].dropna().unique()) if pt["severity_k"].notna().any() else [None]
        for k in ks:
            p = pt if k is None else pt[pt["severity_k"] == k]
            r_if = (p["if_score"] >= thr["if_score"]).mean()
            r_ae = (p["ae_score"] >= thr["ae_score"]).mean()
            r_co = (p["consensus"] >= thr["consensus"]).mean()
            tag = atype if k is None else f"{atype} k={k}"
            print(f"{tag:16s}  IF={r_if:5.1%}  AE={r_ae:5.1%}  consensus={r_co:5.1%}")


def agreement_tiers(df, fp=0.05):
    """Label each row by which models flag it at their own fp threshold:
      'both'    = high confidence (both models agree)
      'if_only' = likely a point shock (the IF's turf)
      'ae_only' = likely a regime change (the AE's turf)
      'none'    = not flagged
    These tiers give a planner a confidence level AND a nature hint per alert."""
    neg = df[df["label"] == 0]
    if_thr = neg["if_score"].quantile(1 - fp)
    ae_thr = neg["ae_score"].quantile(1 - fp)
    out = df.copy()
    f_if = out["if_score"] >= if_thr
    f_ae = out["ae_score"] >= ae_thr
    out["tier"] = np.select(
        [f_if & f_ae, f_if & ~f_ae, ~f_if & f_ae],
        ["both", "if_only", "ae_only"], default="none")
    return out


def tier_distribution(df, fp=0.05):
    """How injected anomalies split across tiers, per type — validates that
    if_only ~ spikes and ae_only ~ level shifts."""
    pos = agreement_tiers(df, fp)
    pos = pos[pos["label"] == 1]
    tab = (pos.groupby(["anomaly_type", "tier"]).size()
              .unstack(fill_value=0)
              .reindex(columns=["both", "if_only", "ae_only", "none"], fill_value=0))
    print(tab)
    return tab


def log_consensus_run(df, run_name="consensus", fp_budgets=(0.05, 0.01)):
    """Log the consensus to the SAME MLflow experiment as IF and AE, recall by
    step=severity so all three overlay in the UI."""
    import mlflow
    from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

    d = add_consensus(df)
    neg = d[d["label"] == 0]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model", "consensus_max_percentile")
        mlflow.log_param("combine", "max(p_if, p_ae)")
        for b in fp_budgets:
            thr = neg["consensus"].quantile(1 - b)
            for atype in ["spike", "drop", "contextual", "level_shift"]:
                pt = d[(d["label"] == 1) & (d["anomaly_type"] == atype)]
                name = f"recall_{atype}_fp{int(b*100)}"
                if pt["severity_k"].notna().any():
                    for k in sorted(pt["severity_k"].dropna().unique()):
                        p = pt[pt["severity_k"] == k]
                        mlflow.log_metric(name, float((p["consensus"] >= thr).mean()), step=int(k * 10))
                else:
                    mlflow.log_metric(name, float((pt["consensus"] >= thr).mean()))
        return run.info.run_id
