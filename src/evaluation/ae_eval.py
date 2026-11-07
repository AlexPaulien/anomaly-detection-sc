"""AE evaluation on injected anomalies — the apples-to-apples counterpart to the
IF evaluation, so IF vs AE can be compared per type at the same FP budget.

Mirrors the injection protocol in pandas (the AE scores sequence WINDOWS, not the
tabular feature vectors the IF uses). For each target we:
  1. take the series' clean sales, compute its robust scaler (median/IQR),
  2. apply the injection to the raw sales (same k*sigma logic as the Spark harness),
  3. normalize the modified sales with the CLEAN scaler (so the injection shows as
     a deviation rather than being absorbed),
  4. build the window centered on the target week, score it by reconstruction error.

Negatives are clearly-normal weeks (|z| < 2), scored from the clean sequence — the
same negative definition as the IF eval.
"""
import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.models.autoencoder import (
    robust_scale, reconstruction_error, WINDOW_L, SEASONAL_PERIOD,
)

L = WINDOW_L
HALF = L // 2
HORIZON = 8               # level-shift plateau length (matches the IF injection)


def _channels(sales, woy, med, iqr):
    """(T,3) array [sales_norm, sin, cos] using a GIVEN scaler (the clean one)."""
    normed = (sales - med) / iqr
    ang = 2 * np.pi * (woy % SEASONAL_PERIOD) / SEASONAL_PERIOD
    return np.stack([normed, np.sin(ang), np.cos(ang)], axis=1)


def _window(ch, center):
    """Window of length L centered at `center`, or None if it runs off an edge."""
    start = center - HALF
    if start < 0 or start + L > len(ch):
        return None
    return ch[start:start + L]


def build_ae_eval(feats, pool_pd, model, device, k_grid=(2.0, 2.5, 3.0, 3.5, 4.0),
                  seed=42):
    """Build + score the AE eval set. Returns a DataFrame [label, anomaly_type,
    severity_k, ae_score]."""
    rng = np.random.default_rng(seed)
    targets = dict(zip(pool_pd["id"], pool_pd["wm_yr_wk"]))

    wins, meta = [], []   # accumulate windows and their (type, k, label)

    def add(w, atype, k, label):
        if w is not None:
            wins.append(w)
            meta.append((atype, k, label))

    for sid in pool_pd["id"].unique():
        s = (feats[feats["id"] == sid].sort_values("wm_yr_wk").reset_index(drop=True))
        if len(s) < L:
            continue
        sales = s["sales"].to_numpy(dtype=float)
        woy = s["wm_yr_wk"].to_numpy() % 100
        _, med, iqr = robust_scale(sales)              # clean scaler
        clean_ch = _channels(sales, woy, med, iqr)
        z = s["sales_zscore"].to_numpy()
        rmean = s["roll_mean"].to_numpy()
        rstd = s["roll_std"].to_numpy()

        tpos = s.index[s["wm_yr_wk"] == targets[sid]]
        if len(tpos) == 0:
            continue
        t = int(tpos[0])

        # ---- NEGATIVES: clean windows at clearly-normal weeks (|z| < 2) ----
        for pos in range(HALF, len(s) - HALF):
            if pos == t or not np.isfinite(z[pos]) or abs(z[pos]) >= 2:
                continue
            add(clean_ch[pos - HALF: pos + HALF + 1], "normal", np.nan, 0)

        rm, rs = rmean[t], rstd[t]
        has_sigma = np.isfinite(rm) and np.isfinite(rs) and rs > 0

        # ---- POSITIVES ----
        for k in k_grid:
            if has_sigma:
                # spike
                mod = sales.copy()
                mod[t] = max(0.0, round(rm + k * rs))
                add(_window(_channels(mod, woy, med, iqr), t), "spike", k, 1)

                # drop (volume-restricted: mean - k*sigma must stay > 0)
                if rm - k * rs > 0:
                    mod = sales.copy()
                    mod[t] = round(rm - k * rs)
                    add(_window(_channels(mod, woy, med, iqr), t), "drop", k, 1)

                # level shift: plateau of HORIZON weeks from t, each week a target
                end = min(t + HORIZON, len(s))
                mod = sales.copy()
                mod[t:end] = np.maximum(0.0, np.round(sales[t:end] + k * rs))
                ch = _channels(mod, woy, med, iqr)
                for pos in range(t, end):
                    add(_window(ch, pos), "level_shift", k, 1)

        # ---- CONTEXTUAL (no k): swap in a value from ~half a year away ----
        woy_t = woy[t]
        gap = np.abs((woy % SEASONAL_PERIOD) - (woy_t % SEASONAL_PERIOD))
        cand = np.where((gap >= 24) & (gap <= 28))[0]
        if len(cand):
            src = int(rng.choice(cand))
            mod = sales.copy()
            mod[t] = sales[src]
            add(_window(_channels(mod, woy, med, iqr), t), "contextual", np.nan, 1)

    X = np.asarray(wins, dtype=np.float32)
    scores = reconstruction_error(model, X, device)
    df = pd.DataFrame(meta, columns=["anomaly_type", "severity_k", "label"])
    df["ae_score"] = scores
    return df


def recall_by_type_ae(df, fp_budgets=(0.05, 0.01)):
    """Recall at fixed FP budget per type and severity (same metric as the IF)."""
    neg = df[df["label"] == 0]["ae_score"]
    thr = {b: neg.quantile(1 - b) for b in fp_budgets}
    print("=== AE recall @ " + " / ".join(f"{int(b*100)}%" for b in fp_budgets) + " FP ===")
    for atype in ["spike", "drop", "contextual", "level_shift"]:
        pt = df[(df["label"] == 1) & (df["anomaly_type"] == atype)]
        if pt["severity_k"].notna().any():
            for k in sorted(pt["severity_k"].dropna().unique()):
                p = pt[pt["severity_k"] == k]
                cells = "  ".join(f"recall@{int(b*100)}%={(p.ae_score >= thr[b]).mean():.1%}"
                                  for b in fp_budgets)
                print(f"{atype:12s} k={k}: {cells}")
        else:
            cells = "  ".join(f"recall@{int(b*100)}%={(pt.ae_score >= thr[b]).mean():.1%}"
                              for b in fp_budgets)
            print(f"{atype:12s}    : {cells}")
