import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from pyspark.sql import functions as F

from src.config import DATA_DIR



FEATURE_COLS = ["sales_zscore", "yoy_ratio", "wow_delta", "snap_days", "cat_code", "month"]

def load_training_matrix(spark):
    """Load the feature matrix, keep fully-valid rows, bring to pandas.

    Unsupervised: the model trains on ALL dense series (not just the injection
    pool). Rows lacking YoY (first 52 weeks) or rolling stats are dropped — no
    imputation of a 'last year' that doesn't exist.
    """
    features = spark.read.parquet(str(DATA_DIR / "features.parquet"))

    # Keep only rows where every model feature is present.
    valid = features
    for c in FEATURE_COLS:
        valid = valid.filter(F.col(c).isNotNull())

    # Bring to pandas
    pdf = valid.select(
        "id", "wm_yr_wk", *FEATURE_COLS
    ).toPandas()

    return pdf



CLIP_COLS = ["sales_zscore", "yoy_ratio", "wow_delta"]   # derived, unbounded
CLIP_Q = 0.999

def fit_clip_bounds(X, cols=CLIP_COLS, q=CLIP_Q):
    """Learn symmetric clip bounds (lower = 1-q, upper = q quantile) on training.

    Returns {col: (low, high)} to be applied identically to train AND eval, so
    injected anomalies are scaled the same way as the data the model learned on.
    """
    bounds = {}
    for c in cols:
        low = X[c].quantile(1 - q)
        high = X[c].quantile(q)
        bounds[c] = (float(low), float(high))
    return bounds



def apply_clip(X, bounds):
    """Apply precomputed clip bounds. Pure, no fitting — safe for eval reuse."""
    X = X.copy()
    for c, (low, high) in bounds.items():
        X[c] = X[c].clip(lower=low, upper=high)
    return X



IF_PARAMS = dict(
    n_estimators=200,
    max_samples=256,       # standard IF subsampling; keeps trees shallow & fast
    contamination="auto",  # we set our OWN threshold later via business cost
    random_state=42,
    n_jobs=-1,
)


def train_isolation_forest(X, bounds, cols=FEATURE_COLS, params=IF_PARAMS):
    """Fit an unsupervised Isolation Forest on the full dense matrix.

    X is clipped with the provided (training) bounds first. contamination is left
    at 'auto' on purpose — the operating threshold is chosen later from the
    precision/recall trade-off, not fixed here.
    """
    X_clipped = apply_clip(X, bounds)
    model = IsolationForest(**params)
    model.fit(X_clipped[cols])
    return model



def score_eval_set(model, eval_pdf, bounds, cols=FEATURE_COLS):
    """Score the labelled eval set (higher = more anomalous).

    Rows with any NaN feature are dropped before scoring — the model was trained
    on fully-valid rows and can only score fully-valid rows. Count is reported.
    """
    n_before = len(eval_pdf)
    eval_pdf = eval_pdf.dropna(subset=cols).copy()
    n_dropped = n_before - len(eval_pdf)
    if n_dropped:
        print(f"dropped {n_dropped:,} rows with NaN features "
              f"({n_dropped/n_before:.1%}) before scoring")

    eval_clipped = apply_clip(eval_pdf, bounds)
    eval_pdf["anomaly_score"] = -model.score_samples(eval_clipped[cols])
    return eval_pdf



def pr_auc_by_type(eval_pdf):
    """PR-AUC per anomaly type: each type's positives vs ALL negatives.

    Never a single global PR-AUC — level shifts are 84% of positives and would
    dominate. Each type is scored against the same shared negative pool.
    """
    negatives = eval_pdf[eval_pdf["label"] == 0]
    results = {}
    for atype in ["spike", "drop", "contextual", "level_shift"]:
        pos = eval_pdf[(eval_pdf["label"] == 1) & (eval_pdf["anomaly_type"] == atype)]
        subset = pd.concat([pos, negatives])
        ap = average_precision_score(subset["label"], subset["anomaly_score"])
        results[atype] = (ap, len(pos))
    return results