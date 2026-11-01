import pandas as pd
import numpy as np
import joblib
import json

from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from src.config import DATA_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, MODELS_DIR



FEATURE_COLS = ["sales_zscore", "yoy_ratio", "wow_delta", "snap_days", "cat_code", "month"]

def load_training_matrix(spark):
    """Load the feature matrix, keep fully-valid rows, bring to pandas.

    Unsupervised: the model trains on ALL dense series (not just the injection
    pool). Rows lacking YoY (first 52 weeks) or rolling stats are dropped — no
    imputation of a 'last year' that doesn't exist.
    """
    from pyspark.sql import functions as F
    
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



def choose_threshold(eval_scored, fp_budget=0.05):
    """Set the operating threshold from a false-positive budget.

    The threshold is the score quantile of the NEGATIVES at (1 - fp_budget):
    flagging above it means ~fp_budget of normal weeks are flagged (the alert
    budget). Returned with the implied alert volume and recall per type.
    """
    neg = eval_scored[eval_scored["label"] == 0]
    threshold = neg["anomaly_score"].quantile(1 - fp_budget)

    # Recall per type at this threshold (all severities pooled).
    recalls = {}
    for atype in ["spike", "drop", "contextual", "level_shift"]:
        pos = eval_scored[(eval_scored["label"] == 1) &
                          (eval_scored["anomaly_type"] == atype)]
        recalls[atype] = float((pos["anomaly_score"] >= threshold).mean())

    return {"threshold": float(threshold),
            "fp_budget": fp_budget,
            "recall_by_type": recalls}



def compute_recall_metrics(eval_scored, fp_budgets=(0.05, 0.01)):
    """Flatten recall@FP per type and severity into MLflow-loggable form.

    Returns (flat_scalars, severity_curves):
      - flat_scalars: {metric_name: value} for single-point metrics (contextual)
      - severity_curves: {metric_name: [(step, value), ...]} for k-dependent types,
        so MLflow renders recall-vs-severity curves (step = int(k*10)).
    """
    neg = eval_scored[eval_scored["label"] == 0]
    thresholds = {f"fp{int(b*100)}": neg["anomaly_score"].quantile(1 - b)
                  for b in fp_budgets}

    flat, curves = {}, {}
    for atype in ["spike", "drop", "contextual", "level_shift"]:
        pt = eval_scored[(eval_scored.label == 1) & (eval_scored.anomaly_type == atype)]
        for fp_name, thr in thresholds.items():
            if pt["severity_k"].notna().any():
                pts = []
                for k in sorted(pt["severity_k"].dropna().unique()):
                    p = pt[pt["severity_k"] == k]
                    pts.append((int(k * 10), float((p.anomaly_score >= thr).mean())))
                curves[f"recall_{atype}_{fp_name}"] = pts
            else:
                flat[f"recall_{atype}_{fp_name}"] = float((pt.anomaly_score >= thr).mean())
    return flat, curves



def log_run(model, bounds, threshold_info, eval_scored,
            if_params=IF_PARAMS, clip_q=CLIP_Q):
    """Log the full IF run to MLflow: params, metrics, model, and the serving
    artifacts (clip bounds + threshold) needed to reproduce scoring."""
    
    import mlflow
    import mlflow.sklearn
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        # --- params ---
        mlflow.log_params(if_params)
        mlflow.log_param("clip_quantile", clip_q)
        mlflow.log_param("fp_budget", threshold_info["fp_budget"])
        mlflow.log_param("threshold", round(threshold_info["threshold"], 6))

        # --- metrics ---
        flat, curves = compute_recall_metrics(eval_scored)
        for name, val in flat.items():
            mlflow.log_metric(name, val)
        for name, pts in curves.items():               # recall-vs-severity curves
            for step, val in pts:
                mlflow.log_metric(name, val, step=step)

        # --- model ---
        mlflow.sklearn.log_model(model, "model")

        # --- serving artifacts: clip bounds + threshold travel with the model ---
        import json, tempfile, os
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "clip_bounds.json"), "w") as f:
                json.dump(bounds, f, indent=2)
            with open(os.path.join(d, "threshold.json"), "w") as f:
                json.dump({"threshold": threshold_info["threshold"],
                           "fp_budget": threshold_info["fp_budget"]}, f, indent=2)
            mlflow.log_artifacts(d, artifact_path="serving")

        return run.info.run_id



def export_model(model, bounds, threshold_info, name="isolation_forest"):
    """Export a standalone serving bundle (no MLflow/SQLite needed at runtime).

    Writes model + clip bounds + threshold + feature order into models/<name>/.
    This is what Phase 5 (FastAPI/Cloud Run) loads. Mirrors the P1 standalone
    export pattern.
    """
    out = MODELS_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out / "model.pkl")
    with open(out / "clip_bounds.json", "w") as f:
        json.dump(bounds, f, indent=2)
    with open(out / "threshold.json", "w") as f:
        json.dump({"threshold": threshold_info["threshold"],
                   "fp_budget": threshold_info["fp_budget"]}, f, indent=2)
    with open(out / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    return out


def load_bundle(name="isolation_forest"):
    """Load the standalone serving bundle. No MLflow dependency."""
    d = MODELS_DIR / name
    model = joblib.load(d / "model.pkl")
    with open(d / "clip_bounds.json") as f:
        bounds = {k: tuple(v) for k, v in json.load(f).items()}
    with open(d / "threshold.json") as f:
        thr = json.load(f)
    with open(d / "feature_cols.json") as f:
        cols = json.load(f)
    return model, bounds, thr, cols


def score_rows(df, model, bounds, threshold, cols):
    """Score a pandas frame of feature rows (serving path).

    Applies the training clip bounds, scores, flags at the operating threshold.
    """
    X = df.copy()
    for c, (lo, hi) in bounds.items():
        X[c] = X[c].clip(lower=lo, upper=hi)
    scores = -model.score_samples(X[cols])
    return pd.DataFrame({"anomaly_score": scores, "is_anomaly": scores >= threshold})