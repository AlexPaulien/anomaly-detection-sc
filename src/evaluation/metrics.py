"""Phase 3 — Metrics harness, reusable across all models.

PR-AUC rather than ROC-AUC: extreme imbalance (rare anomalies), ROC would give
a flattering and misleading score.
"""


def pr_auc(y_true, scores) -> float:
    """Area under the precision-recall curve."""
    raise NotImplementedError("Phase 3")


def recall_vs_severity(y_true, scores, severities):
    """Recall vs severity (k) curve — THE graph that sells the project.
    E.g. '90% of shocks at 3sigma detected, 40% at 1.5sigma'."""
    raise NotImplementedError("Phase 3")


def precision_by_type(y_true, scores, anomaly_types):
    """Precision per anomaly type — reveals what the IF misses, hence the entry
    argument for the LSTM AE in V2."""
    raise NotImplementedError("Phase 3")
