"""Phase 4 — Isolation Forest (MVP).

Tabular. The temporal signal comes EXCLUSIVELY from upstream feature engineering
(rolling z-scores, lags, deviations) — the IF never sees a sequence.

Threshold justified by the precision/recall trade-off translated into BUSINESS
COST (missed stockout != false alarm), not by a `contamination` set at random.
"""
from sklearn.ensemble import IsolationForest


def train_isolation_forest(X, random_state: int = 42) -> IsolationForest:
    """Train an unsupervised IF on the feature matrix (no absolute levels)."""
    raise NotImplementedError("Phase 4")


def choose_threshold(scores, cost_missed: float, cost_false_alarm: float):
    """Choose the flagging threshold by minimizing expected business cost."""
    raise NotImplementedError("Phase 4")
