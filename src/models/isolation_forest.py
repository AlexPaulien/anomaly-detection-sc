"""Phase 4 — Isolation Forest (MVP).

Tabulaire. Le signal temporel vient EXCLUSIVEMENT du feature engineering amont
(z-scores glissants, lags, déviations) — l'IF ne voit pas de séquence.

Seuil justifié par le trade-off precision/recall traduit en COUT METIER
(rupture ratée != fausse alerte), pas par un `contamination` posé au hasard.
"""
from sklearn.ensemble import IsolationForest


def train_isolation_forest(X, random_state: int = 42) -> IsolationForest:
    """Entraîne un IF unsupervised sur la matrice de features (sans niveaux absolus)."""
    raise NotImplementedError("Phase 4")


def choose_threshold(scores, cost_missed: float, cost_false_alarm: float):
    """Choisit le seuil de flag en minimisant le coût métier attendu."""
    raise NotImplementedError("Phase 4")
