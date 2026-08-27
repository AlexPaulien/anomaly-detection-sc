"""Phase 3 — Harnais de métriques, réutilisable pour tous les modèles.

PR-AUC et non ROC-AUC : déséquilibre extrême (anomalies rares), la ROC
donnerait un score flatteur et trompeur.
"""


def pr_auc(y_true, scores) -> float:
    """Aire sous la courbe precision-recall."""
    raise NotImplementedError("Phase 3")


def recall_vs_severity(y_true, scores, severities):
    """Courbe recall vs sévérité (k) — LE graphe qui vend le projet.
    Ex : '90% des chocs à 3sigma détectés, 40% à 1.5sigma'."""
    raise NotImplementedError("Phase 3")


def precision_by_type(y_true, scores, anomaly_types):
    """Precision par type d'anomalie — révèle ce que l'IF rate, et donc
    l'argument d'entrée du LSTM AE en V2."""
    raise NotImplementedError("Phase 3")
