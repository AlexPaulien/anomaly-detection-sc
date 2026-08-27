"""Phase 3 — Synthetic anomaly injection (THE brick that carries evaluation).

Principle: models train unsupervised on all the data; we EVALUATE on anomalies
we injected, hence labeled. This is what yields precision/recall in an
unsupervised setting (analogous to the P1 honest holdout).

Prerequisite: work only on the clean DENSE series pool. Magnitude is ALWAYS
relative to local variance, never absolute.
"""
from dataclasses import dataclass


@dataclass
class InjectionConfig:
    anomaly_type: str   # "spike" | "drop" | "level_shift" | "variance_change" | "contextual"
    severity_k: float   # magnitude in k * local_sigma
    n_injections: int


def inject_anomalies(series_df, config: InjectionConfig):
    """Inject typed anomalies into clean series and return (modified series, labels).

    Expected output: each touched (series, week) -> label 1, everything else -> 0.

    Types:
      - spike / drop: point shock of +/- k*sigma
      - level_shift: sustained level break
      - variance_change: change in variance
      - contextual: value normal in absolute terms but out of season
    """
    raise NotImplementedError("Phase 3 — first brick to code after EDA")
