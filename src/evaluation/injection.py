"""Phase 3 — Injection synthétique d'anomalies (LA brique qui porte l'éval).

Principe : les modèles s'entraînent en unsupervised sur toute la donnée ; on
EVALUE sur des anomalies qu'on a injectées, donc labellisées. C'est ce qui
donne un precision/recall dans un cadre non-supervisé (analogue du holdout
honnête du P1).

Pré-requis : ne travailler que sur le pool de séries DENSES propres. La
magnitude est TOUJOURS relative à la variance locale, jamais absolue.
"""
from dataclasses import dataclass


@dataclass
class InjectionConfig:
    anomaly_type: str   # "spike" | "drop" | "level_shift" | "variance_change" | "contextual"
    severity_k: float   # magnitude en k * sigma_local
    n_injections: int


def inject_anomalies(series_df, config: InjectionConfig):
    """Injecte des anomalies typées dans des séries propres et renvoie
    (série modifiée, labels).

    Retour attendu : chaque (série, semaine) touché -> label 1, reste -> 0.

    Types :
      - spike / drop : choc ponctuel de +/- k*sigma
      - level_shift  : rupture de niveau soutenue
      - variance_change : changement de variance
      - contextual   : valeur normale en absolu mais hors-saison
    """
    raise NotImplementedError("Phase 3 — première brique à coder après l'EDA")
