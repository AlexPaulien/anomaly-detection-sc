"""Phase 2 — Feature engineering CONTEXTUEL.

Coeur du projet en signal "ventes brutes" : la baseline n'absorbe plus la
saisonnalité, donc la charge de "ce qui est attendu" migre ICI.

REGLE D'OR : ne jamais donner le niveau brut absolu à l'Isolation Forest
(sinon les gros magasins ressortent "anormaux"). On nourrit des ECARTS.
"""
import pandas as pd


def add_local_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Déviation locale : z-score glissant, delta WoW.

    Le z-score glissant DOIT utiliser une moyenne/écart-type décalés (shift)
    pour ne pas fuiter la valeur courante dans sa propre statistique.
    """
    raise NotImplementedError("Phase 2")


def add_seasonal_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Déviation saisonnière : écart à la même semaine N-1, écart à une
    moyenne saisonnière. Permet de distinguer 'haut PARCE QUE décembre' de
    'haut anormalement'."""
    raise NotImplementedError("Phase 2")


def add_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flags calendaires : holiday, SNAP, event."""
    raise NotImplementedError("Phase 2")


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble la matrice de features. AUCUNE colonne de niveau absolu côté IF."""
    raise NotImplementedError("Phase 2")
