"""Phase 2 — CONTEXTUAL feature engineering.

Core of the project under the "raw sales" signal: the baseline no longer absorbs
seasonality, so the burden of "what is expected" migrates HERE.

GOLDEN RULE: never feed the raw absolute level to the Isolation Forest
(otherwise large stores show up as "anomalous"). We feed DEVIATIONS.
"""
import pandas as pd


def add_local_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Local deviation: rolling z-score, WoW delta.

    The rolling z-score MUST use shifted mean/std so the current value does not
    leak into its own statistic.
    """
    raise NotImplementedError("Phase 2")


def add_seasonal_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Seasonal deviation: gap to the same week last year, gap to a seasonal
    average. Lets the model tell 'high BECAUSE December' from 'abnormally high'."""
    raise NotImplementedError("Phase 2")


def add_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar flags: holiday, SNAP, event."""
    raise NotImplementedError("Phase 2")


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the feature matrix. NO absolute-level column on the IF side."""
    raise NotImplementedError("Phase 2")
