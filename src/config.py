"""Central project configuration.

Public M5 Walmart dataset only. No internal/company data.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# Agreed common grid: SKU x Store x week
GRAIN = ["sku", "store", "week"]

# GCP — us-central1 region: public/synthetic data, no GDPR localization
# constraint, US free tier.
GCP_REGION = "us-central1"

# Volume regime: density threshold to split dense / intermittent series.
# To be calibrated after EDA (Phase 1). Injection-based evaluation will only
# cover DENSE series (z-score/sigma are ill-defined on intermittent ones).
DENSITY_MIN_NONZERO_RATIO = 0.8 # was 0.5 placeholder; set from EDA-3
MIN_LIFE_WEEKS = 52 # ensures YoY seasonal feature is computable

# MLflow
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "anomaly-detection-sc"
