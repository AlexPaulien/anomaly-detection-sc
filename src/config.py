"""Configuration centrale du projet.

Données publiques M5 Walmart uniquement. Aucune donnée interne.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# Grille commune actée : SKU x Store x semaine
GRAIN = ["sku", "store", "week"]

# GCP — région us-central1 : données publiques/synthétiques, pas de contrainte
# GDPR, free tier US.
GCP_REGION = "us-central1"

# Régime de volume : seuil de densité pour séparer séries denses / intermittentes.
# À calibrer après l'EDA (Phase 1). L'évaluation par injection ne portera que
# sur les séries DENSES (z-score/sigma mal définis sur les intermittentes).
DENSITY_MIN_NONZERO_RATIO = 0.5  # placeholder — à ajuster via EDA

# MLflow
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "anomaly-detection-sc"
