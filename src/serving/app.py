"""Phase 5 — API de scoring FastAPI (endpoint MVP)."""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Anomaly Detection SC", version="0.1.0")


class ScoringRequest(BaseModel):
    sku: str
    store: str
    week: str
    # + features contextuelles attendues par le modèle


class ScoringResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoringResponse)
def score(req: ScoringRequest):
    """TODO Phase 5 : charger le modèle depuis models/ et scorer."""
    raise NotImplementedError("Phase 5")
