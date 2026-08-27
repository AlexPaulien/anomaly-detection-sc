"""Phase 5 — FastAPI scoring API (MVP endpoint)."""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Anomaly Detection SC", version="0.1.0")


class ScoringRequest(BaseModel):
    sku: str
    store: str
    week: str
    # + contextual features expected by the model


class ScoringResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoringResponse)
def score(req: ScoringRequest):
    """TODO Phase 5: load the model from models/ and score."""
    raise NotImplementedError("Phase 5")
