from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pathlib import Path

from src.models.isolation_forest import load_bundle, score_rows

# Bundle loaded once at startup (never per request — loading model.pkl per call
# would be catastrophic for latency).
_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, bounds, thr, cols = load_bundle("isolation_forest")
    _state.update(model=model, bounds=bounds,
                  threshold=thr["threshold"], cols=cols)
    yield
    _state.clear()


app = FastAPI(title="Anomaly Detection SC", version="1.0.0", lifespan=lifespan)


class ScoringRequest(BaseModel):
    # Features (required — Pydantic rejects an incomplete row with 422).
    sales_zscore: float
    yoy_ratio: float
    wow_delta: float
    snap_days: int
    cat_code: int
    month: int
    # Optional identifiers, echoed back for traceability.
    sku: str | None = None
    store: str | None = None
    week: int | None = None


class ScoringResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    sku: str | None = None
    store: str | None = None
    week: int | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/score", response_model=ScoringResponse)
def score(req: ScoringRequest):
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="model not loaded")

    row = pd.DataFrame([{c: getattr(req, c) for c in _state["cols"]}])
    result = score_rows(row, _state["model"], _state["bounds"],
                        _state["threshold"], _state["cols"])

    return ScoringResponse(
        anomaly_score=float(result["anomaly_score"].iloc[0]),
        is_anomaly=bool(result["is_anomaly"].iloc[0]),
        threshold=_state["threshold"],
        sku=req.sku, store=req.store, week=req.week,
    )


VIZ_DIR = Path(__file__).resolve().parents[2] / "viz"


@app.get("/")
def index():
    return FileResponse(VIZ_DIR / "anomaly_timeline.html")


@app.get("/series_data.js")
def series_data():
    f = VIZ_DIR / "series_data.js"
    if not f.exists():
        raise HTTPException(status_code=404,
                            detail="series_data.js not generated — run viz.export_series")
    return FileResponse(f, media_type="application/javascript")