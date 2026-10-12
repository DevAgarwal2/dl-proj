"""FastAPI service exposing prediction + retrieval-augmented explanation.

Run:
    make llama            # start llama-server with LFM2.5-1.2B-Instruct-Q6_K.gguf
    make ingest           # build the vector store
    make serve            # uvicorn openmed.api.server:app
"""

from __future__ import annotations

import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from openmed.api.schemas import (
    CitationModel,
    ExplanationResponse,
    ExplainRequest,
    Finding,
    HealthResponse,
    PredictResponse,
    QARequest,
)
from openmed.config import load_config
from openmed.rag.pipeline import OpenMedRAG
from openmed.utils import default_checkpoint, get_logger

LOGGER = get_logger(__name__)

RAG_CONFIG = os.environ.get("OPENMED_RAG_CONFIG", "configs/rag.yaml")
CHECKPOINT = os.environ.get("OPENMED_CHECKPOINT") or default_checkpoint()

app = FastAPI(title="OpenMed", version="0.1.0", description="Medical vision model + RAG")


@lru_cache(maxsize=1)
def get_rag() -> Optional[OpenMedRAG]:
    try:
        return OpenMedRAG.from_config(load_config(RAG_CONFIG))
    except Exception as exc:  # pragma: no cover - depends on local artifacts
        LOGGER.warning("RAG unavailable: %s", exc)
        return None


@lru_cache(maxsize=1)
def get_predictor():
    from openmed.inference import Predictor

    try:
        return Predictor(CHECKPOINT)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Predictor unavailable: %s", exc)
        return None


def _explanation_response(explanation) -> ExplanationResponse:
    return ExplanationResponse(
        finding=explanation.finding,
        probability=explanation.probability,
        text=explanation.text,
        model=explanation.model,
        citations=[
            CitationModel(
                index=c.index, doc_id=c.doc_id, title=c.title,
                source=c.source, score=c.score, snippet=c.snippet,
            )
            for c in explanation.citations
        ],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    rag = get_rag()
    predictor = get_predictor()
    size = len(rag.retriever.store) if rag else 0
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        vector_store_size=size,
        llm_available=bool(rag and rag.llm.health()),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    top_k: int = 5,
    threshold: float = 0.0,
    gradcam: bool = False,
) -> PredictResponse:
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(503, f"No model at {CHECKPOINT}. Train one first.")

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    findings = predictor.predict(image, top_k=top_k, threshold=threshold or None)

    cam_b64 = None
    if gradcam and findings:
        top = predictor.predict_with_cam(image, top_k=1)[0]
        buffer = io.BytesIO()
        top["overlay"].save(buffer, format="PNG")
        cam_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return PredictResponse(
        image_id=file.filename,
        findings=[Finding(**item) for item in findings],
        gradcam_png_base64=cam_b64,
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain(request: ExplainRequest) -> ExplanationResponse:
    rag = get_rag()
    if rag is None:
        raise HTTPException(503, "RAG pipeline unavailable; check configs/rag.yaml")
    explanation = rag.explain_finding(
        request.finding, request.probability, request.question, k=request.top_k
    )
    return _explanation_response(explanation)


@app.post("/qa", response_model=ExplanationResponse)
def qa(request: QARequest) -> ExplanationResponse:
    rag = get_rag()
    if rag is None:
        raise HTTPException(503, "RAG pipeline unavailable; check configs/rag.yaml")
    return _explanation_response(rag.answer(request.question, k=request.top_k))
