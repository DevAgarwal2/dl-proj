"""Request / response models for the inference API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Finding(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    image_id: Optional[str] = None
    findings: List[Finding]
    gradcam_png_base64: Optional[str] = None


class ExplainRequest(BaseModel):
    finding: str
    probability: float = Field(ge=0.0, le=1.0)
    question: Optional[str] = None
    top_k: Optional[int] = None


class CitationModel(BaseModel):
    index: int
    doc_id: str
    title: str
    source: str
    score: float
    snippet: str


class ExplanationResponse(BaseModel):
    finding: Optional[str] = None
    probability: Optional[float] = None
    text: str
    citations: List[CitationModel]
    model: Optional[str] = None


class QARequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vector_store_size: int
    llm_available: bool
