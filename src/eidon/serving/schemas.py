"""Pydantic request / response models for the Eidon REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Face bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


class FaceResult(BaseModel):
    """Recognition result for a single detected face."""

    identity: str = Field(description="Matched identity name, or 'unknown'")
    similarity: float = Field(description="Cosine similarity to nearest gallery embedding")
    matched: bool = Field(description="True when similarity >= configured threshold")
    bbox: BoundingBox = Field(description="Face bounding box in pixel coordinates")
    detection_score: float = Field(description="RetinaFace detection confidence")


class RecognizeResponse(BaseModel):
    """Response from POST /recognize."""

    faces: list[FaceResult]
    count: int = Field(description="Number of faces detected in the image")
    processing_time_ms: float = Field(description="Total server-side processing time")


class EnrollResponse(BaseModel):
    """Response from POST /enroll."""

    identity: str
    embeddings_count: int = Field(description="Total embeddings now stored for this identity")
    message: str


class IdentityInfo(BaseModel):
    """Name and embedding count for one enrolled identity."""

    name: str
    embeddings: int = Field(description="Number of embeddings stored for this identity")


class IdentitiesResponse(BaseModel):
    """Response from GET /identities."""

    identities: list[IdentityInfo]
    total: int = Field(description="Total number of enrolled identities")


class DeleteResponse(BaseModel):
    """Response from DELETE /identities/{name}."""

    identity: str
    message: str


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str = Field(description="'ok' when the service is ready")
    gallery_identities: int
    gallery_embeddings: int
    model_pack: str
    version: str
