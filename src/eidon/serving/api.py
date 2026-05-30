"""Eidon REST API.

Endpoints
---------
GET  /health                     — liveness + gallery stats
POST /recognize                  — detect and identify faces in an uploaded image
POST /enroll                     — add one face embedding to the gallery
GET  /identities                 — list enrolled identities
DELETE /identities/{name}        — remove an identity from the gallery

Authentication
--------------
Set EIDON_API_KEY in the environment (or .env) to enable API-key auth.
When the env var is absent, all endpoints are open (development mode).
Pass the key via the ``X-API-Key`` request header.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import cast

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader

from eidon import __version__
from eidon.config import settings
from eidon.detection import align_face
from eidon.pipeline import EidonPipeline
from eidon.serving.schemas import (
    BoundingBox,
    DeleteResponse,
    EnrollResponse,
    FaceResult,
    HealthResponse,
    IdentitiesResponse,
    IdentityInfo,
    RecognizeResponse,
)
from eidon.types import RecognitionResult

logger = logging.getLogger(__name__)

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def _verify_api_key(key: str | None = Security(_API_KEY_HEADER)) -> None:
    """Raise 403 when an API key is configured and the request key doesn't match."""
    required = settings.api_key
    if required is None:
        return  # auth disabled
    if key != required:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key header")


# ---------------------------------------------------------------------------
# Pipeline dependency
# ---------------------------------------------------------------------------


def get_pipeline() -> EidonPipeline:
    """Return the loaded pipeline from app state.

    Override this dependency in tests to inject a mock without triggering
    model loading via the lifespan.
    """
    return cast(EidonPipeline, app.state.pipeline)


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Load models on startup; skip if pipeline was already injected (test mode)."""
    if not hasattr(application.state, "pipeline"):
        logger.info("Loading pipeline from settings…")
        application.state.pipeline = EidonPipeline.from_settings()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Eidon",
    description="Real-time face recognition — ArcFace embeddings, RetinaFace detection.",
    version=__version__,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode image — send a valid JPEG, PNG, or BMP",
        )
    return img


def _to_face_result(result: RecognitionResult) -> FaceResult:
    x1, y1, x2, y2 = result.detection.bbox.tolist()
    return FaceResult(
        identity=result.identity,
        similarity=round(float(result.similarity), 4),
        matched=result.matched,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        detection_score=round(float(result.detection.score), 4),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(pipeline: EidonPipeline = Depends(get_pipeline)) -> HealthResponse:
    """Service liveness check — no auth required."""
    g = pipeline.gallery
    return HealthResponse(
        status="ok",
        gallery_identities=len(g),
        gallery_embeddings=g.size,
        model_pack=settings.det_model_pack,
        version=__version__,
    )


@app.post("/recognize", response_model=RecognizeResponse, tags=["recognition"])
def recognize(
    file: UploadFile = File(..., description="Face image (JPEG / PNG / BMP)"),
    pipeline: EidonPipeline = Depends(get_pipeline),
    _auth: None = Depends(_verify_api_key),
) -> RecognizeResponse:
    """Detect and identify all faces in the uploaded image."""
    t0 = time.perf_counter()
    image = _decode_image(file.file.read())
    results = pipeline.recognize(image)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    logger.info(
        "Recognize request",
        extra={"faces": len(results), "processing_ms": elapsed_ms},
    )
    return RecognizeResponse(
        faces=[_to_face_result(r) for r in results],
        count=len(results),
        processing_time_ms=elapsed_ms,
    )


@app.post("/enroll", response_model=EnrollResponse, tags=["gallery"])
def enroll(
    file: UploadFile = File(..., description="Face image containing exactly one person"),
    identity: str = Form(..., description="Name to enroll this face under"),
    pipeline: EidonPipeline = Depends(get_pipeline),
    _auth: None = Depends(_verify_api_key),
) -> EnrollResponse:
    """Detect the largest face in the image and add it to the gallery."""
    image = _decode_image(file.file.read())

    detection = pipeline.detector.detect_largest(image, threshold=0.5)
    if detection is None:
        raise HTTPException(
            status_code=422,
            detail="No face detected in the image — use a clear, frontal face photo",
        )

    crop = align_face(image, detection.landmarks)
    embedding = pipeline.embedder.embed(crop)

    gallery = pipeline.gallery
    gallery.add(identity, embedding)
    gallery.save(settings.gallery_path)
    pipeline.update_gallery(gallery)

    count = int(gallery.embeddings_for(identity).shape[0])
    logger.info("Enrolled", extra={"identity": identity, "total_embeddings": count})

    return EnrollResponse(
        identity=identity,
        embeddings_count=count,
        message=f"Successfully enrolled '{identity}' ({count} embedding(s) total)",
    )


@app.get("/identities", response_model=IdentitiesResponse, tags=["gallery"])
def list_identities(
    pipeline: EidonPipeline = Depends(get_pipeline),
    _auth: None = Depends(_verify_api_key),
) -> IdentitiesResponse:
    """List all enrolled identities and their embedding counts."""
    gallery = pipeline.gallery
    items = [
        IdentityInfo(name=name, embeddings=int(gallery.embeddings_for(name).shape[0]))
        for name in gallery.identities
    ]
    return IdentitiesResponse(identities=items, total=len(items))


@app.delete("/identities/{name}", response_model=DeleteResponse, tags=["gallery"])
def delete_identity(
    name: str,
    pipeline: EidonPipeline = Depends(get_pipeline),
    _auth: None = Depends(_verify_api_key),
) -> DeleteResponse:
    """Remove an identity and all its embeddings from the gallery."""
    gallery = pipeline.gallery
    if name not in gallery:
        raise HTTPException(status_code=404, detail=f"Identity '{name}' not found")

    gallery.remove(name)
    gallery.save(settings.gallery_path)
    pipeline.update_gallery(gallery)

    logger.info("Identity deleted", extra={"identity": name})
    return DeleteResponse(identity=name, message=f"'{name}' removed from gallery")
