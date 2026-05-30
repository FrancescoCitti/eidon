"""Pipeline orchestrator: Detection → Alignment → Embedding → Matching.

Initialise once at startup (model loading is expensive) and reuse across frames.
"""

from __future__ import annotations

import logging

import numpy as np

from eidon.config import Settings
from eidon.config import settings as _default_settings
from eidon.detection import RetinaFaceDetector, align_face
from eidon.embedding import ArcFaceEmbedder
from eidon.matching import Gallery, Matcher
from eidon.types import RecognitionResult

logger = logging.getLogger(__name__)


class EidonPipeline:
    """Wires RetinaFace → ArcFace → Gallery matching into a single call.

    Args:
        detector:      Loaded :class:`~eidon.detection.RetinaFaceDetector`.
        embedder:      Loaded :class:`~eidon.embedding.ArcFaceEmbedder`.
        matcher:       Configured :class:`~eidon.matching.Matcher`.
        gallery:       Populated :class:`~eidon.matching.Gallery`.
                       Can be hot-swapped at any time via :meth:`update_gallery`.
        det_threshold: Minimum RetinaFace confidence to accept a face (default 0.5).
    """

    def __init__(
        self,
        detector: RetinaFaceDetector,
        embedder: ArcFaceEmbedder,
        matcher: Matcher,
        gallery: Gallery,
        det_threshold: float = 0.5,
    ) -> None:
        self._detector = detector
        self._embedder = embedder
        self._matcher = matcher
        self._gallery = gallery
        self._det_threshold = det_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize(self, image: np.ndarray) -> list[RecognitionResult]:
        """Run the full pipeline on *image*.

        Detect all faces → align each → batch-embed → match against gallery.

        Args:
            image: BGR uint8 array of shape (H, W, 3).

        Returns:
            One :class:`~eidon.types.RecognitionResult` per detected face,
            ordered by detection confidence (highest first).
            Returns an empty list when no faces are detected.

        Raises:
            ValueError: if *image* has wrong shape or dtype.
        """
        self._validate_image(image)

        detections = self._detector.detect(image, threshold=self._det_threshold)
        if not detections:
            return []

        # Align all detected faces, then embed in a single batch pass.
        crops = [align_face(image, d.landmarks) for d in detections]
        embeddings = self._embedder.embed_batch(crops)  # (N, 512)

        results: list[RecognitionResult] = []
        for detection, embedding in zip(detections, embeddings, strict=True):
            match = self._matcher.match(embedding, self._gallery)
            results.append(
                RecognitionResult(
                    detection=detection,
                    identity=match.identity,
                    similarity=match.similarity,
                    matched=match.matched,
                )
            )
            logger.debug(
                "Face recognised",
                extra={
                    "identity": match.identity,
                    "similarity": round(match.similarity, 4),
                    "matched": match.matched,
                    "det_score": round(detection.score, 4),
                },
            )

        return results

    # --- Read-only properties used by the API layer --------------------------

    @property
    def detector(self) -> RetinaFaceDetector:
        """Loaded RetinaFace detection model."""
        return self._detector

    @property
    def embedder(self) -> ArcFaceEmbedder:
        """Loaded ArcFace embedding model."""
        return self._embedder

    @property
    def gallery(self) -> Gallery:
        """Active identity gallery (can be hot-swapped via :meth:`update_gallery`)."""
        return self._gallery

    def update_gallery(self, gallery: Gallery) -> None:
        """Hot-swap the gallery without restarting the pipeline.

        The Python GIL makes attribute assignment atomic, so this is safe
        for the single-writer pattern used by the webcam and API server.
        """
        self._gallery = gallery
        logger.info(
            "Gallery updated",
            extra={"identities": len(gallery), "embeddings": gallery.size},
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, cfg: Settings | None = None) -> EidonPipeline:
        """Build a fully loaded pipeline from a :class:`~eidon.config.Settings` object.

        Loads all ONNX models from disk — call once at startup, not per-frame.

        Args:
            cfg: Settings to use.  Defaults to the module-level singleton.

        Returns:
            A ready-to-use :class:`EidonPipeline`.
        """
        if cfg is None:
            cfg = _default_settings

        logger.info("Building pipeline from settings…")

        detector = RetinaFaceDetector(
            model_dir=cfg.model_dir,
            pack=cfg.det_model_pack,
            det_size=cfg.det_size,
        )
        embedder = ArcFaceEmbedder(model_path=cfg.rec_model_path)
        matcher = Matcher(threshold=cfg.similarity_threshold)
        gallery = Gallery.load_or_empty(cfg.gallery_path)

        logger.info(
            "Pipeline ready",
            extra={"gallery_identities": len(gallery), "gallery_embeddings": gallery.size},
        )
        return cls(
            detector=detector,
            embedder=embedder,
            matcher=matcher,
            gallery=gallery,
            det_threshold=0.5,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected BGR image (H, W, 3), got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got dtype {image.dtype}")
