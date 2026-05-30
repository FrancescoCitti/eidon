"""RetinaFace detector wrapper.

Loads the det_500m ONNX model from the buffalo_s InsightFace pack and exposes
a clean typed interface.  InsightFace handles anchor generation, NMS, and
landmark decoding internally; this module owns the boundary between InsightFace
objects and our Detection dataclass.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from insightface.app import FaceAnalysis

from eidon.types import Detection

logger = logging.getLogger(__name__)


class RetinaFaceDetector:
    """Wraps InsightFace's RetinaFace detector with a typed, testable interface.

    Args:
        model_dir: Directory that contains the pack sub-folder
                   (e.g. ``Path("models")`` → models/buffalo_s/).
        pack:      InsightFace model pack name (default ``buffalo_s``).
        det_size:  Detector input resolution as (width, height).
        providers: ONNX Runtime execution providers in priority order.
    """

    def __init__(
        self,
        model_dir: Path,
        pack: str = "buffalo_s",
        det_size: tuple[int, int] = (640, 640),
        providers: list[str] | None = None,
    ) -> None:
        self._det_size = det_size
        self._providers = providers or ["CPUExecutionProvider"]
        self._pack = pack
        self._app: FaceAnalysis = self._load(model_dir, pack)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray, threshold: float = 0.5) -> list[Detection]:
        """Detect all faces in *image* above *threshold* confidence.

        Args:
            image:     BGR image of shape (H, W, 3), dtype uint8.
            threshold: Minimum detection score to include a face.

        Returns:
            List of :class:`~eidon.types.Detection`, sorted by score descending.

        Raises:
            ValueError: if the image has an unexpected shape.
        """
        self._validate_image(image)
        raw_faces = self._app.get(image)

        results: list[Detection] = []
        for face in raw_faces:
            if float(face.det_score) < threshold:
                continue
            results.append(
                Detection(
                    bbox=np.array(face.bbox, dtype=np.float32),
                    landmarks=np.array(face.kps, dtype=np.float32),
                    score=float(face.det_score),
                )
            )

        results.sort(key=lambda d: d.score, reverse=True)
        return results

    def detect_largest(
        self, image: np.ndarray, threshold: float = 0.5
    ) -> Detection | None:
        """Return the largest (by bounding-box area) face above *threshold*.

        Returns ``None`` if no face is found.
        """
        detections = self.detect(image, threshold)
        if not detections:
            return None
        return max(detections, key=lambda d: d.area)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, model_dir: Path, pack: str) -> FaceAnalysis:
        # InsightFace resolves pack files at {root}/models/{pack}/.
        # resolve() makes the path absolute so it works regardless of cwd.
        root = str(model_dir.resolve().parent)

        app = FaceAnalysis(
            name=pack,
            root=root,
            allowed_modules=["detection"],
            providers=self._providers,
        )
        app.prepare(ctx_id=0, det_size=self._det_size)

        logger.info(
            "RetinaFace detector ready",
            extra={"pack": pack, "det_size": self._det_size, "providers": self._providers},
        )
        return app

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected BGR image with shape (H, W, 3), got {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got dtype {image.dtype}")
