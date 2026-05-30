"""ArcFace MobileNet embedding module.

Loads w600k_mbf.onnx via ONNX Runtime and produces L2-normalised 512-dim
embeddings from 112×112 aligned BGR crops (output of detection.align_face).

Preprocessing contract (matches InsightFace buffalo_s training):
  BGR uint8 → RGB float32 → subtract 127.5, divide 127.5 → range [-1, 1]
  → transpose HWC→CHW → batch dimension prepended
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)

_INPUT_SIZE: int = 112
_EMBEDDING_DIM: int = 512
_MEAN: float = 127.5
_STD: float = 127.5


class ArcFaceEmbedder:
    """Wraps the ArcFace MobileNet ONNX model with a typed, testable interface.

    Args:
        model_path: Absolute or relative path to ``w600k_mbf.onnx``.
        providers:  ONNX Runtime execution providers in priority order.
                    Defaults to CPU-only inference.
    """

    def __init__(
        self,
        model_path: Path,
        providers: list[str] | None = None,
    ) -> None:
        self._providers = providers or ["CPUExecutionProvider"]
        self._session, self._input_name, self._output_name = self._load(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Return a 512-dim L2-normalised embedding for a single face crop.

        Args:
            crop: ``(112, 112, 3)`` uint8 BGR aligned face crop.

        Returns:
            ``(512,)`` float32 L2-normalised embedding vector.

        Raises:
            ValueError: if ``crop`` has wrong shape or dtype.
        """
        self._validate_crop(crop)
        result: np.ndarray = self.embed_batch([crop])[0]
        return result

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        """Return embeddings for multiple face crops in a single inference pass.

        Args:
            crops: List of ``(112, 112, 3)`` uint8 BGR aligned face crops.

        Returns:
            ``(N, 512)`` float32 L2-normalised embedding matrix.
            Returns shape ``(0, 512)`` for an empty input list.

        Raises:
            ValueError: if any crop has wrong shape or dtype.
        """
        if not crops:
            return np.empty((0, _EMBEDDING_DIM), dtype=np.float32)

        for i, crop in enumerate(crops):
            self._validate_crop(crop, index=i)

        batch = np.stack([_preprocess(c) for c in crops])  # (N, 3, 112, 112)
        raw = self._session.run([self._output_name], {self._input_name: batch})[0]
        return _l2_normalize(raw)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self, model_path: Path) -> tuple[InferenceSession, str, str]:
        if not model_path.exists():
            raise FileNotFoundError(
                f"ArcFace model not found at '{model_path}'. "
                "Run `python scripts/download_models.py` first."
            )

        session = InferenceSession(str(model_path), providers=self._providers)
        input_name: str = session.get_inputs()[0].name
        output_name: str = session.get_outputs()[0].name

        logger.info(
            "ArcFace embedder ready",
            extra={
                "model": model_path.name,
                "providers": self._providers,
                "input_name": input_name,
                "output_name": output_name,
            },
        )
        return session, input_name, output_name

    @staticmethod
    def _validate_crop(crop: np.ndarray, index: int | None = None) -> None:
        label = f"crops[{index}]" if index is not None else "crop"
        expected = (_INPUT_SIZE, _INPUT_SIZE, 3)
        if crop.shape != expected:
            raise ValueError(f"{label} must be {expected}, got {crop.shape}")
        if crop.dtype != np.uint8:
            raise ValueError(f"{label} must be uint8, got dtype {crop.dtype}")


# ------------------------------------------------------------------
# Module-level pure functions (independently testable)
# ------------------------------------------------------------------

def _preprocess(crop: np.ndarray) -> np.ndarray:
    """Convert one BGR uint8 crop to a normalised CHW float32 tensor.

    Steps:
        1. BGR → RGB
        2. Normalise to [-1, 1]: ``(pixel - 127.5) / 127.5``
        3. HWC → CHW transpose
    """
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - _MEAN) / _STD
    return img.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalise an ``(N, D)`` float32 matrix.

    Safe against zero vectors: clips the norm denominator to 1e-10.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    out: np.ndarray = (embeddings / np.maximum(norms, 1e-10)).astype(np.float32)
    return out
