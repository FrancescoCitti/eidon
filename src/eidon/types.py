"""Shared data types used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    """A single face detected in an image.

    Landmarks order: left_eye, right_eye, nose_tip, left_mouth, right_mouth.
    """

    bbox: np.ndarray  # float32 (4,)   — [x1, y1, x2, y2] in pixel coords
    landmarks: np.ndarray  # float32 (5, 2) — five facial keypoints
    score: float  # detection confidence in [0, 1]

    @property
    def area(self) -> float:
        """Bounding-box area in pixels²."""
        x1, y1, x2, y2 = self.bbox
        return float((x2 - x1) * (y2 - y1))

    @property
    def width(self) -> int:
        """Bounding-box width in pixels."""
        return int(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        """Bounding-box height in pixels."""
        return int(self.bbox[3] - self.bbox[1])


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Output of the matcher for one embedding query (no detection context)."""

    identity: str  # best-match identity name, or "unknown" when below threshold
    similarity: float  # cosine similarity to the nearest gallery embedding
    matched: bool  # True when similarity >= configured threshold


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    """Output of the full recognition pipeline for one detected face."""

    detection: Detection
    identity: str  # matched identity name, or "unknown"
    similarity: float  # cosine similarity to matched gallery embedding
    matched: bool  # True when similarity >= threshold
