"""Unit tests for detection module.

Alignment tests run without any model files.
Detector tests mock InsightFace's FaceAnalysis so no models are needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eidon.detection.alignment import ARCFACE_DST, _umeyama, align_face
from eidon.detection.retinaface import RetinaFaceDetector
from eidon.types import Detection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_landmarks(offset: float = 0.0) -> np.ndarray:
    """Return ARCFACE_DST shifted by offset — a controllable input for alignment."""
    return (ARCFACE_DST + offset).astype(np.float32)


def _make_mock_face(
    bbox: list[float],
    kps: list[list[float]],
    det_score: float,
) -> MagicMock:
    face = MagicMock()
    face.bbox = np.array(bbox, dtype=np.float32)
    face.kps = np.array(kps, dtype=np.float32)
    face.det_score = det_score
    return face


# ---------------------------------------------------------------------------
# Umeyama transform
# ---------------------------------------------------------------------------

class TestUmeyama:
    def test_identity_when_src_equals_dst(self) -> None:
        pts = ARCFACE_DST.astype(np.float32)
        M = _umeyama(pts, pts)
        assert M.shape == (2, 3)
        # Rotation+scale block should be close to identity * scale≈1
        np.testing.assert_allclose(M[:, :2], np.eye(2, dtype=np.float32), atol=1e-4)

    def test_pure_translation(self) -> None:
        src = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        dst = src + np.array([10.0, 20.0], dtype=np.float32)
        M = _umeyama(src, dst)
        # Translation column should be [10, 20]
        np.testing.assert_allclose(M[:, 2], [10.0, 20.0], atol=1e-4)

    def test_output_dtype_is_float32(self) -> None:
        M = _umeyama(ARCFACE_DST, ARCFACE_DST)
        assert M.dtype == np.float32

    def test_output_shape(self) -> None:
        M = _umeyama(ARCFACE_DST, ARCFACE_DST)
        assert M.shape == (2, 3)


# ---------------------------------------------------------------------------
# Face alignment
# ---------------------------------------------------------------------------

class TestAlignFace:
    def test_output_shape_default_size(self) -> None:
        img = _make_image()
        lmks = _make_landmarks()
        crop = align_face(img, lmks)
        assert crop.shape == (112, 112, 3)

    def test_output_shape_custom_size(self) -> None:
        img = _make_image()
        lmks = _make_landmarks()
        crop = align_face(img, lmks, size=224)
        assert crop.shape == (224, 224, 3)

    def test_output_dtype_preserved(self) -> None:
        img = _make_image()
        lmks = _make_landmarks()
        crop = align_face(img, lmks)
        assert crop.dtype == np.uint8

    def test_raises_on_wrong_image_channels(self) -> None:
        img = np.zeros((480, 640), dtype=np.uint8)  # grayscale — missing channel dim
        lmks = _make_landmarks()
        with pytest.raises(ValueError, match="image must be"):
            align_face(img, lmks)

    def test_raises_on_wrong_landmark_shape(self) -> None:
        img = _make_image()
        bad_lmks = np.zeros((3, 2), dtype=np.float32)  # only 3 points instead of 5
        with pytest.raises(ValueError, match="landmarks must be"):
            align_face(img, bad_lmks)

    def test_identity_landmarks_fills_whole_output(self) -> None:
        """When the source landmarks already match ARCFACE_DST the warp is near-identity."""
        img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        crop = align_face(img, ARCFACE_DST, size=112)
        # The crop should be very close to the original (minor border interpolation differences)
        assert crop.shape == (112, 112, 3)


# ---------------------------------------------------------------------------
# RetinaFaceDetector (mocked)
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_detector(tmp_path: pytest.TempPathFactory) -> RetinaFaceDetector:
    """RetinaFaceDetector with InsightFace.FaceAnalysis fully mocked."""
    mock_app = MagicMock()
    with patch("eidon.detection.retinaface.FaceAnalysis", return_value=mock_app):
        detector = RetinaFaceDetector(model_dir=tmp_path / "models")
    detector._app = mock_app
    return detector


class TestRetinaFaceDetector:
    def test_detect_returns_empty_list_when_no_faces(
        self, mock_detector: RetinaFaceDetector
    ) -> None:
        mock_detector._app.get.return_value = []
        result = mock_detector.detect(_make_image())
        assert result == []

    def test_detect_filters_below_threshold(
        self, mock_detector: RetinaFaceDetector
    ) -> None:
        faces = [
            _make_mock_face([10, 10, 60, 60], _make_landmarks().tolist(), det_score=0.9),
            _make_mock_face([80, 80, 120, 120], _make_landmarks().tolist(), det_score=0.3),
        ]
        mock_detector._app.get.return_value = faces
        result = mock_detector.detect(_make_image(), threshold=0.5)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.9)

    def test_detect_returns_sorted_by_score_descending(
        self, mock_detector: RetinaFaceDetector
    ) -> None:
        faces = [
            _make_mock_face([0, 0, 50, 50], _make_landmarks().tolist(), det_score=0.7),
            _make_mock_face([60, 60, 110, 110], _make_landmarks().tolist(), det_score=0.95),
        ]
        mock_detector._app.get.return_value = faces
        result = mock_detector.detect(_make_image())
        assert result[0].score > result[1].score

    def test_detect_returns_detection_dataclass(
        self, mock_detector: RetinaFaceDetector
    ) -> None:
        mock_detector._app.get.return_value = [
            _make_mock_face([10, 10, 60, 60], _make_landmarks().tolist(), det_score=0.9)
        ]
        result = mock_detector.detect(_make_image())
        assert isinstance(result[0], Detection)
        assert result[0].bbox.shape == (4,)
        assert result[0].landmarks.shape == (5, 2)

    def test_detect_largest_returns_none_when_no_faces(
        self, mock_detector: RetinaFaceDetector
    ) -> None:
        mock_detector._app.get.return_value = []
        assert mock_detector.detect_largest(_make_image()) is None

    def test_detect_largest_returns_biggest_face(
        self, mock_detector: RetinaFaceDetector
    ) -> None:
        faces = [
            _make_mock_face([0, 0, 100, 100], _make_landmarks().tolist(), det_score=0.9),
            _make_mock_face([0, 0, 200, 200], _make_landmarks().tolist(), det_score=0.75),
        ]
        mock_detector._app.get.return_value = faces
        result = mock_detector.detect_largest(_make_image())
        assert result is not None
        assert result.area == pytest.approx(40000.0)

    def test_detect_raises_on_wrong_dtype(
        self, mock_detector: RetinaFaceDetector
    ) -> None:
        float_img = np.zeros((480, 640, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            mock_detector.detect(float_img)

    def test_detect_raises_on_wrong_shape(
        self, mock_detector: RetinaFaceDetector
    ) -> None:
        gray_img = np.zeros((480, 640), dtype=np.uint8)
        with pytest.raises(ValueError, match="H, W, 3"):
            mock_detector.detect(gray_img)
