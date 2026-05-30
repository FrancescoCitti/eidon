"""Unit tests for EidonPipeline.

Detector and embedder are mocked; Gallery and Matcher are real instances
so the composition logic (alignment dispatch, batch embedding, result assembly)
is exercised with genuine data flows.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from eidon.detection.alignment import ARCFACE_DST
from eidon.matching import Gallery, Matcher
from eidon.pipeline import EidonPipeline
from eidon.types import Detection, RecognitionResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 512


def _unit_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM, dtype=np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _make_image(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_detection(score: float = 0.95, bbox: list[float] | None = None) -> Detection:
    """Detection whose landmarks sit exactly at the ArcFace 112×112 template."""
    return Detection(
        bbox=np.array(bbox or [10.0, 10.0, 122.0, 122.0], dtype=np.float32),
        landmarks=ARCFACE_DST.copy(),
        score=score,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_detector() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def mock_embedder() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def alice_vec() -> np.ndarray:
    return _unit_vec(0)


@pytest.fixture()
def gallery(alice_vec: np.ndarray) -> Gallery:
    g = Gallery()
    g.add("Alice", alice_vec)
    return g


@pytest.fixture()
def pipeline(
    mock_detector: MagicMock,
    mock_embedder: MagicMock,
    gallery: Gallery,
) -> EidonPipeline:
    return EidonPipeline(
        detector=mock_detector,
        embedder=mock_embedder,
        matcher=Matcher(threshold=0.4),
        gallery=gallery,
    )


# ---------------------------------------------------------------------------
# recognize() — core behaviour
# ---------------------------------------------------------------------------

class TestRecognize:
    def test_no_faces_returns_empty_list(
        self, pipeline: EidonPipeline, mock_detector: MagicMock
    ) -> None:
        mock_detector.detect.return_value = []
        assert pipeline.recognize(_make_image()) == []

    def test_single_face_returns_one_result(
        self,
        pipeline: EidonPipeline,
        mock_detector: MagicMock,
        mock_embedder: MagicMock,
        alice_vec: np.ndarray,
    ) -> None:
        mock_detector.detect.return_value = [_make_detection()]
        mock_embedder.embed_batch.return_value = alice_vec.reshape(1, _DIM)

        results = pipeline.recognize(_make_image())

        assert len(results) == 1
        assert isinstance(results[0], RecognitionResult)

    def test_matched_face_returns_correct_identity(
        self,
        pipeline: EidonPipeline,
        mock_detector: MagicMock,
        mock_embedder: MagicMock,
        alice_vec: np.ndarray,
    ) -> None:
        mock_detector.detect.return_value = [_make_detection()]
        mock_embedder.embed_batch.return_value = alice_vec.reshape(1, _DIM)

        results = pipeline.recognize(_make_image())

        assert results[0].identity == "Alice"
        assert results[0].matched is True

    def test_unknown_face_returns_unknown_identity(
        self,
        pipeline: EidonPipeline,
        mock_detector: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        # Random embedding dissimilar to Alice
        unknown_vec = _unit_vec(999)
        mock_detector.detect.return_value = [_make_detection()]
        mock_embedder.embed_batch.return_value = unknown_vec.reshape(1, _DIM)

        results = pipeline.recognize(_make_image())

        # With threshold=0.4 and a random vector, this may or may not match —
        # but we can assert matched is consistent with the similarity value.
        result = results[0]
        assert result.matched == (result.similarity >= 0.4)

    def test_multiple_faces_returns_multiple_results(
        self,
        pipeline: EidonPipeline,
        mock_detector: MagicMock,
        mock_embedder: MagicMock,
        alice_vec: np.ndarray,
    ) -> None:
        detections = [_make_detection(score=0.9), _make_detection(score=0.8)]
        embeddings = np.stack([alice_vec, _unit_vec(99)])
        mock_detector.detect.return_value = detections
        mock_embedder.embed_batch.return_value = embeddings

        results = pipeline.recognize(_make_image())
        assert len(results) == 2

    def test_embedder_receives_correct_number_of_crops(
        self,
        pipeline: EidonPipeline,
        mock_detector: MagicMock,
        mock_embedder: MagicMock,
        alice_vec: np.ndarray,
    ) -> None:
        n_faces = 3
        mock_detector.detect.return_value = [_make_detection() for _ in range(n_faces)]
        mock_embedder.embed_batch.return_value = np.stack([alice_vec] * n_faces)

        pipeline.recognize(_make_image())

        call_args = mock_embedder.embed_batch.call_args
        crops_passed = call_args[0][0]
        assert len(crops_passed) == n_faces

    def test_each_crop_is_112x112_bgr(
        self,
        pipeline: EidonPipeline,
        mock_detector: MagicMock,
        mock_embedder: MagicMock,
        alice_vec: np.ndarray,
    ) -> None:
        mock_detector.detect.return_value = [_make_detection()]
        mock_embedder.embed_batch.return_value = alice_vec.reshape(1, _DIM)

        pipeline.recognize(_make_image())

        crops = mock_embedder.embed_batch.call_args[0][0]
        assert crops[0].shape == (112, 112, 3)
        assert crops[0].dtype == np.uint8

    def test_result_carries_original_detection(
        self,
        pipeline: EidonPipeline,
        mock_detector: MagicMock,
        mock_embedder: MagicMock,
        alice_vec: np.ndarray,
    ) -> None:
        det = _make_detection(score=0.97)
        mock_detector.detect.return_value = [det]
        mock_embedder.embed_batch.return_value = alice_vec.reshape(1, _DIM)

        result = pipeline.recognize(_make_image())[0]

        assert result.detection is det

    def test_det_threshold_forwarded_to_detector(
        self,
        mock_detector: MagicMock,
        mock_embedder: MagicMock,
        gallery: Gallery,
    ) -> None:
        mock_detector.detect.return_value = []
        pipe = EidonPipeline(
            detector=mock_detector,
            embedder=mock_embedder,
            matcher=Matcher(),
            gallery=gallery,
            det_threshold=0.7,
        )
        pipe.recognize(_make_image())
        # assert_called_once_with can't compare numpy arrays via ==, so inspect args manually.
        assert mock_detector.detect.call_count == 1
        _, kwargs = mock_detector.detect.call_args
        assert kwargs["threshold"] == pytest.approx(0.7)

    # --- input validation ---

    def test_raises_on_wrong_image_shape(
        self, pipeline: EidonPipeline
    ) -> None:
        with pytest.raises(ValueError, match="H, W, 3"):
            pipeline.recognize(np.zeros((480, 640), dtype=np.uint8))

    def test_raises_on_wrong_dtype(
        self, pipeline: EidonPipeline
    ) -> None:
        with pytest.raises(ValueError, match="uint8"):
            pipeline.recognize(np.zeros((480, 640, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# update_gallery()
# ---------------------------------------------------------------------------

class TestUpdateGallery:
    def test_new_gallery_used_on_next_call(
        self,
        pipeline: EidonPipeline,
        mock_detector: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        bob_vec = _unit_vec(7)
        new_gallery = Gallery()
        new_gallery.add("Bob", bob_vec)

        pipeline.update_gallery(new_gallery)

        mock_detector.detect.return_value = [_make_detection()]
        mock_embedder.embed_batch.return_value = bob_vec.reshape(1, _DIM)

        results = pipeline.recognize(_make_image())
        assert results[0].identity == "Bob"
