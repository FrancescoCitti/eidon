"""Integration tests for EidonPipeline.

Detector and embedder are mocked (no ONNX models required).
Gallery and Matcher are real instances, validating the full data flow:
  mock detection → real alignment → mock embedding → real matching → result assembly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from eidon.detection.alignment import ARCFACE_DST
from eidon.matching import Gallery, Matcher
from eidon.pipeline import EidonPipeline
from eidon.types import Detection

_DIM = 512


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM, dtype=np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _detection_at(x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> Detection:
    """Detection with standard ArcFace landmarks — alignment produces a valid 112×112 crop."""
    return Detection(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        landmarks=ARCFACE_DST.copy(),
        score=score,
    )


def _build_pipeline(
    detections: list[Detection],
    embeddings: np.ndarray,
    gallery: Gallery,
    threshold: float = 0.4,
) -> EidonPipeline:
    detector = MagicMock()
    detector.detect.return_value = detections

    embedder = MagicMock()
    embedder.embed_batch.return_value = embeddings

    return EidonPipeline(
        detector=detector,
        embedder=embedder,
        matcher=Matcher(threshold=threshold),
        gallery=gallery,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullDataFlow:
    def test_known_face_is_recognised(self) -> None:
        alice = _unit_vec(0)
        gallery = Gallery()
        gallery.add("Alice", alice)

        pipeline = _build_pipeline(
            detections=[_detection_at(10, 10, 122, 122)],
            embeddings=alice.reshape(1, _DIM),
            gallery=gallery,
        )

        results = pipeline.recognize(np.zeros((480, 640, 3), dtype=np.uint8))

        assert len(results) == 1
        assert results[0].identity == "Alice"
        assert results[0].matched is True
        np.testing.assert_allclose(results[0].similarity, 1.0, atol=1e-5)

    def test_unknown_face_is_rejected(self) -> None:
        alice = _unit_vec(0)
        gallery = Gallery()
        gallery.add("Alice", alice)

        # Build an orthogonal unit vector via Gram-Schmidt; guard against near-zero norm.
        rng = np.random.default_rng(42)
        unknown = rng.random(_DIM, dtype=np.float32)
        unknown -= unknown.dot(alice) * alice
        norm = np.linalg.norm(unknown)
        if norm < 1e-6:
            unknown = rng.random(_DIM, dtype=np.float32)
        unknown = (unknown / np.linalg.norm(unknown)).astype(np.float32)

        pipeline = _build_pipeline(
            detections=[_detection_at(10, 10, 122, 122)],
            embeddings=unknown.reshape(1, _DIM),
            gallery=gallery,
            threshold=0.4,
        )

        results = pipeline.recognize(np.zeros((480, 640, 3), dtype=np.uint8))
        assert results[0].identity == "unknown"
        assert results[0].matched is False

    def test_two_faces_matched_independently(self) -> None:
        alice = _unit_vec(0)
        bob = _unit_vec(1)
        gallery = Gallery()
        gallery.add("Alice", alice)
        gallery.add("Bob", bob)

        pipeline = _build_pipeline(
            detections=[
                _detection_at(0, 0, 112, 112, score=0.95),
                _detection_at(200, 0, 312, 112, score=0.88),
            ],
            embeddings=np.stack([alice, bob]),
            gallery=gallery,
        )

        results = pipeline.recognize(np.zeros((480, 640, 3), dtype=np.uint8))

        assert len(results) == 2
        assert results[0].identity == "Alice"
        assert results[1].identity == "Bob"
        assert all(r.matched for r in results)

    def test_empty_gallery_all_faces_unknown(self) -> None:
        pipeline = _build_pipeline(
            detections=[_detection_at(10, 10, 122, 122)],
            embeddings=_unit_vec(5).reshape(1, _DIM),
            gallery=Gallery(),
        )

        results = pipeline.recognize(np.zeros((480, 640, 3), dtype=np.uint8))
        assert results[0].identity == "unknown"
        assert results[0].matched is False

    def test_alignment_step_produces_valid_crop_for_embedder(self) -> None:
        """Verify alignment produces exactly (112, 112, 3) uint8 crops."""
        embedder = MagicMock()
        embedder.embed_batch.return_value = _unit_vec(0).reshape(1, _DIM)

        detector = MagicMock()
        detector.detect.return_value = [_detection_at(50, 50, 162, 162)]

        pipeline = EidonPipeline(
            detector=detector,
            embedder=embedder,
            matcher=Matcher(),
            gallery=Gallery(),
        )
        pipeline.recognize(np.zeros((480, 640, 3), dtype=np.uint8))

        crops = embedder.embed_batch.call_args[0][0]
        assert len(crops) == 1
        assert crops[0].shape == (112, 112, 3)
        assert crops[0].dtype == np.uint8

    def test_gallery_hot_swap_takes_effect_immediately(self) -> None:
        alice = _unit_vec(0)
        bob = _unit_vec(1)

        gallery_a = Gallery()
        gallery_a.add("Alice", alice)

        detector = MagicMock()
        detector.detect.return_value = [_detection_at(10, 10, 122, 122)]

        embedder = MagicMock()
        embedder.embed_batch.return_value = bob.reshape(1, _DIM)

        pipeline = EidonPipeline(
            detector=detector,
            embedder=embedder,
            matcher=Matcher(threshold=0.9),
            gallery=gallery_a,
        )

        # Bob's embedding against Alice-only gallery → unknown
        before = pipeline.recognize(np.zeros((480, 640, 3), dtype=np.uint8))
        assert before[0].identity == "unknown"

        # Swap gallery to include Bob
        gallery_b = Gallery()
        gallery_b.add("Bob", bob)
        pipeline.update_gallery(gallery_b)

        after = pipeline.recognize(np.zeros((480, 640, 3), dtype=np.uint8))
        assert after[0].identity == "Bob"
        assert after[0].matched is True

    def test_gallery_persisted_and_reloaded(self, tmp_path: pytest.TempPathFactory) -> None:
        """Gallery saved to disk and loaded back produces the same recognition result."""
        alice = _unit_vec(0)
        path = tmp_path / "gallery.npz"

        g = Gallery()
        g.add("Alice", alice)
        g.save(path)

        loaded = Gallery.load(path)
        pipeline = _build_pipeline(
            detections=[_detection_at(10, 10, 122, 122)],
            embeddings=alice.reshape(1, _DIM),
            gallery=loaded,
        )

        results = pipeline.recognize(np.zeros((480, 640, 3), dtype=np.uint8))
        assert results[0].identity == "Alice"
        assert results[0].matched is True
