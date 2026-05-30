"""Unit tests for the embedding module.

All tests mock onnxruntime.InferenceSession — no model files needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eidon.embedding.arcface import (
    _EMBEDDING_DIM,
    _INPUT_SIZE,
    ArcFaceEmbedder,
    _l2_normalize,
    _preprocess,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_crop(fill: int = 128) -> np.ndarray:
    return np.full((_INPUT_SIZE, _INPUT_SIZE, 3), fill_value=fill, dtype=np.uint8)


def _fake_run(output_names: list[str], input_dict: dict) -> list[np.ndarray]:
    """Mock InferenceSession.run — returns random embeddings matching batch size."""
    n = next(iter(input_dict.values())).shape[0]
    return [np.random.default_rng(seed=42).random((n, _EMBEDDING_DIM), dtype=np.float32)]


@pytest.fixture()
def mock_embedder(tmp_path: pytest.TempPathFactory) -> ArcFaceEmbedder:
    """ArcFaceEmbedder with InferenceSession fully mocked."""
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input.1")]
    mock_session.get_outputs.return_value = [MagicMock(name="683")]
    mock_session.run.side_effect = _fake_run

    model_path = tmp_path / "w600k_mbf.onnx"
    model_path.touch()

    with patch("eidon.embedding.arcface.InferenceSession", return_value=mock_session):
        embedder = ArcFaceEmbedder(model_path=model_path)

    return embedder


# ---------------------------------------------------------------------------
# _preprocess
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_output_shape_is_chw(self) -> None:
        out = _preprocess(_make_crop())
        assert out.shape == (3, _INPUT_SIZE, _INPUT_SIZE)

    def test_output_dtype_is_float32(self) -> None:
        assert _preprocess(_make_crop()).dtype == np.float32

    def test_black_image_normalises_to_minus_one(self) -> None:
        out = _preprocess(_make_crop(fill=0))
        np.testing.assert_allclose(out, -1.0, atol=1e-5)

    def test_white_image_normalises_to_plus_one(self) -> None:
        # 255 / 127.5 - 1 = 1.0 (within float32 precision)
        out = _preprocess(_make_crop(fill=255))
        np.testing.assert_allclose(out, 1.0, atol=1e-3)

    def test_mid_grey_normalises_to_zero(self) -> None:
        # pixel=127.5 not representable as uint8; 127 and 128 should bracket 0
        out_127 = _preprocess(_make_crop(fill=127))
        out_128 = _preprocess(_make_crop(fill=128))
        assert float(out_127.mean()) < 0 < float(out_128.mean())

    def test_values_stay_within_range(self) -> None:
        rng = np.random.default_rng(0)
        crop = rng.integers(0, 255, (_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
        out = _preprocess(crop)
        assert out.min() >= -1.0 - 1e-4
        assert out.max() <= 1.0 + 1e-4


# ---------------------------------------------------------------------------
# _l2_normalize
# ---------------------------------------------------------------------------


class TestL2Normalize:
    def test_unit_norm_after_normalisation(self) -> None:
        v = np.random.default_rng(1).random((4, _EMBEDDING_DIM), dtype=np.float32)
        normed = _l2_normalize(v)
        np.testing.assert_allclose(np.linalg.norm(normed, axis=1), 1.0, atol=1e-6)

    def test_zero_vector_does_not_produce_nan(self) -> None:
        v = np.zeros((1, _EMBEDDING_DIM), dtype=np.float32)
        normed = _l2_normalize(v)
        assert not np.any(np.isnan(normed))

    def test_output_shape_preserved(self) -> None:
        v = np.random.default_rng(2).random((5, _EMBEDDING_DIM), dtype=np.float32)
        assert _l2_normalize(v).shape == (5, _EMBEDDING_DIM)

    def test_output_dtype_is_float32(self) -> None:
        v = np.random.default_rng(3).random((2, _EMBEDDING_DIM), dtype=np.float64)
        assert _l2_normalize(v).dtype == np.float32


# ---------------------------------------------------------------------------
# ArcFaceEmbedder
# ---------------------------------------------------------------------------


class TestArcFaceEmbedder:
    def test_embed_returns_shape_512(self, mock_embedder: ArcFaceEmbedder) -> None:
        emb = mock_embedder.embed(_make_crop())
        assert emb.shape == (_EMBEDDING_DIM,)

    def test_embed_returns_float32(self, mock_embedder: ArcFaceEmbedder) -> None:
        assert mock_embedder.embed(_make_crop()).dtype == np.float32

    def test_embed_is_l2_normalised(self, mock_embedder: ArcFaceEmbedder) -> None:
        emb = mock_embedder.embed(_make_crop())
        np.testing.assert_allclose(np.linalg.norm(emb), 1.0, atol=1e-5)

    def test_embed_batch_correct_shape(self, mock_embedder: ArcFaceEmbedder) -> None:
        embs = mock_embedder.embed_batch([_make_crop() for _ in range(5)])
        assert embs.shape == (5, _EMBEDDING_DIM)

    def test_embed_batch_empty_input(self, mock_embedder: ArcFaceEmbedder) -> None:
        embs = mock_embedder.embed_batch([])
        assert embs.shape == (0, _EMBEDDING_DIM)

    def test_embed_batch_all_rows_normalised(self, mock_embedder: ArcFaceEmbedder) -> None:
        embs = mock_embedder.embed_batch([_make_crop() for _ in range(3)])
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_identical_crops_produce_cosine_similarity_one(
        self, mock_embedder: ArcFaceEmbedder
    ) -> None:
        """Two identical crops → same raw embedding → cosine similarity = 1.0."""
        crop = _make_crop()
        # Force the session to return the same vector for both crops
        fixed = np.random.default_rng(99).random((1, _EMBEDDING_DIM), dtype=np.float32)
        mock_embedder._session.run.side_effect = (  # type: ignore[attr-defined]
            lambda *_: [np.tile(fixed, (2, 1))]
        )
        embs = mock_embedder.embed_batch([crop, crop])
        # L2-normalised dot product == cosine similarity
        similarity = float(embs[0] @ embs[1])
        np.testing.assert_allclose(similarity, 1.0, atol=1e-5)

    def test_raises_on_wrong_spatial_size(self, mock_embedder: ArcFaceEmbedder) -> None:
        bad = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="112, 112, 3"):
            mock_embedder.embed(bad)

    def test_raises_on_wrong_dtype(self, mock_embedder: ArcFaceEmbedder) -> None:
        bad = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            mock_embedder.embed(bad)

    def test_raises_on_missing_model_file(self, tmp_path: pytest.TempPathFactory) -> None:
        with pytest.raises(FileNotFoundError, match="download_models"):
            ArcFaceEmbedder(model_path=tmp_path / "nonexistent.onnx")

    def test_batch_index_in_error_message(self, mock_embedder: ArcFaceEmbedder) -> None:
        crops = [_make_crop(), np.zeros((64, 64, 3), dtype=np.uint8)]
        with pytest.raises(ValueError, match=r"crops\[1\]"):
            mock_embedder.embed_batch(crops)
