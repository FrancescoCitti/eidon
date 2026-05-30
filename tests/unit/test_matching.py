"""Unit tests for the Gallery and Matcher classes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eidon.matching.gallery import Gallery
from eidon.matching.matcher import Matcher
from eidon.types import MatchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 512


def _unit_vec(seed: int = 0) -> np.ndarray:
    """Return a reproducible L2-normalised (512,) float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM, dtype=np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _orthogonal_vec(v: np.ndarray) -> np.ndarray:
    """Return a unit vector orthogonal to v (cosine similarity 0 with v)."""
    rng = np.random.default_rng(99)
    u = rng.random(_DIM, dtype=np.float32)
    u -= u.dot(v) * v          # Gram-Schmidt projection
    return (u / np.linalg.norm(u)).astype(np.float32)


# ---------------------------------------------------------------------------
# Gallery — mutation
# ---------------------------------------------------------------------------

class TestGalleryAdd:
    def test_first_add_creates_entry(self) -> None:
        g = Gallery()
        g.add("Alice", _unit_vec(0))
        assert "Alice" in g

    def test_add_increments_size(self) -> None:
        g = Gallery()
        g.add("Alice", _unit_vec(0))
        g.add("Alice", _unit_vec(1))
        assert g.size == 2

    def test_add_second_identity_grows_gallery(self) -> None:
        g = Gallery()
        g.add("Alice", _unit_vec(0))
        g.add("Bob", _unit_vec(1))
        assert len(g) == 2
        assert g.size == 2

    def test_embeddings_matrix_shape(self) -> None:
        g = Gallery()
        for i in range(3):
            g.add("Alice", _unit_vec(i))
        assert g.embeddings_for("Alice").shape == (3, _DIM)

    def test_add_raises_on_empty_identity(self) -> None:
        g = Gallery()
        with pytest.raises(ValueError, match="non-empty string"):
            g.add("", _unit_vec())

    def test_add_raises_on_whitespace_identity(self) -> None:
        g = Gallery()
        with pytest.raises(ValueError, match="non-empty string"):
            g.add("   ", _unit_vec())

    def test_add_raises_on_wrong_embedding_shape(self) -> None:
        g = Gallery()
        with pytest.raises(ValueError, match=r"\(512,\)"):
            g.add("Alice", np.zeros(256, dtype=np.float32))

    def test_add_raises_on_wrong_dtype(self) -> None:
        g = Gallery()
        with pytest.raises(ValueError, match="float32"):
            g.add("Alice", np.zeros(_DIM, dtype=np.int32))

    def test_add_accepts_float64(self) -> None:
        g = Gallery()
        g.add("Alice", _unit_vec().astype(np.float64))
        assert g.embeddings_for("Alice").dtype == np.float32


class TestGalleryRemove:
    def test_remove_deletes_identity(self) -> None:
        g = Gallery()
        g.add("Alice", _unit_vec())
        g.remove("Alice")
        assert "Alice" not in g

    def test_remove_updates_size(self) -> None:
        g = Gallery()
        g.add("Alice", _unit_vec(0))
        g.add("Alice", _unit_vec(1))
        g.add("Bob", _unit_vec(2))
        g.remove("Alice")
        assert g.size == 1

    def test_remove_nonexistent_raises_key_error(self) -> None:
        g = Gallery()
        with pytest.raises(KeyError):
            g.remove("Ghost")


class TestGalleryQueries:
    def test_identities_returns_sorted_list(self) -> None:
        g = Gallery()
        for name in ["Charlie", "Alice", "Bob"]:
            g.add(name, _unit_vec())
        assert g.identities == ["Alice", "Bob", "Charlie"]

    def test_size_empty_gallery_is_zero(self) -> None:
        assert Gallery().size == 0

    def test_embeddings_for_unknown_raises_key_error(self) -> None:
        g = Gallery()
        with pytest.raises(KeyError):
            g.embeddings_for("Ghost")

    def test_contains_operator(self) -> None:
        g = Gallery()
        g.add("Alice", _unit_vec())
        assert "Alice" in g
        assert "Bob" not in g


# ---------------------------------------------------------------------------
# Gallery — persistence
# ---------------------------------------------------------------------------

class TestGalleryPersistence:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        g = Gallery()
        v0, v1 = _unit_vec(0), _unit_vec(1)
        g.add("Alice", v0)
        g.add("Alice", v1)
        g.add("Bob", _unit_vec(2))

        path = tmp_path / "gallery.npz"
        g.save(path)

        loaded = Gallery.load(path)
        assert loaded.identities == ["Alice", "Bob"]
        assert loaded.size == 3
        np.testing.assert_allclose(loaded.embeddings_for("Alice")[0], v0, atol=1e-6)

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        g = Gallery()
        g.add("Alice", _unit_vec())
        path = tmp_path / "deep" / "nested" / "gallery.npz"
        g.save(path)
        assert path.exists()

    def test_load_raises_when_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Gallery.load(tmp_path / "missing.npz")

    def test_load_or_empty_returns_empty_when_missing(self, tmp_path: Path) -> None:
        g = Gallery.load_or_empty(tmp_path / "missing.npz")
        assert g.size == 0

    def test_load_or_empty_loads_when_present(self, tmp_path: Path) -> None:
        g = Gallery()
        g.add("Alice", _unit_vec())
        path = tmp_path / "g.npz"
        g.save(path)
        loaded = Gallery.load_or_empty(path)
        assert "Alice" in loaded

    def test_empty_gallery_save_load(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.npz"
        Gallery().save(path)
        loaded = Gallery.load(path)
        assert loaded.size == 0


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

class TestMatcherInit:
    def test_threshold_stored(self) -> None:
        m = Matcher(threshold=0.35)
        assert m.threshold == pytest.approx(0.35)

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            Matcher(threshold=1.5)


class TestMatcherMatch:
    def test_empty_gallery_returns_unknown(self) -> None:
        result = Matcher().match(_unit_vec(), Gallery())
        assert result.identity == "unknown"
        assert result.matched is False

    def test_exact_match_returns_correct_identity(self) -> None:
        g = Gallery()
        v = _unit_vec(0)
        g.add("Alice", v)
        result = Matcher(threshold=0.9).match(v, g)
        assert result.identity == "Alice"
        assert result.matched is True

    def test_similarity_of_identical_embedding_is_one(self) -> None:
        g = Gallery()
        v = _unit_vec(0)
        g.add("Alice", v)
        result = Matcher().match(v, g)
        np.testing.assert_allclose(result.similarity, 1.0, atol=1e-5)

    def test_below_threshold_returns_unknown(self) -> None:
        g = Gallery()
        v = _unit_vec(0)
        g.add("Alice", v)
        # Use an orthogonal vector — similarity ≈ 0, well below any reasonable threshold
        query = _orthogonal_vec(v)
        result = Matcher(threshold=0.4).match(query, g)
        assert result.identity == "unknown"
        assert result.matched is False

    def test_returns_closest_of_multiple_identities(self) -> None:
        alice = _unit_vec(0)
        bob = _unit_vec(1)
        g = Gallery()
        g.add("Alice", alice)
        g.add("Bob", bob)
        # Query closest to Alice
        result = Matcher(threshold=0.0).match(alice, g)
        assert result.identity == "Alice"

    def test_similarity_stored_even_when_unmatched(self) -> None:
        g = Gallery()
        v = _unit_vec(0)
        g.add("Alice", v)
        query = _orthogonal_vec(v)
        result = Matcher(threshold=0.99).match(query, g)
        # similarity is still the best we found, even though matched=False
        assert isinstance(result.similarity, float)
        assert result.matched is False

    def test_returns_match_result_type(self) -> None:
        g = Gallery()
        g.add("Alice", _unit_vec())
        result = Matcher().match(_unit_vec(), g)
        assert isinstance(result, MatchResult)

    def test_multiple_embeddings_per_identity_uses_max(self) -> None:
        """Gallery with two embeddings per identity: query should match the closest one."""
        alice_near = _unit_vec(0)
        alice_far = _unit_vec(1)
        bob = _unit_vec(2)

        g = Gallery()
        g.add("Alice", alice_far)   # less similar to query
        g.add("Alice", alice_near)  # more similar to query (same as query)
        g.add("Bob", bob)

        result = Matcher(threshold=0.0).match(alice_near, g)
        assert result.identity == "Alice"

    def test_raises_on_wrong_embedding_shape(self) -> None:
        with pytest.raises(ValueError, match="512"):
            Matcher().match(np.zeros(256, dtype=np.float32), Gallery())

    def test_raises_on_wrong_dtype(self) -> None:
        with pytest.raises(ValueError, match="float"):
            Matcher().match(np.zeros(512, dtype=np.int8), Gallery())
