"""Identity matcher using cosine similarity against a Gallery.

Since all embeddings are L2-normalised, cosine similarity reduces to
a plain dot product — no division needed at query time.
"""

from __future__ import annotations

import logging

import numpy as np

from eidon.matching.gallery import Gallery
from eidon.types import MatchResult

logger = logging.getLogger(__name__)

_UNKNOWN = "unknown"


class Matcher:
    """Nearest-neighbour identity matcher with a configurable similarity threshold.

    Strategy:
        For each identity in the gallery, compute the maximum cosine similarity
        across all of its stored embeddings.  Return the identity with the highest
        max-similarity, provided it meets *threshold*.  If no identity does, return
        ``"unknown"``.

    Args:
        threshold: Minimum cosine similarity to accept a match (default 0.4).
                   Typical ArcFace values: 0.3 (lenient) – 0.5 (strict).
    """

    def __init__(self, threshold: float = 0.4) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """Minimum cosine similarity required to accept a match."""
        return self._threshold

    def match(self, embedding: np.ndarray, gallery: Gallery) -> MatchResult:
        """Find the closest identity in *gallery* to *embedding*.

        Args:
            embedding: ``(512,)`` float32 L2-normalised query embedding.
            gallery:   Populated :class:`~eidon.matching.gallery.Gallery`.

        Returns:
            :class:`~eidon.types.MatchResult` with the best identity and its
            cosine similarity.  ``matched`` is ``False`` and ``identity`` is
            ``"unknown"`` when the gallery is empty or the best similarity falls
            below *threshold*.

        Raises:
            ValueError: if *embedding* has wrong shape or dtype.
        """
        self._validate_embedding(embedding)

        if not gallery.identities:
            logger.debug("Match attempted on empty gallery")
            return MatchResult(identity=_UNKNOWN, similarity=0.0, matched=False)

        best_identity = _UNKNOWN
        best_similarity = -1.0

        query = embedding.astype(np.float32)

        for identity in gallery.identities:
            stored = gallery.embeddings_for(identity)  # (N, 512)
            # Dot product == cosine similarity for L2-normalised vectors
            sims: np.ndarray = stored @ query          # (N,)
            max_sim = float(sims.max())
            if max_sim > best_similarity:
                best_similarity = max_sim
                best_identity = identity

        matched = best_similarity >= self._threshold
        result = MatchResult(
            identity=best_identity if matched else _UNKNOWN,
            similarity=best_similarity,
            matched=matched,
        )

        logger.debug(
            "Match result",
            extra={
                "identity": result.identity,
                "similarity": round(best_similarity, 4),
                "matched": matched,
                "threshold": self._threshold,
            },
        )
        return result

    @staticmethod
    def _validate_embedding(embedding: np.ndarray) -> None:
        if embedding.shape != (512,):
            raise ValueError(f"embedding must have shape (512,), got {embedding.shape}")
        if embedding.dtype not in (np.float32, np.float64):
            raise ValueError(f"embedding must be float32 or float64, got {embedding.dtype}")
