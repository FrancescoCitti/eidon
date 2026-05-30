"""Face identity gallery backed by a numpy .npz archive.

Stores one (N, 512) float32 embedding matrix per identity.
Designed for small galleries (~10 identities, ~5-20 embeddings each).
Not thread-safe — callers must serialise writes if used concurrently.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 512


class Gallery:
    """In-memory store of L2-normalised face embeddings, keyed by identity name.

    Typical usage::

        gallery = Gallery.load_or_empty(settings.gallery_path)
        gallery.add("Alice", embedder.embed(crop))
        gallery.save(settings.gallery_path)
    """

    def __init__(self) -> None:
        # identity → (N, 512) float32
        self._data: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, identity: str, embedding: np.ndarray) -> None:
        """Append one embedding to the gallery entry for *identity*.

        Args:
            identity:  Non-empty string label (e.g. ``"Alice"``).
            embedding: ``(512,)`` float32 L2-normalised embedding.

        Raises:
            ValueError: if *identity* is empty or *embedding* has wrong shape/dtype.
        """
        self._validate_identity(identity)
        self._validate_embedding(embedding)

        row = embedding.astype(np.float32).reshape(1, _EMBEDDING_DIM)
        if identity in self._data:
            self._data[identity] = np.vstack([self._data[identity], row])
        else:
            self._data[identity] = row
            logger.debug("New identity added to gallery", extra={"identity": identity})

    def remove(self, identity: str) -> None:
        """Remove all embeddings for *identity*.

        Raises:
            KeyError: if *identity* is not in the gallery.
        """
        if identity not in self._data:
            raise KeyError(f"Identity '{identity}' not found in gallery")
        del self._data[identity]
        logger.debug("Identity removed from gallery", extra={"identity": identity})

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def identities(self) -> list[str]:
        """Sorted list of identity names currently in the gallery."""
        return sorted(self._data)

    @property
    def size(self) -> int:
        """Total number of stored embeddings across all identities."""
        return sum(arr.shape[0] for arr in self._data.values())

    def embeddings_for(self, identity: str) -> np.ndarray:
        """Return ``(N, 512)`` float32 embeddings for *identity*.

        Raises:
            KeyError: if *identity* is not in the gallery.
        """
        if identity not in self._data:
            raise KeyError(f"Identity '{identity}' not found in gallery")
        return self._data[identity]

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, identity: object) -> bool:
        return identity in self._data

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist the gallery to *path* as a ``.npz`` archive.

        Creates parent directories as needed.  Overwrites any existing file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), **self._data)  # type: ignore[arg-type]
        logger.info(
            "Gallery saved",
            extra={"path": str(path), "identities": len(self._data), "total_embeddings": self.size},
        )

    @classmethod
    def load(cls, path: Path) -> Gallery:
        """Load a gallery from a ``.npz`` archive.

        Raises:
            FileNotFoundError: if *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Gallery file not found: {path}")

        gallery = cls()
        with np.load(str(path)) as data:
            for identity in data.files:
                gallery._data[identity] = data[identity].astype(np.float32)

        logger.info(
            "Gallery loaded",
            extra={"path": str(path), "identities": len(gallery), "total_embeddings": gallery.size},
        )
        return gallery

    @classmethod
    def load_or_empty(cls, path: Path) -> Gallery:
        """Load if *path* exists, otherwise return a new empty gallery."""
        path = Path(path)
        if path.exists():
            return cls.load(path)
        logger.debug("Gallery file not found, starting empty", extra={"path": str(path)})
        return cls()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_identity(identity: str) -> None:
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(f"identity must be a non-empty string, got {identity!r}")

    @staticmethod
    def _validate_embedding(embedding: np.ndarray) -> None:
        if embedding.shape != (_EMBEDDING_DIM,):
            raise ValueError(
                f"embedding must have shape ({_EMBEDDING_DIM},), got {embedding.shape}"
            )
        if embedding.dtype not in (np.float32, np.float64):
            raise ValueError(f"embedding must be float32 or float64, got {embedding.dtype}")
