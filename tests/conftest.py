"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def blank_bgr_image() -> np.ndarray:
    """480×640 black BGR image — useful as a no-op input."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture()
def white_bgr_image() -> np.ndarray:
    """480×640 white BGR image."""
    return np.full((480, 640, 3), fill_value=255, dtype=np.uint8)
