"""Face alignment — affine-warps a detected face to the ArcFace 112×112 template."""

from __future__ import annotations

import cv2
import numpy as np

# Standard ArcFace 5-landmark destination template (for 112×112 output).
# Order: left_eye, right_eye, nose_tip, left_mouth, right_mouth.
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate a 2D similarity transform (scale + rotation + translation).

    Uses the Umeyama algorithm: optimal in the least-squares sense.

    Args:
        src: (N, 2) float32 — source landmark coordinates.
        dst: (N, 2) float32 — destination landmark coordinates.

    Returns:
        (2, 3) float32 transformation matrix suitable for cv2.warpAffine.
    """
    n = src.shape[0]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    src_var: float = float((src_demean**2).sum() / n)
    cov = (dst_demean.T @ src_demean) / n

    U, s, Vt = np.linalg.svd(cov)

    # Flip the last singular vector if the solution would produce a reflection.
    d = np.ones(2, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[-1] = -1.0

    scale = float(np.dot(s, d) / src_var)
    R = U @ np.diag(d) @ Vt
    t = dst_mean - scale * (R @ src_mean)

    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = (scale * R).astype(np.float32)
    M[:, 2] = t.astype(np.float32)
    return M


def align_face(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Return a square aligned face crop centred on the ArcFace template.

    Args:
        image:     BGR image array of shape (H, W, 3), dtype uint8.
        landmarks: (5, 2) float32 — detected facial keypoints from RetinaFace.
        size:      Side length of the output square in pixels (default 112).

    Returns:
        (size, size, 3) uint8 BGR aligned crop.

    Raises:
        ValueError: if ``image`` or ``landmarks`` have unexpected shapes.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3), got {image.shape}")
    if landmarks.shape != (5, 2):
        raise ValueError(f"landmarks must be (5, 2), got {landmarks.shape}")

    scale = size / 112.0
    dst = (ARCFACE_DST * scale).astype(np.float32)
    M = _umeyama(landmarks.astype(np.float32), dst)
    return cv2.warpAffine(image, M, (size, size), borderValue=0.0)
