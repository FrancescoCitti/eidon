"""Guided webcam enrollment with automatic head-pose capture.

Walks the user through five poses (frontal → look left → look right →
tilt left → tilt right) and auto-captures frames when the target pose is
held steadily — no SPACE-bar required.  Produces a richer, more robust
gallery than enrolling from static photos.

Usage:
    python scripts/enroll_webcam.py --identity "Your Name"
    python scripts/enroll_webcam.py --identity "Your Name" --captures-per-step 3

Press q at any time to abort without saving.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

from eidon.config import settings
from eidon.detection import RetinaFaceDetector, align_face
from eidon.embedding import ArcFaceEmbedder
from eidon.matching import Gallery
from eidon.utils.logging import configure_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colours (BGR)
# ---------------------------------------------------------------------------
_GREEN = (0, 200, 0)
_YELLOW = (0, 200, 220)
_CYAN = (220, 200, 0)
_RED = (60, 60, 220)
_WHITE = (255, 255, 255)
_DARK = (25, 25, 25)

# Frames the pose must be held before auto-capture fires.
# At ~15 FPS on an N100 this is roughly 0.7 s.
_DWELL_REQUIRED = 11

# Seconds to pause display after each capture (feedback window).
_POST_CAPTURE_PAUSE = 0.6


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Step:
    instruction: str
    # Acceptable yaw range.
    #   yaw > 0  → person's nose is to the RIGHT of eye midpoint in image
    #            = person is looking to THEIR left
    #   yaw < 0  → person looking to THEIR right
    yaw_lo: float
    yaw_hi: float
    # Acceptable roll range (degrees).
    #   roll > 0 → right eye lower than left in image = tilt to right shoulder
    #   roll < 0 → tilt to left shoulder
    roll_lo: float
    roll_hi: float
    # Arrow drawn on-screen as (dx, dy) in image-coordinate direction.
    # None = no arrow (frontal step).
    arrow_dir: tuple[float, float] | None = None


_STEPS: list[_Step] = [
    _Step(
        "Look straight at the camera",
        yaw_lo=-0.09,
        yaw_hi=0.09,
        roll_lo=-9.0,
        roll_hi=9.0,
        arrow_dir=None,
    ),
    _Step(
        "Slowly turn your head LEFT",
        yaw_lo=0.15,
        yaw_hi=0.55,
        roll_lo=-20.0,
        roll_hi=20.0,
        arrow_dir=(-1.0, 0.0),
    ),
    _Step(
        "Slowly turn your head RIGHT",
        yaw_lo=-0.55,
        yaw_hi=-0.15,
        roll_lo=-20.0,
        roll_hi=20.0,
        arrow_dir=(1.0, 0.0),
    ),
    _Step(
        "Tilt head to your LEFT shoulder",
        yaw_lo=-0.12,
        yaw_hi=0.12,
        roll_lo=-35.0,
        roll_hi=-12.0,
        arrow_dir=(-0.7, 0.7),
    ),
    _Step(
        "Tilt head to your RIGHT shoulder",
        yaw_lo=-0.12,
        yaw_hi=0.12,
        roll_lo=12.0,
        roll_hi=35.0,
        arrow_dir=(0.7, 0.7),
    ),
]


# ---------------------------------------------------------------------------
# Pose estimation from 5 RetinaFace landmarks
# ---------------------------------------------------------------------------


def _estimate_pose(landmarks: np.ndarray) -> tuple[float, float]:
    """Return (yaw, roll) from 5-point landmarks.

    Landmark order: left_eye, right_eye, nose_tip, left_mouth, right_mouth.

    yaw  > 0  → person looking to their LEFT  (nose right of eye midpoint in image)
    yaw  < 0  → person looking to their RIGHT
    roll > 0  → tilted toward right shoulder (right eye lower in image)
    roll < 0  → tilted toward left shoulder
    """
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]

    eye_mid_x = float(left_eye[0] + right_eye[0]) / 2.0
    eye_width = float(abs(right_eye[0] - left_eye[0]))

    if eye_width < 1.0:
        return 0.0, 0.0

    yaw = float(nose[0] - eye_mid_x) / eye_width
    roll = float(
        np.degrees(
            np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0],
            )
        )
    )
    return yaw, roll


def _pose_ok(yaw: float, roll: float, step: _Step) -> bool:
    return step.yaw_lo <= yaw <= step.yaw_hi and step.roll_lo <= roll <= step.roll_hi


# ---------------------------------------------------------------------------
# UI drawing
# ---------------------------------------------------------------------------


def _draw_ui(
    frame: np.ndarray,
    step_idx: int,
    step: _Step,
    dwell: int,
    captures_in_step: int,
    captures_per_step: int,
    total_captured: int,
    total_needed: int,
    face_ok: bool,
    flash_until: float,
) -> None:
    """Render instruction, step dots, dwell bar, guide arrow and flash in place."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- Capture flash: bright border ----------------------------------------
    if time.monotonic() < flash_until:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), _GREEN, 12)

    # --- Step dots at the top -------------------------------------------------
    dot_r, dot_gap, dot_y = 8, 26, 18
    total_steps = len(_STEPS)
    strip_w = total_steps * (2 * dot_r + dot_gap)
    dot_x0 = (w - strip_w) // 2

    for i in range(total_steps):
        cx = dot_x0 + i * (2 * dot_r + dot_gap) + dot_r
        if i < step_idx:
            cv2.circle(frame, (cx, dot_y), dot_r, _GREEN, -1)  # done
        elif i == step_idx:
            cv2.circle(frame, (cx, dot_y), dot_r, _CYAN, -1)  # active
            cv2.circle(frame, (cx, dot_y), dot_r + 2, _WHITE, 2)
        else:
            cv2.circle(frame, (cx, dot_y), dot_r, _DARK, -1)  # pending
            cv2.circle(frame, (cx, dot_y), dot_r, _WHITE, 1)

    # --- Main instruction text ------------------------------------------------
    inst = step.instruction
    (tw, th), _ = cv2.getTextSize(inst, font, 0.70, 2)
    ix = (w - tw) // 2
    iy = dot_y + dot_r + 30
    cv2.rectangle(frame, (ix - 6, iy - th - 6), (ix + tw + 6, iy + 8), _DARK, cv2.FILLED)
    cv2.putText(frame, inst, (ix, iy), font, 0.70, _WHITE, 2, cv2.LINE_AA)

    # --- Sub-caption: captures within this step ------------------------------
    sub = f"Step {step_idx + 1}/{total_steps}   Captured: {total_captured}/{total_needed}"
    (sw, sh), _ = cv2.getTextSize(sub, font, 0.48, 1)
    sx = (w - sw) // 2
    sy = iy + 30
    cv2.rectangle(frame, (sx - 4, sy - sh - 4), (sx + sw + 4, sy + 6), _DARK, cv2.FILLED)
    cv2.putText(frame, sub, (sx, sy), font, 0.48, _YELLOW, 1, cv2.LINE_AA)

    # --- Dwell progress bar --------------------------------------------------
    bar_h = 10
    bar_y = sy + 20
    bar_margin = 40
    bar_w = w - 2 * bar_margin
    fill_w = int(bar_w * min(dwell, _DWELL_REQUIRED) / _DWELL_REQUIRED)

    bg_rect = ((bar_margin, bar_y), (bar_margin + bar_w, bar_y + bar_h))
    cv2.rectangle(frame, *bg_rect, _DARK, cv2.FILLED)
    if fill_w > 0:
        bar_colour = _GREEN if face_ok else _YELLOW
        fill_rect = ((bar_margin, bar_y), (bar_margin + fill_w, bar_y + bar_h))
        cv2.rectangle(frame, *fill_rect, bar_colour, cv2.FILLED)
    cv2.rectangle(frame, *bg_rect, _WHITE, 1)

    # --- Guide arrow ---------------------------------------------------------
    if step.arrow_dir is not None:
        dx, dy = step.arrow_dir
        arrow_len = min(w, h) // 7
        cx_a = w // 2
        cy_a = h * 3 // 4
        pt1 = (cx_a, cy_a)
        pt2 = (int(cx_a + dx * arrow_len), int(cy_a + dy * arrow_len))
        cv2.arrowedLine(frame, pt1, pt2, _CYAN, 4, tipLength=0.35)


# ---------------------------------------------------------------------------
# Main enrollment loop
# ---------------------------------------------------------------------------


def enroll_from_webcam(
    identity: str,
    captures_per_step: int,
    webcam_index: int,
    gallery_path: Path,
    min_det_score: float,
) -> None:
    configure_logging(settings.log_level)

    logger.info("Loading models…")
    detector = RetinaFaceDetector(
        model_dir=settings.model_dir,
        pack=settings.det_model_pack,
        det_size=settings.det_size,
    )
    embedder = ArcFaceEmbedder(model_path=settings.rec_model_path)
    gallery = Gallery.load_or_empty(gallery_path)

    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam at index {webcam_index}.")

    title = f"Enrolling: {identity}  —  follow the on-screen instructions  (q = quit)"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    total_needed = len(_STEPS) * captures_per_step
    total_captured = 0
    step_idx = 0
    captures_in_step = 0
    dwell = 0
    flash_until = 0.0
    pause_until = 0.0
    embeddings_collected: list[np.ndarray] = []

    logger.info(
        "Guided enrollment started",
        extra={"identity": identity, "steps": len(_STEPS), "captures_per_step": captures_per_step},
    )

    try:
        while step_idx < len(_STEPS):
            ok, frame = cap.read()
            if not ok:
                logger.warning("Frame read failed")
                break

            now = time.monotonic()
            step = _STEPS[step_idx]
            detection = detector.detect_largest(frame, threshold=min_det_score)

            # --- Pose check --------------------------------------------------
            face_ok = False
            if detection is not None and now >= pause_until:
                yaw, roll = _estimate_pose(detection.landmarks)
                face_ok = _pose_ok(yaw, roll, step)

                # Draw face bounding box
                x1, y1, x2, y2 = detection.bbox.astype(int)
                h_f, w_f = frame.shape[:2]
                x1, x2 = max(0, x1), min(w_f - 1, x2)
                y1, y2 = max(0, y1), min(h_f - 1, y2)
                box_colour = _GREEN if face_ok else _YELLOW
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2)

                if face_ok:
                    dwell += 1
                else:
                    dwell = 0

            elif detection is None:
                dwell = 0

            # --- Auto-capture ------------------------------------------------
            if face_ok and dwell >= _DWELL_REQUIRED and now >= pause_until:
                crop = align_face(frame, detection.landmarks)  # type: ignore[union-attr]
                embedding = embedder.embed(crop)
                embeddings_collected.append(embedding)
                captures_in_step += 1
                total_captured += 1
                dwell = 0
                flash_until = now + _POST_CAPTURE_PAUSE
                pause_until = now + _POST_CAPTURE_PAUSE

                logger.info(
                    "Auto-captured",
                    extra={
                        "step": step.instruction,
                        "capture": captures_in_step,
                        "total": total_captured,
                    },
                )

                if captures_in_step >= captures_per_step:
                    step_idx += 1
                    captures_in_step = 0
                    dwell = 0

            # --- Draw UI -----------------------------------------------------
            _draw_ui(
                frame,
                step_idx=min(step_idx, len(_STEPS) - 1),
                step=_STEPS[min(step_idx, len(_STEPS) - 1)],
                dwell=dwell,
                captures_in_step=captures_in_step,
                captures_per_step=captures_per_step,
                total_captured=total_captured,
                total_needed=total_needed,
                face_ok=face_ok,
                flash_until=flash_until,
            )

            cv2.imshow(title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Enrollment cancelled by user")
                embeddings_collected.clear()
                break

            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                embeddings_collected.clear()
                break

        # Brief "all done" display before closing
        if step_idx >= len(_STEPS) and embeddings_collected:
            done_frame = np.zeros((200, 500, 3), dtype=np.uint8)
            msg = f"All done!  {total_captured} frames captured."
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(
                done_frame,
                msg,
                ((500 - tw) // 2, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                _GREEN,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(title, done_frame)
            cv2.waitKey(1200)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not embeddings_collected:
        logger.error("No frames captured — gallery not updated")
        raise SystemExit(1)

    for emb in embeddings_collected:
        gallery.add(identity, emb)

    gallery.save(gallery_path)
    logger.info(
        "Guided enrollment complete",
        extra={
            "identity": identity,
            "frames_enrolled": len(embeddings_collected),
            "gallery_path": str(gallery_path),
        },
    )
    print(f"\nEnrolled {len(embeddings_collected)} frames for '{identity}'")
    print(f"Gallery saved to: {gallery_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guided webcam enrollment: automatically captures poses for robust recognition"
    )
    parser.add_argument("--identity", required=True, help="Name of the person to enroll")
    parser.add_argument(
        "--captures-per-step",
        type=int,
        default=2,
        help="Frames to auto-capture per pose step (default: 2, total: 10)",
    )
    parser.add_argument(
        "--webcam",
        type=int,
        default=settings.webcam_index,
        help=f"Camera device index (default: {settings.webcam_index})",
    )
    parser.add_argument(
        "--gallery-path",
        type=Path,
        default=settings.gallery_path,
        help=f"Gallery file to update (default: {settings.gallery_path})",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="Minimum RetinaFace detection score to accept a frame (default: 0.6)",
    )
    args = parser.parse_args()

    enroll_from_webcam(
        identity=args.identity,
        captures_per_step=args.captures_per_step,
        webcam_index=args.webcam,
        gallery_path=args.gallery_path,
        min_det_score=args.min_score,
    )


if __name__ == "__main__":
    main()
