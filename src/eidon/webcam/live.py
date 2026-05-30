"""Real-time face recognition via webcam.

Frame-skip strategy: inference runs every ``skip_frames`` frames and the
latest results are cached and redrawn on every frame in between.  This
keeps the display smooth on CPU-only hardware (Intel N100) where a single
inference pass takes ~80-150 ms.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import cv2
import numpy as np

from eidon.pipeline import EidonPipeline
from eidon.types import RecognitionResult

logger = logging.getLogger(__name__)

# BGR colour palette
_GREEN = (0, 200, 0)
_ORANGE = (0, 140, 255)
_WHITE = (255, 255, 255)
_SHADOW = (20, 20, 20)


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------


class _FPSCounter:
    """Rolling-window frames-per-second estimator."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Record one frame and return the current FPS estimate."""
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Drawing utilities  (module-level so they are independently testable)
# ---------------------------------------------------------------------------


def annotate_frame(
    frame: np.ndarray,
    results: list[RecognitionResult],
    fps: float = 0.0,
) -> np.ndarray:
    """Return a copy of *frame* annotated with bounding boxes, name labels, and FPS.

    Args:
        frame:   BGR uint8 source image — not modified in place.
        results: Recognition results for the faces visible in *frame*.
        fps:     Rolling FPS value to display in the top-left corner.

    Returns:
        Annotated BGR uint8 image with the same shape as *frame*.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    for r in results:
        x1, y1, x2, y2 = r.detection.bbox.astype(int)
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)

        colour = _GREEN if r.matched else _ORANGE
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness=2)

        label = f"{r.identity} ({r.similarity:.2f})" if r.matched else "Unknown"
        _draw_label(out, label, x=x1, y=y1, colour=colour, above=True)

    _draw_label(out, f"FPS {fps:.1f}", x=8, y=8, colour=_WHITE, above=False)
    return out


def _draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    colour: tuple[int, int, int],
    *,
    above: bool = True,
    font_scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """Draw *text* with a dark filled background on *frame* in place.

    When ``above=True`` the label is placed just above (x, y); when ``False``
    it is placed just below — used for the FPS counter in the top-left corner.
    If the label would extend above the image boundary it flips to draw below.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 3

    if above and y - th - 2 * pad < 0:
        above = False  # not enough room above — draw below the box edge

    if above:
        text_y = y - pad
        bg_y1 = text_y - th - pad
        bg_y2 = text_y + baseline + pad
    else:
        text_y = y + th + pad
        bg_y1 = y
        bg_y2 = text_y + baseline + pad

    cv2.rectangle(frame, (x - pad, bg_y1), (x + tw + pad, bg_y2), _SHADOW, cv2.FILLED)
    cv2.putText(frame, text, (x, text_y), font, font_scale, colour, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


class LiveRecognition:
    """Webcam capture loop with real-time face recognition overlay.

    Args:
        pipeline:     Fully initialised :class:`~eidon.pipeline.EidonPipeline`.
        webcam_index: OpenCV ``VideoCapture`` device index (default 0).
        skip_frames:  Run inference every *N* frames; redraw cached results between
                      runs.  Higher values improve perceived smoothness at the cost
                      of recognition update rate.  Must be >= 1.
        window_title: Title of the OpenCV display window.
    """

    def __init__(
        self,
        pipeline: EidonPipeline,
        webcam_index: int = 0,
        skip_frames: int = 2,
        window_title: str = "Face Recognition  —  press q to quit",
    ) -> None:
        if skip_frames < 1:
            raise ValueError(f"skip_frames must be >= 1, got {skip_frames}")
        self._pipeline = pipeline
        self._webcam_index = webcam_index
        self._skip_frames = skip_frames
        self._window_title = window_title

    def run(self) -> None:
        """Open the webcam and block until the user presses ``q`` or closes the window."""
        cap = cv2.VideoCapture(self._webcam_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam at index {self._webcam_index}. "
                "Ensure the device is connected and not already in use by another process."
            )

        logger.info(
            "Webcam live started",
            extra={"index": self._webcam_index, "skip_frames": self._skip_frames},
        )

        # Create the window before the loop so the OS has time to map it.
        cv2.namedWindow(self._window_title, cv2.WINDOW_NORMAL)

        fps_counter = _FPSCounter()
        cached_results: list[RecognitionResult] = []
        frame_index = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    logger.warning("Frame read failed — stopping capture loop")
                    break

                if frame_index % self._skip_frames == 0:
                    cached_results = self._pipeline.recognize(frame)

                fps = fps_counter.tick()
                cv2.imshow(self._window_title, annotate_frame(frame, cached_results, fps))
                frame_index += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User pressed 'q' — stopping")
                    break

                # Only check window visibility after the first few frames — on Wayland
                # the compositor needs a moment to map the window before WND_PROP_VISIBLE
                # becomes reliable.  Checking on frame 0 always returns -1 on Wayland.
                if (
                    frame_index > 5
                    and cv2.getWindowProperty(self._window_title, cv2.WND_PROP_VISIBLE) < 1
                ):
                    logger.info("Window closed — stopping")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam released")
