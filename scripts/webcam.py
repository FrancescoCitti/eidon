"""Launch real-time face recognition via webcam.

Usage:
    python scripts/webcam.py
    python scripts/webcam.py --webcam 1 --skip-frames 3

Make sure you have enrolled at least one identity first:
    python scripts/enroll.py --identity "Your Name" --images-dir /path/to/photos
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# The opencv-python wheel only bundles the xcb (X11) Qt platform plugin — the
# Wayland plugin is not included.  On GNOME/Wayland, XWayland provides a
# compatible X11 display, so xcb works correctly via the DISPLAY env var.
# Explicitly pin to xcb so Qt never tries (and crashes on) the missing wayland plugin.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from eidon.config import settings
from eidon.pipeline import EidonPipeline
from eidon.utils.logging import configure_logging
from eidon.webcam.live import LiveRecognition

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description="Real-time face recognition via webcam")
    parser.add_argument(
        "--webcam",
        type=int,
        default=settings.webcam_index,
        help=f"Camera device index (default: {settings.webcam_index})",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=settings.inference_skip_frames,
        help=f"Run inference every N frames (default: {settings.inference_skip_frames})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.similarity_threshold,
        help=f"Match similarity threshold 0-1 (default: {settings.similarity_threshold})",
    )
    args = parser.parse_args()

    logger.info("Loading pipeline — this may take a few seconds…")
    try:
        pipeline = EidonPipeline.from_settings()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        print(
            "\nModels not found. Run first:\n  python scripts/download_models.py\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    live = LiveRecognition(
        pipeline=pipeline,
        webcam_index=args.webcam,
        skip_frames=args.skip_frames,
    )

    try:
        live.run()
    except RuntimeError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
