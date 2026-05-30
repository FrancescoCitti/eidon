"""Enroll identities into the face gallery.

Reads images from a directory, detects the largest face in each, generates
an ArcFace embedding, and appends it to the gallery.

Usage:
    python scripts/enroll.py --identity "Alice" --images-dir data/raw/Alice
    python scripts/enroll.py --identity "Bob"   --images-dir data/raw/Bob --min-score 0.7
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eidon.config import settings
from eidon.detection import RetinaFaceDetector, align_face
from eidon.embedding import ArcFaceEmbedder
from eidon.matching import Gallery
from eidon.utils.logging import configure_logging

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def enroll(
    identity: str,
    images_dir: Path,
    gallery_path: Path,
    min_score: float,
) -> None:
    configure_logging(settings.log_level)

    logger.info("Loading models…")
    detector = RetinaFaceDetector(
        model_dir=settings.model_dir,
        pack=settings.det_model_pack,
        det_size=settings.det_size,
    )
    embedder = ArcFaceEmbedder(
        model_path=settings.rec_model_path,
        providers=["CPUExecutionProvider"],
    )

    gallery = Gallery.load_or_empty(gallery_path)
    existing = gallery.size

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        logger.error("No supported images found", extra={"dir": str(images_dir)})
        raise SystemExit(1)

    enrolled = 0
    skipped = 0

    for img_path in image_paths:
        import cv2

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read image, skipping", extra={"file": img_path.name})
            skipped += 1
            continue

        detection = detector.detect_largest(img, threshold=min_score)
        if detection is None:
            logger.warning(
                "No face detected, skipping",
                extra={"file": img_path.name, "min_score": min_score},
            )
            skipped += 1
            continue

        crop = align_face(img, detection.landmarks)
        embedding = embedder.embed(crop)
        gallery.add(identity, embedding)
        enrolled += 1
        logger.info(
            "Enrolled",
            extra={
                "file": img_path.name,
                "identity": identity,
                "det_score": round(detection.score, 3),
            },
        )

    if enrolled == 0:
        logger.error("No faces enrolled — check images and --min-score")
        raise SystemExit(1)

    gallery.save(gallery_path)
    logger.info(
        "Enrollment complete",
        extra={
            "identity": identity,
            "enrolled": enrolled,
            "skipped": skipped,
            "total_for_identity": enrolled
            + (gallery.size - existing - enrolled if identity in gallery else 0),
            "gallery_path": str(gallery_path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a new identity into the face gallery")
    parser.add_argument("--identity", required=True, help="Name of the person to enroll")
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing face images for this identity",
    )
    parser.add_argument(
        "--gallery-path",
        type=Path,
        default=settings.gallery_path,
        help=f"Gallery .npz file to update (default: {settings.gallery_path})",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Minimum detection confidence to accept a face (default: 0.5)",
    )
    args = parser.parse_args()

    if not args.images_dir.is_dir():
        print(f"Error: --images-dir '{args.images_dir}' is not a directory", file=sys.stderr)
        raise SystemExit(1)

    enroll(
        identity=args.identity,
        images_dir=args.images_dir,
        gallery_path=args.gallery_path,
        min_score=args.min_score,
    )


if __name__ == "__main__":
    main()
