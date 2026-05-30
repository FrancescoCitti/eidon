"""Download pretrained ONNX model pack (RetinaFace + ArcFace MobileNet).

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --model-dir /path/to/models --pack buffalo_s
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eidon.config import settings
from eidon.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def download_pack(model_dir: Path, pack: str) -> None:
    """Download and verify an InsightFace model pack.

    InsightFace stores packs at {model_dir}/{pack}/.
    It skips the download if the directory already exists and is non-empty.
    """
    pack_dir = model_dir / pack
    if pack_dir.exists() and any(pack_dir.glob("*.onnx")):
        logger.info(
            "Model pack already present, skipping download",
            extra={"pack": pack, "path": str(pack_dir)},
        )
        return

    logger.info("Downloading model pack", extra={"pack": pack, "dest": str(model_dir)})

    try:
        # FaceAnalysis uses {root}/models/{pack}/ as the download target.
        # By setting root=model_dir.parent we get model_dir/{pack}/.
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name=pack,
            root=str(model_dir.parent),
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))

    except Exception as exc:
        logger.error("Download failed", extra={"error": str(exc)})
        raise SystemExit(1) from exc

    onnx_files = list(pack_dir.glob("*.onnx"))
    if not onnx_files:
        logger.error("Download completed but no .onnx files found", extra={"path": str(pack_dir)})
        raise SystemExit(1)

    logger.info(
        "Download complete",
        extra={"pack": pack, "files": [f.name for f in onnx_files]},
    )


def main() -> None:
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description="Download InsightFace ONNX model pack")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=settings.model_dir,
        help=f"Target directory for model files (default: {settings.model_dir})",
    )
    parser.add_argument(
        "--pack",
        default=settings.det_model_pack,
        help=f"Model pack name (default: {settings.det_model_pack})",
    )
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    download_pack(args.model_dir, args.pack)


if __name__ == "__main__":
    main()
