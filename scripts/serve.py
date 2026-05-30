"""Launch the Eidon REST API server.

Usage:
    python scripts/serve.py
    python scripts/serve.py --host 0.0.0.0 --port 8000

Docs available at http://localhost:8000/docs once running.

Note: use --workers 1 (the default).  Multiple workers each load their own
copy of the models and maintain separate in-process gallery state, which
will cause enroll/delete changes from one worker to be invisible to others.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn

from eidon.config import settings
from eidon.utils.logging import configure_logging


def main() -> None:
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description="Eidon REST API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--reload", action="store_true", help="Auto-reload on code changes (dev only)"
    )
    args = parser.parse_args()

    uvicorn.run(
        "eidon.serving.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
