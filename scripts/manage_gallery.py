"""Gallery management CLI.

Subcommands:
    list    — show all enrolled identities and their embedding counts
    delete  — remove one identity from the gallery
    rename  — rename an identity without re-enrolling
    info    — show gallery file path, size on disk, total embeddings

Usage:
    python scripts/manage_gallery.py list
    python scripts/manage_gallery.py delete --identity "francesco2"
    python scripts/manage_gallery.py rename --identity "francesco2" --new-name "Francesco"
    python scripts/manage_gallery.py info
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eidon.config import settings
from eidon.matching import Gallery


def _load(gallery_path: Path) -> Gallery:
    if not gallery_path.exists():
        print(f"No gallery found at {gallery_path}", file=sys.stderr)
        raise SystemExit(1)
    return Gallery.load(gallery_path)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_list(gallery_path: Path) -> None:
    gallery = _load(gallery_path)
    if not gallery.identities:
        print("Gallery is empty.")
        return
    print(f"{'Identity':<30}  Embeddings")
    print("-" * 44)
    for name in gallery.identities:
        n = gallery.embeddings_for(name).shape[0]
        print(f"  {name:<28}  {n}")
    print("-" * 44)
    print(f"  {'TOTAL':<28}  {gallery.size}")


def cmd_delete(gallery_path: Path, identity: str) -> None:
    gallery = _load(gallery_path)
    if identity not in gallery:
        print(f"Identity '{identity}' not found.  Enrolled names:", file=sys.stderr)
        for name in gallery.identities:
            print(f"  {name}", file=sys.stderr)
        raise SystemExit(1)

    n = gallery.embeddings_for(identity).shape[0]
    gallery.remove(identity)
    gallery.save(gallery_path)
    print(f"Deleted '{identity}' ({n} embeddings removed).")
    print(f"Remaining: {gallery.identities}")


def cmd_rename(gallery_path: Path, identity: str, new_name: str) -> None:
    gallery = _load(gallery_path)
    if identity not in gallery:
        print(f"Identity '{identity}' not found.", file=sys.stderr)
        raise SystemExit(1)
    if new_name in gallery:
        print(f"'{new_name}' already exists in the gallery.", file=sys.stderr)
        raise SystemExit(1)

    embeddings = gallery.embeddings_for(identity)
    gallery.remove(identity)
    for emb in embeddings:
        gallery.add(new_name, emb)
    gallery.save(gallery_path)
    print(f"Renamed '{identity}' → '{new_name}' ({embeddings.shape[0]} embeddings).")


def cmd_info(gallery_path: Path) -> None:
    print(f"Path  : {gallery_path.resolve()}")
    if not gallery_path.exists():
        print("Status: file does not exist yet")
        return
    size_kb = gallery_path.stat().st_size / 1024
    gallery = Gallery.load(gallery_path)
    print(f"Size  : {size_kb:.1f} KB")
    print(f"Identities : {len(gallery)}")
    print(f"Embeddings : {gallery.size} total")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the face recognition gallery")
    parser.add_argument(
        "--gallery-path",
        type=Path,
        default=settings.gallery_path,
        help=f"Gallery .npz file (default: {settings.gallery_path})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list",  help="List enrolled identities")
    sub.add_parser("info",  help="Show gallery file info")

    p_del = sub.add_parser("delete", help="Remove an identity from the gallery")
    p_del.add_argument("--identity", required=True, help="Identity name to delete")

    p_ren = sub.add_parser("rename", help="Rename an identity")
    p_ren.add_argument("--identity",  required=True, help="Current identity name")
    p_ren.add_argument("--new-name",  required=True, help="New identity name")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args.gallery_path)
    elif args.command == "delete":
        cmd_delete(args.gallery_path, args.identity)
    elif args.command == "rename":
        cmd_rename(args.gallery_path, args.identity, args.new_name)
    elif args.command == "info":
        cmd_info(args.gallery_path)


if __name__ == "__main__":
    main()
