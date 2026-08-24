"""
scripts/build_index.py
One-time script to build the ChromaDB vector index from the knowledge-base/.
Run this once before starting the agent or running evaluations.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --rebuild   # force full rebuild
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.indexer import build_index
from agent.config import DEBUG

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the ChromaDB knowledge-base index for the Aster & Row support agent."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a full rebuild even if an index already exists.",
    )
    args = parser.parse_args()

    print("Building knowledge base index...")
    collection = build_index(force_rebuild=args.rebuild)
    print(f"[OK] Index ready. Total chunks stored: {collection.count()}")


if __name__ == "__main__":
    main()
