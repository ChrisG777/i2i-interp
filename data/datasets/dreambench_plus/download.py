"""Instruct the user how to fetch the DreamBench++ subject images + captions.

The dataset is hosted on HuggingFace; see the project README in
``dreambench_plus/README.md``. There is no anonymous-tarball URL we can hit
directly, so this script just prints the steps.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW_DIR = REPO_ROOT / "data" / "datasets" / "dreambench_plus" / "raw"


def main() -> int:
    print("==> DreamBench++ data fetch")
    print()
    print("Manual steps:")
    print("  1. Open the data link from dreambench_plus/README.md")
    print("     (HuggingFace dataset; you may need 'huggingface-cli login').")
    print("  2. Download the dataset into a local directory; you should see")
    print("     <dir>/images/{animal,human,object,style}/<NN>.jpg and")
    print("     <dir>/captions/{...}/<NN>.txt next to it.")
    print(f"  3. Move (or symlink) that directory to: {RAW_DIR}/")
    print()
    print("Expected layout after the move:")
    print(f"  {RAW_DIR}/images/object/01.jpg ...")
    print(f"  {RAW_DIR}/captions/object/01.txt ...")
    print()
    print(
        "Then run:  uv run python data/datasets/dreambench_plus/extract.py"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
