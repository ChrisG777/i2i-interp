"""Tiny load-or-run helper used by every per-task runner to avoid recomputing
baselines and intermediate cells that already exist on disk. Combined with
per-setting / per-mode skip logic, this lets a re-run with an additional
variant (e.g. ``--text-token-mode all padding_only``) generate only the
missing artifacts without burning GPU on the already-completed cells.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image


def load_or_run(
    path: Path,
    *,
    generate: Callable[[], Image.Image],
) -> Image.Image:
    """Return ``Image.open(path)`` if the file exists, otherwise call
    ``generate()``, save the result to ``path`` (creating parents), and
    return it. The caller chooses the file format via ``path.suffix``.
    """
    if path.exists():
        return Image.open(path)
    img = generate()
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return img


__all__ = ["load_or_run"]
