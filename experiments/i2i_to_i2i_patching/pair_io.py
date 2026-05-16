"""Shared pair-list I/O for i2i->i2i tooling.

The pair-list file format is one (source_task_id, target_task_id) pair per
line, whitespace-separated. Blank lines and ``#`` comments are skipped.

Used by:
* ``build_pairs_*.py`` — write pair lists.
* ``i2i_to_i2i_patch.py`` — consume ``--pair-list`` at runtime.
* ``scripts/reproduce_i2i_to_i2i_patching.py`` — point at checked-in pair files.
* ``scripts/judge/configs.py`` — derive judge entity-ids from the same file.
"""

from __future__ import annotations

from pathlib import Path


def read_pair_list(path: Path) -> list[tuple[str, str]]:
    """Parse a pair-list file. Asserts every non-blank line has 2 fields and
    that the file contains at least one pair."""
    pairs: list[tuple[str, str]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        assert len(parts) == 2, f"{path}: bad pair line {line!r}"
        pairs.append((parts[0], parts[1]))
    assert pairs, f"{path}: no pairs found"
    return pairs
