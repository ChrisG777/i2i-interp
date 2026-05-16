"""Stable per-task seed derivation.

Used by dataset extractors to give every task a unique ``noise_seed``
that depends only on its ``task_id``. Without this, every task in a
dataset-derived bucket would share the extractor's hardcoded constant
and cross-task generations would all start from the same noise latent.

This is the convention for new tasks: ``noise_seed = task_seed(task_id)``.
A hand-chosen 32-bit int is also acceptable.
"""

from __future__ import annotations

import zlib


def task_seed(task_id: str) -> int:
    """Stable 32-bit non-negative seed derived from ``task_id``.

    ``zlib.crc32`` is salt-free across Python versions and processes, so
    re-running the reseed script on the same JSONL is a no-op. The
    32-bit unsigned range is well within torch's accepted seed range.
    """
    assert isinstance(task_id, str) and task_id, (
        f"task_seed requires a non-empty str task_id, got {task_id!r}"
    )
    return zlib.crc32(task_id.encode("utf-8"))
