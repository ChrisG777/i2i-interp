"""Structured task layer for diffusion image-editing experiments.

Tasks live in ``data/tasks/<bucket>/tasks.jsonl`` where ``bucket`` is one of:

* ``add`` / ``remove`` / ``customize`` — dataset-derived tasks
  (sun397, dreambench_plus). Bucket name == edit_type.
* ``manual`` — the 450 real-photo analogues paired against the style
  customize tasks. Each entry's ``edit_type`` field gives the semantic
  category.
* ``solid_color`` — synthetic color × object grid; every task uses
  ``edit_type=customize`` and ``real_ref_name=solid_<color>``.

Loading API:

* :func:`load_tasks(bucket, source=None, limit=None)` — list of
  ``TaskDefinition`` for one or more buckets.
* :func:`get_task(task_id)` — single task by id.

Generation defaults (``NUM_INFERENCE_STEPS``, ``HEIGHT``, ``WIDTH``,
``gen_kwargs``) are exported from this module too so the priority experiments
have one import path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

# ---------------------------------------------------------------------------
# Generation defaults (model-level constants, kept here so callers have one
# import path).
# ---------------------------------------------------------------------------

NUM_INFERENCE_STEPS = 1
HEIGHT = 1024
WIDTH = 1024


def gen_kwargs() -> dict:
    """Generation kwargs for ``DiffusionModel.generate()``."""
    return dict(
        num_inference_steps=NUM_INFERENCE_STEPS,
        height=HEIGHT,
        width=WIDTH,
    )


# ---------------------------------------------------------------------------
# TaskDefinition
# ---------------------------------------------------------------------------

EditType = Literal["add", "remove", "customize"]
Source = Literal[
    "dreambench_plus",
    "manual",
    "solid_color",
    "property_manual",
    "sun397",
]
EDIT_TYPES: tuple[EditType, ...] = ("add", "remove", "customize")
BUCKETS: tuple[str, ...] = (
    "add",
    "remove",
    "manual",
    "solid_color",
    "style",
    "dreambench_humans",
    "dreambench_humans_shared",
)


@dataclass(frozen=True)
class TaskDefinition:
    """One image-editing task.

    ``source_image_path``, ``real_ref_name``, ``ref_seed`` describe the
    reference image:

    * ``source_image_path`` — load a real image from disk (dataset tasks).
    * ``real_ref_name`` — load ``<real_ref_dir>/<name>.{png,jpg,jpeg}``.
      ``real_ref_dir`` is per-task; no hardcoded default.
    * ``ref_seed`` — generate a synthetic reference via
      ``model.generate(source_caption, seed=ref_seed)``; ``source_caption``
      is the prompt.
    """

    task_id: str
    edit_type: EditType
    source: Source
    instruction: str

    source_image_path: str | None = None
    source_caption: str | None = None
    ref_seed: int | None = None
    noise_seed: int | None = None
    real_ref_name: str | None = None
    real_ref_dir: str | None = None
    height: int = HEIGHT
    width: int = WIDTH
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_path = self.source_image_path is not None
        has_real = self.real_ref_name is not None
        has_synth = self.ref_seed is not None
        assert has_path or has_real or has_synth, (
            f"task {self.task_id}: must set at least one of "
            f"source_image_path / real_ref_name / ref_seed"
        )
        assert not (has_path and has_real), (
            f"task {self.task_id}: source_image_path and real_ref_name are "
            f"mutually exclusive"
        )
        assert (self.real_ref_name is None) == (self.real_ref_dir is None), (
            f"task {self.task_id}: real_ref_name and real_ref_dir must be set together "
            f"(got real_ref_name={self.real_ref_name!r}, real_ref_dir={self.real_ref_dir!r})"
        )
        if self.ref_seed is not None and self.noise_seed is not None:
            assert self.ref_seed != self.noise_seed, (
                f"task {self.task_id}: ref_seed == noise_seed == {self.ref_seed}"
            )
        assert self.height % 16 == 0 and self.width % 16 == 0, (
            f"task {self.task_id}: height/width must be multiples of 16, "
            f"got {self.height}x{self.width}"
        )
        assert self.edit_type in EDIT_TYPES, (
            f"task {self.task_id}: unknown edit_type {self.edit_type!r}"
        )

    @property
    def ref_label(self) -> str:
        """Human-readable description of this task's reference source.

        Single source of truth for both presentation (suptitles, logs) and
        debug output. Mirrors the three branches enforced by ``__post_init__``
        — anything that adds a new ref source has to touch both the invariant
        and this property in the same dataclass.
        """
        if self.real_ref_name is not None:
            return f"real_ref={self.real_ref_dir}/{self.real_ref_name!r}"
        if self.source_image_path is not None:
            return f"source_image={os.path.basename(self.source_image_path)!r}"
        assert self.ref_seed is not None and self.source_caption is not None, (
            f"task {self.task_id}: TaskDefinition invariant violated — no ref source"
        )
        return f"'{self.source_caption}' (seed={self.ref_seed})"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

TASKS_ROOT = Path(__file__).resolve().parents[2] / "data" / "tasks"


def _entry_to_task(d: dict) -> TaskDefinition:
    return TaskDefinition(
        task_id=d["task_id"],
        edit_type=d["edit_type"],
        source=d["source"],
        instruction=d["instruction"],
        source_image_path=d.get("source_image_path"),
        source_caption=d.get("source_caption"),
        ref_seed=d.get("ref_seed"),
        noise_seed=d.get("noise_seed"),
        real_ref_name=d.get("real_ref_name"),
        real_ref_dir=d.get("real_ref_dir"),
        height=d.get("height", HEIGHT),
        width=d.get("width", WIDTH),
        metadata=d.get("metadata", {}) or {},
    )


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_tasks(
    bucket: str | Iterable[str],
    *,
    source: str | None = None,
    limit: int | None = None,
) -> list[TaskDefinition]:
    """Load tasks from one or more buckets under ``data/tasks/<bucket>/``.

    ``bucket`` may be a single name or an iterable; tasks are returned in
    bucket order, then in source-file order. ``source`` filters within each
    bucket. ``limit`` caps the total returned across all buckets.
    """
    if isinstance(bucket, str):
        buckets = [bucket]
    else:
        buckets = list(bucket)
    for b in buckets:
        assert b in BUCKETS, f"unknown bucket {b!r}; known: {BUCKETS}"

    out: list[TaskDefinition] = []
    for b in buckets:
        path = TASKS_ROOT / b / "tasks.jsonl"
        if not path.exists():
            continue
        for entry in _read_jsonl(path):
            if source is not None and entry.get("source") != source:
                continue
            out.append(_entry_to_task(entry))
            if limit is not None and len(out) >= limit:
                return out
    return out


def get_task(task_id: str) -> TaskDefinition:
    """Look up one task by id. Searches every bucket."""
    for b in BUCKETS:
        path = TASKS_ROOT / b / "tasks.jsonl"
        if not path.exists():
            continue
        for entry in _read_jsonl(path):
            if entry["task_id"] == task_id:
                return _entry_to_task(entry)
    raise KeyError(f"Unknown task_id: {task_id!r}")


__all__ = [
    "TaskDefinition",
    "EditType",
    "Source",
    "EDIT_TYPES",
    "BUCKETS",
    "TASKS_ROOT",
    "NUM_INFERENCE_STEPS",
    "HEIGHT",
    "WIDTH",
    "gen_kwargs",
    "load_tasks",
    "get_task",
]
