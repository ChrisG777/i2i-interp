"""ExperimentRunner ABC.

A subclass implements ``run_one(task)``; the base class provides:

* ``run_many(tasks)`` — sequential driver that frees CUDA cache between tasks
  and writes a top-level ``run_metadata.json`` once after the last task.
* ``task_dir(task)`` — per-task output dir. Two layouts:
  - **Default (legacy)**: ``<results_root>/<edit_type>/<task_id>/[<run_timestamp>/]``.
  - **Flat (`--results-subdir NAME` set)**: ``<results_root>/<NAME>/<task_id>/``.
    Used by the paper-scale runs; pairs naturally with ``--skip-if-completed``.
* ``write_task_metadata(task, extra)`` — emits ``task_metadata.json`` and
  ``cli_args.json`` per task.
* ``reference_image(task)`` — dispatches to ``load_or_make_reference``.
* ``mark_completed(task, files)`` / ``is_completed(task_id, expected_files)`` —
  append-only ``_completion.jsonl`` audit log under the (root, subdir) directory
  (``fcntl.flock`` for concurrent SLURM array tasks). Skip-eligibility is purely
  file-existence based: a task is considered completed iff every file in
  ``expected_files`` exists under ``task_dir``. The completion log is informational
  only; the disk is the source of truth so partial state (e.g. an existing
  variant done, a new variant missing) correctly re-runs only the missing parts.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image

from experiments.common.baselines import load_or_make_reference
from experiments.common.tasks import TaskDefinition


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "<unknown>"


class ExperimentRunner(ABC):
    """Base class for per-experiment runners.

    Subclasses set:
        ``name`` — short slug used in ``run_metadata.json`` (e.g.
        ``"attention_knockout"``).
        ``results_root`` — output root, e.g. ``"results/attention_knockout"``.
    """

    name: str
    results_root: str

    def __init__(self, model, *, extra_args: argparse.Namespace | None = None):
        self.model = model
        self.extra_args = extra_args if extra_args is not None else argparse.Namespace()
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.git_sha = _git_sha()
        self._results_root_override: str | None = getattr(
            self.extra_args, "results_root", None,
        )
        self._results_subdir: str | None = getattr(
            self.extra_args, "results_subdir", None,
        )
        self._skip_if_completed: bool = bool(
            getattr(self.extra_args, "skip_if_completed", False),
        )
        self._no_timestamp: bool = bool(
            getattr(self.extra_args, "no_timestamp", False),
        )
        if self._skip_if_completed:
            assert self._results_subdir is not None, (
                "--skip-if-completed requires --results-subdir"
            )
            self._no_timestamp = True
        if self._results_subdir is not None:
            # Flat layout always implies no timestamp; the subdir IS the
            # stable home for re-runs.
            self._no_timestamp = True

    # ------------------------------------------------------------------
    # Subclass entry points
    # ------------------------------------------------------------------

    @abstractmethod
    def run_one(self, task: TaskDefinition) -> Path:
        """Run the experiment on a single task; return the per-task output dir."""

    def expected_artifacts(self, task: TaskDefinition) -> list[str]:
        """List of files (relative to ``task_dir(task)``) that the current
        invocation will produce. Override in subclasses to enable
        artifact-aware ``--skip-if-completed`` (skip iff every file already
        exists). Returning ``[]`` disables the skip for this runner.
        """
        return []

    def run_many(self, tasks: list[TaskDefinition]) -> list[Path]:
        """Run a list of tasks sequentially.

        With ``--skip-if-completed`` set, tasks whose every expected artifact
        already exists on disk are skipped (no model load, no work, no metadata
        rewrite). Adding a new variant to the CLI flags makes ``expected_artifacts``
        return a superset, so tasks missing the new variant are correctly re-run.
        """
        assert len(tasks) > 0, "run_many: empty task list"
        out_dirs: list[Path] = []
        kept: list[TaskDefinition] = []
        for i, task in enumerate(tasks):
            expected = self.expected_artifacts(task)
            if self._skip_if_completed and self.is_completed(task.task_id, expected):
                print(f"\n[{i + 1}/{len(tasks)}] skip (completed): {task.task_id}")
                continue
            print(f"\n{'=' * 60}\n[{i + 1}/{len(tasks)}] task={task.task_id}\n{'=' * 60}")
            out_dir = self.run_one(task)
            out_dirs.append(out_dir)
            kept.append(task)
            torch.cuda.empty_cache()
        if kept:
            self._write_run_metadata(kept, out_dirs)
        return out_dirs

    # ------------------------------------------------------------------
    # Helpers exposed to subclasses
    # ------------------------------------------------------------------

    @property
    def is_flat_layout(self) -> bool:
        """True when ``--results-subdir`` is set; subclasses use this to drop
        nested sweep/setting/category subdirs."""
        return self._results_subdir is not None

    def setting_root(self) -> Path:
        """``<results_root>/[<results_subdir>/]`` — the parent of all per-task
        dirs. Used for the completion log and for run metadata under the
        flat layout.

        Default root: ``results/<exp>`` (legacy). When ``--results-subdir`` is
        set without an explicit ``--results-root``, the root switches to
        ``results_v4/<exp>`` so paper-scale runs are isolated from the
        legacy-layout output. ``--results-root`` always wins if explicitly
        set; pass ``--results-root results_v3/<exp>`` to write back to v3.
        """
        if self._results_root_override is not None:
            root = Path(self._results_root_override)
        elif self._results_subdir is not None:
            root = Path("results_v4") / self.name
        else:
            root = Path(self.results_root)
        if self._results_subdir is not None:
            root = root / self._results_subdir
        return root

    def task_dir(self, task: TaskDefinition) -> Path:
        if self.is_flat_layout:
            # Flat: <root>/<subdir>/<task_id>/  (no edit_type, no timestamp).
            return self.setting_root() / task.task_id
        out = self.setting_root() / task.edit_type / task.task_id
        if not self._no_timestamp:
            out = out / self.run_timestamp
        return out

    def reference_image(self, task: TaskDefinition) -> Image.Image:
        return load_or_make_reference(self.model, task)

    def write_task_metadata(self, task: TaskDefinition, extra: dict[str, Any]) -> Path:
        out = self.task_dir(task)
        out.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {
            "task_id": task.task_id,
            "edit_type": task.edit_type,
            "source": task.source,
            "instruction": task.instruction,
            "source_image_path": task.source_image_path,
            "source_caption": task.source_caption,
            "real_ref_name": task.real_ref_name,
            "ref_seed": task.ref_seed,
            "noise_seed": task.noise_seed,
            "height": task.height,
            "width": task.width,
            "task_metadata": task.metadata,
            "experiment": self.name,
            "model": self.model.name,
            "run_timestamp": self.run_timestamp,
            "git_sha": self.git_sha,
            **extra,
        }
        path = out / "task_metadata.json"
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        cli_path = out / "cli_args.json"
        with open(cli_path, "w") as f:
            json.dump(vars(self.extra_args), f, indent=2, default=str)
        return path

    # ------------------------------------------------------------------
    # Completion log
    # ------------------------------------------------------------------

    def completion_log(self) -> Path:
        """Path to the per-(root, subdir) JSONL completion log. Append-only
        audit trail; not consulted for skip decisions (those go via file
        existence under ``task_dir``)."""
        return self.setting_root() / "_completion.jsonl"

    def is_completed(
        self,
        task_id: str,
        expected_files: list[str | Path],
    ) -> bool:
        """Skip-eligible iff every ``expected_files`` entry exists under the
        flat-layout ``task_dir`` for ``task_id``. ``expected_files=[]`` returns
        False (no skip). Falls back to False outside the flat layout."""
        if self._results_subdir is None:
            return False
        if not expected_files:
            return False
        out = self.setting_root() / task_id
        return all((out / Path(f)).exists() for f in expected_files)

    def mark_completed(self, task_id: str, files: Iterable[str | Path]) -> None:
        """Append a JSON record to the completion log under flock. Subclasses
        call this at the end of ``run_one`` (or ``run_pair``) once all
        per-task output files are on disk."""
        if self._results_subdir is None:
            return
        path = self.completion_log()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "task_id": task_id,
            "git_sha": self.git_sha,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "files": [str(f) for f in files],
        }
        line = json.dumps(record, default=str) + "\n"
        with open(path, "a") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(line)
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_run_metadata(
        self,
        tasks: list[TaskDefinition],
        out_dirs: list[Path],
    ) -> Path:
        # Under flat layout, write run metadata next to the completion log
        # rather than under <root>/_runs/<ts>/, so all bookkeeping lives in
        # the setting dir.
        if self.is_flat_layout:
            meta_dir = self.setting_root() / "_runs"
        else:
            root = self._results_root_override or self.results_root
            meta_dir = Path(root) / "_runs" / self.run_timestamp
        meta_dir.mkdir(parents=True, exist_ok=True)
        path = meta_dir / f"run_{self.run_timestamp}.json" if self.is_flat_layout \
            else meta_dir / "run_metadata.json"
        meta = {
            "experiment": self.name,
            "model": self.model.name,
            "run_timestamp": self.run_timestamp,
            "git_sha": self.git_sha,
            "n_tasks": len(tasks),
            "task_ids": [t.task_id for t in tasks],
            "task_dirs": [str(p) for p in out_dirs],
            "extra_args": vars(self.extra_args),
        }
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        return path
