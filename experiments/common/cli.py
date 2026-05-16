"""Shared CLI flags for experiment scripts.

Three task-selection modes (mutually exclusive, one required):

* ``--task-id ID [ID ...]`` — manual inspection of one or more named tasks.
  Tasks are resolved by their full ID (e.g. ``manual_solid_yellow_couch``,
  ``solid_red_couch``). See :func:`experiments.common.tasks.get_task`.

* ``--edit-type {add,remove,customize} [{...} ...]`` — batch sweep.
  Loads every task in the matching dataset bucket plus the manual bucket
  filtered by edit_type. ``--limit N`` caps tasks **per edit-type**, so
  ``--edit-type add remove --limit 50`` gives up to 100 tasks total.

* ``--bucket NAME`` — load every task in the named bucket as-is. Useful for
  buckets like ``solid_color`` whose tasks all share one edit_type. Combine
  with ``--shard-index I --shard-total N`` for SLURM array sharding.

``--source`` (sun397 / dreambench_plus / manual / property_manual) is
intentionally NOT exposed: not all (source, edit_type) combinations are
legal, and the loader handles bucketing internally.

Output overrides (via :func:`add_output_overrides`):

* ``--results-root PATH`` — replaces the runner's default ``results/<exp>``
  prefix.
* ``--no-timestamp`` — drops the ``<run_timestamp>`` segment from per-task
  output dirs.
"""

from __future__ import annotations

import argparse
from math import ceil

from experiments.common.tasks import BUCKETS, EDIT_TYPES, TaskDefinition, get_task, load_tasks


def add_task_selection(parser: argparse.ArgumentParser) -> None:
    """Add task-selection flags (``--task-id`` / ``--edit-type`` / ``--bucket``)
    plus ``--limit`` and ``--shard-index`` / ``--shard-total`` to ``parser``."""
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--task-id", type=str, nargs="+", default=None, metavar="ID",
        help="One or more task IDs (or legacy short names). Mutually exclusive "
             "with --edit-type and --bucket.",
    )
    group.add_argument(
        "--edit-type", type=str, nargs="+", default=None,
        choices=list(EDIT_TYPES),
        help="One or more edit types; loads every task in those buckets "
             "(dataset bucket + manual bucket filtered to that edit_type). "
             "Mutually exclusive with --task-id and --bucket.",
    )
    group.add_argument(
        "--bucket", type=str, default=None,
        choices=list(BUCKETS),
        help="Name of a single bucket to load every task from. Mutually "
             "exclusive with --task-id and --edit-type.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Cap tasks. With --edit-type: per edit-type. With --bucket: "
             "total. Ignored with --task-id.",
    )
    parser.add_argument(
        "--shard-index", type=int, default=None, metavar="I",
        help="0-indexed shard number for SLURM array sharding. Slices the "
             "resolved task list into --shard-total contiguous chunks and "
             "returns chunk I. Requires --shard-total.",
    )
    parser.add_argument(
        "--shard-total", type=int, default=None, metavar="N",
        help="Total number of shards for --shard-index. Requires --shard-index.",
    )


def add_output_overrides(parser: argparse.ArgumentParser) -> None:
    """Add output-layout flags to ``parser``.

    Picked up by :class:`experiments.common.runner.ExperimentRunner`:

    * ``--results-root`` overrides the runner's default output prefix.
    * ``--no-timestamp`` drops the per-run timestamp segment so re-runs
      overwrite the same directory.
    * ``--results-subdir <name>`` inserts ``<name>`` between the experiment
      root and ``<task_id>``, and switches the runner into the
      large-scale flat layout (no per-edit_type dir, no nested
      sweep/setting/category subdirs, names encode the variant).
    * ``--skip-if-completed`` skips any task already recorded in the
      ``_completion.jsonl`` file under the resolved (root, subdir). Implies
      ``--no-timestamp`` and requires ``--results-subdir``.
    """
    parser.add_argument(
        "--results-root", type=str, default=None, metavar="PATH",
        help="Override the runner's default output root (e.g. "
             "results/attention_knockout) with PATH.",
    )
    parser.add_argument(
        "--no-timestamp", action="store_true",
        help="Drop the <run_timestamp> segment from per-task output dirs.",
    )
    parser.add_argument(
        "--results-subdir", type=str, default=None, metavar="NAME",
        help="Insert NAME between the results root and <task_id>/, and "
             "switch the runner into a flat per-task layout (no edit_type "
             "subdir, no nested sweep/setting/category dirs).",
    )
    parser.add_argument(
        "--skip-if-completed", action="store_true",
        help="Skip tasks already recorded in the per-(root, subdir) "
             "_completion.jsonl. Implies --no-timestamp; requires "
             "--results-subdir.",
    )


def _shard_slice(tasks: list[TaskDefinition], index: int, total: int) -> list[TaskDefinition]:
    assert total >= 1, f"--shard-total must be >= 1, got {total}"
    assert 0 <= index < total, (
        f"--shard-index must be in [0, --shard-total={total}), got {index}"
    )
    n = len(tasks)
    start = ceil(n * index / total)
    end = ceil(n * (index + 1) / total)
    return tasks[start:end]


def resolve_tasks(args: argparse.Namespace) -> list[TaskDefinition]:
    """Resolve tasks from parsed args. Exactly one of ``--task-id`` /
    ``--edit-type`` / ``--bucket`` must be set. Applies ``--shard-index`` /
    ``--shard-total`` slicing at the end when both are provided.
    """
    if (args.shard_index is None) != (args.shard_total is None):
        raise SystemExit("--shard-index and --shard-total must be set together")

    if args.task_id is not None:
        tasks = [get_task(t) for t in args.task_id]
    elif args.bucket is not None:
        tasks = load_tasks(args.bucket, limit=args.limit)
    else:
        assert args.edit_type is not None, (
            "resolve_tasks: none of --task-id / --edit-type / --bucket was set"
        )
        tasks = []
        seen_ids: set[str] = set()
        for et in args.edit_type:
            et_count = 0
            # Bucket name == edit_type for the dataset buckets; manual holds
            # hand-built scenes whose edit_types vary, so we filter by edit_type.
            for bucket in (et, "manual"):
                for task in load_tasks(bucket):
                    if task.edit_type != et:
                        continue
                    if task.task_id in seen_ids:
                        continue
                    if args.limit is not None and et_count >= args.limit:
                        break
                    tasks.append(task)
                    seen_ids.add(task.task_id)
                    et_count += 1
                if args.limit is not None and et_count >= args.limit:
                    break

    if args.shard_index is not None:
        tasks = _shard_slice(tasks, args.shard_index, args.shard_total)

    return tasks
