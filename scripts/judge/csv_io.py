"""Append-only CSV layer for paper-scale judge results.

CSVs live under ``results_v2/vlm_judge/<judge_name>.csv``. On resume,
``load_existing_rows`` keeps the *last* occurrence per ``entity_id`` so a
re-run that re-attempts an errored row simply appends a new row and the
later read returns the new verdict.

``entity_id`` is generic — for per-task judges it's the ``task_id``; for
i2i->i2i pair judges it's the ``<source>__<target>`` pair id.
"""

from __future__ import annotations

import csv
from pathlib import Path

CSV_COLUMNS = [
    "entity_id", "pass", "reason", "model", "input_tokens", "output_tokens",
]


def load_existing_rows(csv_path: Path) -> dict[str, dict]:
    """Map ``entity_id`` -> last-written row (dict-overwrite semantics)."""
    if not csv_path.exists():
        return {}
    with csv_path.open() as f:
        return {row["entity_id"]: row for row in csv.DictReader(f)}


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def make_row(
    entity_id: str,
    *,
    pass_: int | None,
    reason: str,
    in_tok: int,
    out_tok: int,
    model: str,
) -> dict:
    pass_int = None if pass_ is None else int(pass_)
    return {
        "entity_id": entity_id,
        "pass": "" if pass_int is None else pass_int,
        "reason": reason,
        "model": model,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }


def dedupe_rows(rows: list[dict]) -> list[dict]:
    out: dict[str, dict] = {}
    for r in rows:
        out[r["entity_id"]] = r
    return list(out.values())


def should_skip(prev: dict | None, no_retry_errors: bool) -> bool:
    if prev is None:
        return False
    if prev.get("pass", "") != "":
        return True
    return no_retry_errors
