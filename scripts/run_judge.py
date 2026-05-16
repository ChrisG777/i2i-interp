"""Generic VLM-as-judge orchestrator for the paper-scale run.

Each judge config in :mod:`scripts.judge.configs` defines: where to find the
per-entity result dirs, which entity-ids to iterate, what bundle (image
labels / paths / question) to send the VLM, and where to write the CSV.

Resume semantics: rows already in the CSV with a non-empty ``pass`` field
are skipped. Re-running is safe.

Two modes:
  --judge NAME   one judge at a time (legacy, useful for re-runs).
  --group NAME   all sibling judges that share an image prefix, run
                 per-entity-id sequentially so calls 2..N hit Anthropic's
                 ephemeral prompt cache. Each member still writes to its own
                 CSV with identical wording, columns, and per-row content.

Usage::

    uv run python scripts/run_judge.py --judge ko_style_ref_to_image
    uv run python scripts/run_judge.py --group ko_color --concurrency 10
    uv run python scripts/run_judge.py --list
    uv run python scripts/run_judge.py --write-readme  # regenerate vlm_judge/README.md
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from scripts.judge import api, csv_io
from scripts.judge.configs import (
    JUDGE_DIR,
    JUDGE_GROUPS,
    JUDGES,
    get,
    get_group,
)
from utils import vlm

REPO_ROOT = Path(__file__).resolve().parent.parent


async def _judge_one(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    cfg,
    entity_id: str,
) -> dict:
    bundle = cfg.bundle_builder(entity_id, cfg.base_dir)
    async with sem:
        pass_, reason, in_tok, out_tok = await api.call_judge(
            client, bundle.image_labels, bundle.image_paths, bundle.question,
        )
    return csv_io.make_row(
        entity_id, pass_=pass_, reason=reason,
        in_tok=in_tok, out_tok=out_tok, model=api.MODEL_ID,
    )


async def _run_judge(
    client, cfg, *, concurrency: int, no_retry_errors: bool, limit: int | None,
) -> None:
    intended = cfg.entity_ids()
    on_disk = {p.name for p in cfg.base_dir.iterdir()} if cfg.base_dir.exists() else set()
    todo = [eid for eid in intended if eid in on_disk]
    skipped_missing = [eid for eid in intended if eid not in on_disk]
    print(
        f"{cfg.name}: {len(todo)} entities present on disk "
        f"({len(skipped_missing)} not yet generated)"
    )

    existing = csv_io.load_existing_rows(cfg.csv_path)
    todo = [
        eid for eid in todo
        if not csv_io.should_skip(existing.get(eid), no_retry_errors)
    ]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print(f"{cfg.name}: nothing to judge (all already verdicted)")
        return

    sem = asyncio.Semaphore(concurrency)
    coros = [_judge_one(client, sem, cfg, eid) for eid in todo]
    print(f"{cfg.name}: judging {len(todo)} entities (concurrency={concurrency})")
    n_pass = n_fail = n_err = 0
    for fut in asyncio.as_completed(coros):
        row = await fut
        csv_io.append_row(cfg.csv_path, row)
        verdict = row["pass"]
        if verdict == "":
            n_err += 1
        elif int(verdict) == 1:
            n_pass += 1
        else:
            n_fail += 1
        print(f"  {row['entity_id']}: pass={verdict} ({row['reason'][:60]})")
    print(f"{cfg.name}: pass={n_pass} fail={n_fail} err={n_err}")


async def _judge_one_entity_group(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    members: list,
    todo_per_judge: dict[str, set[str]],
    prefix_len: int,
    entity_id: str,
) -> list[tuple, ...]:
    """Run every member whose ``todo_per_judge[member.name]`` contains
    ``entity_id``, sequentially, so calls 2..N hit the warm cache. Returns
    a list of ``(judge_config, row)`` to append after release.

    Concurrency is on the OUTER entity loop only; within one entity we never
    parallelize sibling calls (the second sibling needs the first's cache
    write to land before it fires).
    """
    out: list[tuple] = []
    async with sem:
        for cfg in members:
            if entity_id not in todo_per_judge[cfg.name]:
                continue
            bundle = cfg.bundle_builder(entity_id, cfg.base_dir)
            pass_, reason, in_tok, out_tok = await api.call_judge(
                client,
                bundle.image_labels,
                bundle.image_paths,
                bundle.question,
                cache_prefix_len=prefix_len,
            )
            row = csv_io.make_row(
                entity_id, pass_=pass_, reason=reason,
                in_tok=in_tok, out_tok=out_tok, model=api.MODEL_ID,
            )
            out.append((cfg, row))
    return out


async def _run_group(
    client, group, *, concurrency: int, no_retry_errors: bool, limit: int | None,
) -> None:
    members = [get(name) for name in group.members]

    # Per-judge resume + on-disk filter — identical to single-judge mode,
    # applied independently to each member. A given entity may need 1..N
    # member calls depending on which CSV rows are already filled.
    todo_per_judge: dict[str, set[str]] = {}
    all_eids: set[str] = set()
    for cfg in members:
        intended = cfg.entity_ids()
        on_disk = (
            {p.name for p in cfg.base_dir.iterdir()}
            if cfg.base_dir.exists() else set()
        )
        existing = csv_io.load_existing_rows(cfg.csv_path)
        todo = {
            eid for eid in intended
            if eid in on_disk
            and not csv_io.should_skip(existing.get(eid), no_retry_errors)
        }
        todo_per_judge[cfg.name] = todo
        all_eids.update(todo)
        print(
            f"{cfg.name}: {len(todo)} entities to judge "
            f"(of {len(intended)} intended)"
        )

    entity_ids = sorted(all_eids)
    if limit is not None:
        entity_ids = entity_ids[:limit]
    if not entity_ids:
        print(f"group {group.name}: nothing to judge")
        return

    total_calls = sum(
        1 for eid in entity_ids
        for cfg in members if eid in todo_per_judge[cfg.name]
    )
    print(
        f"group {group.name}: {len(entity_ids)} entities, "
        f"{total_calls} API calls (concurrency={concurrency}, prefix_len={group.prefix_len})"
    )

    sem = asyncio.Semaphore(concurrency)
    coros = [
        _judge_one_entity_group(
            client, sem, members, todo_per_judge, group.prefix_len, eid,
        )
        for eid in entity_ids
    ]

    tallies = {cfg.name: {"pass": 0, "fail": 0, "err": 0} for cfg in members}
    for fut in asyncio.as_completed(coros):
        rows = await fut
        for cfg, row in rows:
            csv_io.append_row(cfg.csv_path, row)
            verdict = row["pass"]
            if verdict == "":
                tallies[cfg.name]["err"] += 1
            elif int(verdict) == 1:
                tallies[cfg.name]["pass"] += 1
            else:
                tallies[cfg.name]["fail"] += 1
            print(
                f"  [{cfg.name}] {row['entity_id']}: pass={verdict} "
                f"({row['reason'][:60]})"
            )

    print(f"\ngroup {group.name} summary:")
    for cfg in members:
        t = tallies[cfg.name]
        print(
            f"  {cfg.name}: pass={t['pass']} fail={t['fail']} err={t['err']}"
        )


def _list_judges() -> None:
    name_w = max(len(j.name) for j in JUDGES)
    for j in JUDGES:
        group_name = next(
            (g.name for g in JUDGE_GROUPS if j.name in g.members), "—",
        )
        print(f"{j.name:<{name_w}}  [{group_name}]  {j.description}")
    print()
    print("Groups:")
    gw = max(len(g.name) for g in JUDGE_GROUPS)
    for g in JUDGE_GROUPS:
        print(f"  {g.name:<{gw}}  {len(g.members)} judges, prefix_len={g.prefix_len}")


def _write_readme() -> None:
    """Emit ``results_v4/vlm_judge/README.md`` summarizing every judge.

    The README is the single source of truth for what each judge asks the
    VLM. Any change to ``bundles.py`` should be followed by re-running
    ``--write-readme`` so reviewers can compare prompts.
    """
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    out = JUDGE_DIR / "README.md"
    lines = [
        "<!-- Generated by `uv run python -m scripts.run_judge --write-readme`. Do not edit by hand. -->",
        "# VLM judges for the paper-scale run",
        "",
        "Each judge resolves an entity to a labeled image bundle, sends one cached system message ([`scripts/judge/api.py::SYSTEM_PROMPT`](../../scripts/judge/api.py)) plus a per-bundle user message, and parses the VLM's single-line JSON reply `{\"pass\": 0|1, \"reason\": \"…\"}`. Per-cell pass rates in the paper tables are pooled directly from `results_v4/vlm_judge/<judge>.csv`.",
        "",
        "## System prompt",
        "",
        "```",
        api.SYSTEM_PROMPT,
        "```",
        "",
        f"## {len(JUDGES)} judges",
        "",
    ]
    for j in JUDGES:
        bundle = j.bundle_builder("<ENTITY_ID>", j.base_dir)
        lines.append(f"### `{j.name}`")
        lines.append("")
        lines.append(j.description)
        lines.append("")
        lines.append(f"- **CSV**: `{j.csv_path.relative_to(REPO_ROOT)}`")
        lines.append(f"- **Result dir**: `{j.base_dir.relative_to(REPO_ROOT)}/<ENTITY_ID>/`")
        lines.append("- **Bundle**:")
        for label, path in zip(bundle.image_labels, bundle.image_paths):
            rel = path.relative_to(j.base_dir)
            lines.append(f"  - {label} `{rel}`")
        lines.append("")
        lines.append("> " + bundle.question.replace("\n", "\n> "))
        lines.append("")
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--judge", choices=[j.name for j in JUDGES], default=None,
        help="Run one judge (legacy single-judge mode).",
    )
    ap.add_argument(
        "--group", choices=[g.name for g in JUDGE_GROUPS], default=None,
        help="Run a group of sibling judges per-entity (cache-friendly).",
    )
    ap.add_argument(
        "--all", action="store_true",
        help="Run every group, then every ungrouped judge, sequentially.",
    )
    ap.add_argument("--list", action="store_true", help="List all judges and exit.")
    ap.add_argument(
        "--write-readme", action="store_true",
        help="(Re)generate results_v4/vlm_judge/README.md and exit.",
    )
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--no-retry-errors", action="store_true",
        help="Skip entities whose previous CSV row was an error (empty pass).",
    )
    ap.add_argument(
        "--verbose-cache", action="store_true",
        help="Log Anthropic cache_read / cache_creation token counts to stderr.",
    )
    args = ap.parse_args()

    if args.list:
        _list_judges()
        return
    if args.write_readme:
        _write_readme()
        return
    n_modes = sum(x is not None and x is not False for x in (args.judge, args.group, args.all or None))
    assert n_modes == 1, (
        "Pass exactly one of --judge NAME, --group NAME, or --all "
        "(or --list / --write-readme)."
    )

    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env or export it.",
              file=sys.stderr)
        sys.exit(1)

    if args.verbose_cache:
        vlm.VERBOSE_CACHE = True

    client = anthropic.AsyncAnthropic()
    if args.judge is not None:
        cfg = get(args.judge)
        asyncio.run(_run_judge(
            client, cfg,
            concurrency=args.concurrency,
            no_retry_errors=args.no_retry_errors,
            limit=args.limit,
        ))
    elif args.group is not None:
        group = get_group(args.group)
        asyncio.run(_run_group(
            client, group,
            concurrency=args.concurrency,
            no_retry_errors=args.no_retry_errors,
            limit=args.limit,
        ))
    else:
        # --all: run every group (cache-friendly), then every ungrouped judge.
        grouped_names = {name for g in JUDGE_GROUPS for name in g.members}
        for group in JUDGE_GROUPS:
            print(f"\n=== group: {group.name} ({len(group.members)} judges) ===")
            asyncio.run(_run_group(
                client, group,
                concurrency=args.concurrency,
                no_retry_errors=args.no_retry_errors,
                limit=args.limit,
            ))
        for cfg in JUDGES:
            if cfg.name in grouped_names:
                continue
            print(f"\n=== judge: {cfg.name} ===")
            asyncio.run(_run_judge(
                client, cfg,
                concurrency=args.concurrency,
                no_retry_errors=args.no_retry_errors,
                limit=args.limit,
            ))


if __name__ == "__main__":
    main()
