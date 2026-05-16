"""v4 results status + consistency tool.

Walks the JUDGES registry and on-disk SLURM scripts to surface the v4 plan's
state at a glance.

Per-judge columns:
  expected     = len(j.entity_ids())
  local-cell   = entity dirs on this machine containing the bundle's cell file
  judged       = rows in j.csv_path with pass in {0, 1}
  to-gen       = max(0, expected - local-cell)  - GPU runs still owed
  to-judge     = max(0, expected - judged)      - judge invocations still owed

NB: ``local-cell`` reflects this filesystem only; if ``judged > local-cell``
the artifacts exist on the cluster but haven't been rsynced down (the CSVs
are committed and travel with git).

Consistency check (bidirectional):
  MISSING SLURM   - judge has no SLURM script that would produce its cell
  ORPHAN SCRIPT   - SLURM script produces a cell with no consuming judge

Usage:
    uv run python scripts/v4_status.py
    uv run python scripts/v4_status.py --needs-judge
    uv run python scripts/v4_status.py --needs-gen
    uv run python scripts/v4_status.py --judge ko_color_ref_to_text_content
    uv run python scripts/v4_status.py --consistency-check
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.judge.configs import (
    JUDGE_GROUPS,
    JUDGES,
    JUDGES_BY_NAME,
    JudgeConfig,
)
from utils.flux2_klein import ALL_BLOCK_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_V4 = REPO_ROOT / "results_v4"
SLURM_ROOT = REPO_ROOT / "slurm"


# ---------------------------------------------------------------------------
# Per-judge status
# ---------------------------------------------------------------------------


@dataclass
class JudgeStatus:
    name: str
    expected: int
    local_cell: int
    judged: int
    csv_exists: bool

    @property
    def to_gen(self) -> int:
        """GPU runs still owed (cells missing from this filesystem)."""
        return max(0, self.expected - self.local_cell)

    @property
    def to_judge(self) -> int:
        """Judge invocations still owed (entities not yet scored)."""
        return max(0, self.expected - self.judged)


def _expected_cell_name(j: JudgeConfig) -> str:
    """Probe the bundle to extract the cell filename (last image path)."""
    bundle = j.bundle_builder("__PROBE__", j.base_dir)
    return bundle.image_paths[-1].name


def _csv_judged_count(csv_path: Path) -> tuple[int, bool]:
    if not csv_path.exists():
        return 0, False
    n = 0
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            v = (row.get("pass") or "").strip()
            if v in ("0", "1", "0.0", "1.0"):
                n += 1
    return n, True


def status_for(j: JudgeConfig) -> JudgeStatus:
    eids = j.entity_ids()
    expected = len(eids)
    cell = _expected_cell_name(j)
    local_cell = sum(1 for eid in eids if (j.base_dir / eid / cell).exists())
    judged, csv_exists = _csv_judged_count(j.csv_path)
    return JudgeStatus(
        name=j.name, expected=expected, local_cell=local_cell,
        judged=judged, csv_exists=csv_exists,
    )


def render_status(rows: list[JudgeStatus]) -> str:
    if not rows:
        return "(no judges to show)"
    nw = max(len(r.name) for r in rows)
    cols = ("expected", "local-cell", "judged", "to-gen", "to-judge")
    header = f"{'judge':<{nw}}  " + "  ".join(f"{c:>10}" for c in cols)
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.name:<{nw}}  "
            f"{r.expected:>10}  {r.local_cell:>10}  "
            f"{r.judged:>10}  {r.to_gen:>10}  {r.to_judge:>10}"
        )
    total_g = sum(r.to_gen for r in rows)
    total_j = sum(r.to_judge for r in rows)
    lines.append("-" * len(header))
    lines.append(f"TOTAL  to-gen: {total_g}    to-judge: {total_j}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------

# Match a quoted-string list following ``--settings`` (KO) or a whitespace-
# separated word list following ``--text-token-mode`` (i2i2i / i2i_unc).
_KO_SETTINGS_RE = re.compile(r"--settings\s+((?:'[^']+'\s*)+)")
_TEXT_MODE_RE = re.compile(r"--text-token-mode\s+([\w]+(?:\s+[\w]+)*)")
_RESULTS_SUBDIR_RE = re.compile(r"--results-subdir\s+(\S+)")
_BLOCK_RANGE_RE = re.compile(r"--block-range\s+(\d+)\s+(\d+)")


def _ko_setting_to_cell(setting_name: str) -> str:
    """'ref->text[padding]' -> 'ref_to_text_padding_full_ko.png'."""
    slug = setting_name.replace("->", "_to_")
    slug = re.sub(r"\[(\w+)\]", r"_\1", slug)
    return f"{slug}_full_ko.png"


_I2I2I_MODE_TO_CELL = {
    "all": "patched.png",
    "padding_only": "patched_text_padding.png",
    "content_only": "patched_text_content.png",
}


def _i2i_unc_block_suffix(block_idx: int) -> str:
    block_name = ALL_BLOCK_NAMES[block_idx]
    return block_name.replace(
        "transformer_blocks.", "mm",
    ).replace("single_transformer_blocks.", "single")


def _i2i_unc_mode_to_cell(mode: str, block_idx: int) -> str | None:
    suffix = _i2i_unc_block_suffix(block_idx)
    if mode == "all":
        return f"patched_{suffix}.png"
    if mode == "padding_only":
        return f"patched_{suffix}_text_padding.png"
    if mode == "content_only":
        return f"patched_{suffix}_text_content.png"
    return None


def parse_slurm_script(path: Path) -> list[tuple[str, str, str]]:
    """Return ``(experiment, results_subdir, cell_filename)`` triples this
    script will produce. Empty list if the script has no ``--results-subdir``
    or doesn't match a known experiment."""
    raw = path.read_text()
    # Normalize shell line continuations so single-flag lists that span
    # multiple lines are recoverable by the regexes.
    text = raw.replace("\\\n", " ")
    rel_parts = path.relative_to(SLURM_ROOT).parts
    if not rel_parts:
        return []
    exp = rel_parts[0]
    sub_match = _RESULTS_SUBDIR_RE.search(text)
    if not sub_match:
        return []
    subdir = sub_match.group(1)
    triples: list[tuple[str, str, str]] = []
    if exp == "attention_knockout":
        m = _KO_SETTINGS_RE.search(text)
        if not m:
            return []
        for name in re.findall(r"'([^']+)'", m.group(1)):
            triples.append((exp, subdir, _ko_setting_to_cell(name)))
    elif exp == "i2i_to_i2i_patching":
        m = _TEXT_MODE_RE.search(text)
        modes = m.group(1).split() if m else ["all"]
        for mode in modes:
            cell = _I2I2I_MODE_TO_CELL.get(mode)
            if cell:
                triples.append((exp, subdir, cell))
    elif exp == "i2i_to_unconditional":
        br = _BLOCK_RANGE_RE.search(text)
        if not br:
            return []
        block_idx = int(br.group(1))
        m = _TEXT_MODE_RE.search(text)
        modes = m.group(1).split() if m else ["all"]
        for mode in modes:
            cell = _i2i_unc_mode_to_cell(mode, block_idx)
            if cell:
                triples.append((exp, subdir, cell))
    return triples


def consistency_check() -> int:
    # Judge side: (exp, subdir, cell) -> [judge_names]
    judge_triples: dict[tuple[str, str, str], list[str]] = {}
    for j in JUDGES:
        rel = j.base_dir.relative_to(RESULTS_V4)
        parts = rel.parts
        if len(parts) < 2:
            continue
        exp = parts[0]
        subdir = "/".join(parts[1:])
        cell = _expected_cell_name(j)
        judge_triples.setdefault((exp, subdir, cell), []).append(j.name)

    # SLURM side: (exp, subdir, cell) -> [script_paths]
    slurm_triples: dict[tuple[str, str, str], list[str]] = {}
    for sh in sorted(SLURM_ROOT.rglob("*.sh")):
        for triple in parse_slurm_script(sh):
            slurm_triples.setdefault(triple, []).append(
                str(sh.relative_to(REPO_ROOT))
            )

    missing_slurm = sorted(set(judge_triples) - set(slurm_triples))
    orphan_script = sorted(set(slurm_triples) - set(judge_triples))

    rc = 0
    if missing_slurm:
        rc = 1
        print("MISSING SLURM (judge has no producer):")
        for t in missing_slurm:
            judges = ", ".join(judge_triples[t])
            print(f"  {t[0]}/{t[1]}/{t[2]}  <- {judges}")
    if orphan_script:
        rc = 1
        if missing_slurm:
            print()
        print("ORPHAN SCRIPT (script produces cell with no consumer):")
        for t in orphan_script:
            print(f"  {t[0]}/{t[1]}/{t[2]}")
            for sp in slurm_triples[t]:
                print(f"      {sp}")

    # Group-coverage check: a judge may live in zero or one JUDGE_GROUPS
    # entry (groups exist purely to batch sibling judges that share a leading
    # image prefix for the Anthropic prompt cache; standalone judges with no
    # cacheable siblings legitimately belong to no group). No group may list
    # a judge that isn't in JUDGES, and no judge may appear in two groups.
    judge_names = {j.name for j in JUDGES}
    seen: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    orphan_members: list[tuple[str, str]] = []
    for g in JUDGE_GROUPS:
        for m in g.members:
            if m not in judge_names:
                orphan_members.append((g.name, m))
            if m in seen:
                duplicates.append((m, seen[m], g.name))
            else:
                seen[m] = g.name

    if orphan_members:
        rc = 1
        if missing_slurm or orphan_script:
            print()
        print("ORPHAN GROUP MEMBER (group references unknown judge):")
        for group_name, member in orphan_members:
            print(f"  {group_name} -> {member}")
    if duplicates:
        rc = 1
        if missing_slurm or orphan_script or orphan_members:
            print()
        print("DUPLICATE GROUP MEMBER (judge in multiple groups):")
        for member, g1, g2 in duplicates:
            print(f"  {member}  in  {g1}  AND  {g2}")

    if not (missing_slurm or orphan_script or orphan_members or duplicates):
        print(
            "Consistency check OK: every judge has a SLURM producer, every "
            "script-produced cell has a consuming judge, no judge appears in "
            "two cache groups, and every cache group references known judges."
        )
    return rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--needs-judge", action="store_true",
        help="Only show judges with to-judge > 0 (CSV-driven; works "
             "everywhere since CSVs travel with git).",
    )
    ap.add_argument(
        "--needs-gen", action="store_true",
        help="Only show judges with to-gen > 0 (filesystem-driven; on a "
             "laptop without synced results, this matches every judge).",
    )
    ap.add_argument(
        "--judge", default=None, metavar="NAME",
        help="Restrict to one judge.",
    )
    ap.add_argument(
        "--consistency-check", action="store_true",
        help="Bidirectional SLURM <-> JUDGES consistency check.",
    )
    args = ap.parse_args()

    if args.consistency_check:
        return consistency_check()

    if args.judge is not None:
        if args.judge not in JUDGES_BY_NAME:
            print(f"unknown judge {args.judge!r}", file=sys.stderr)
            return 2
        judges = [JUDGES_BY_NAME[args.judge]]
    else:
        judges = list(JUDGES)
    rows = [status_for(j) for j in judges]
    if args.needs_judge or args.needs_gen:
        rows = [
            r for r in rows
            if (args.needs_judge and r.to_judge)
            or (args.needs_gen and r.to_gen)
        ]
        if not rows:
            print("All judges complete (no work owed).")
            return 0
    print(render_status(rows))
    incomplete = any(r.to_gen or r.to_judge for r in rows)
    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
