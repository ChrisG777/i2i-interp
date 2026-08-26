"""Agreement report between two judge models' verdict CSVs.

Reads the per-judge CSVs written by ``scripts/run_judge.py`` for a baseline
model (default: the paper's Claude judge at ``vlm_judge/<judge>.csv``) and a
comparison model (default: ``gpt-5.6-terra`` at ``vlm_judge/<model>/``), and
prints per-judge pass rates, percent agreement on shared entity ids, Cohen's
kappa, and every disagreeing entity id with both reasons. No API calls.

Usage::

    uv run python scripts/compare_judges.py --model gpt-5.6-terra
    uv run python scripts/compare_judges.py --model gpt-5.6-terra --judge i2i2i_color
"""

from __future__ import annotations

import argparse

from scripts.judge.configs import PAPER_JUDGES, JUDGES, get
from scripts.judge.csv_io import load_existing_rows
from utils.vlm import DEFAULT_MODEL, MODELS


def _verdicts(rows: dict[str, dict]) -> dict[str, tuple[int, str]]:
    """entity_id -> (pass, reason), dropping error rows (empty ``pass``)."""
    return {
        eid: (int(r["pass"]), r.get("reason", ""))
        for eid, r in rows.items()
        if r.get("pass", "") in ("0", "1")
    }


def _kappa(pairs: list[tuple[int, int]]) -> float | None:
    """Cohen's kappa for paired binary verdicts; None when degenerate."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return None
    return (po - pe) / (1 - pe)


def compare_judge(cfg, baseline: str, model: str) -> dict | None:
    """Per-judge comparison stats, or None if the model CSV doesn't exist."""
    model_csv = cfg.csv_path_for(model)
    if not model_csv.exists():
        return None
    a = _verdicts(load_existing_rows(cfg.csv_path_for(baseline)))
    b = _verdicts(load_existing_rows(model_csv))
    shared = sorted(a.keys() & b.keys())
    pairs = [(a[eid][0], b[eid][0]) for eid in shared]
    disagreements = [
        (eid, a[eid], b[eid]) for eid in shared if a[eid][0] != b[eid][0]
    ]
    return {
        "n_baseline": len(a),
        "n_model": len(b),
        "n_shared": len(shared),
        "pass_rate_baseline": (
            sum(v for v, _ in a.values()) / len(a) if a else None
        ),
        "pass_rate_model": (
            sum(v for v, _ in b.values()) / len(b) if b else None
        ),
        "n_agree": len(shared) - len(disagreements),
        "kappa": _kappa(pairs),
        "disagreements": disagreements,
    }


def _fmt_rate(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:5.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", choices=sorted(MODELS), default="gpt-5.6-terra",
        help="Comparison model (reads vlm_judge/<model>/<judge>.csv).",
    )
    ap.add_argument(
        "--baseline", choices=sorted(MODELS), default=DEFAULT_MODEL,
        help="Baseline model (default: the paper's Claude judge).",
    )
    ap.add_argument(
        "--judge", action="append", default=None,
        choices=[j.name for j in JUDGES], metavar="NAME",
        help="Judge to compare (repeatable). Default: all PAPER_JUDGES "
        "whose comparison CSV exists.",
    )
    ap.add_argument(
        "--max-disagreements", type=int, default=None,
        help="Print at most N disagreeing entities per judge.",
    )
    args = ap.parse_args()
    assert args.model != args.baseline, "Comparison and baseline model are the same."

    if args.judge is not None:
        names = args.judge
        for name in names:
            cfg = get(name)
            assert cfg.csv_path_for(args.baseline).exists(), (
                f"{name}: no baseline CSV at {cfg.csv_path_for(args.baseline)}"
            )
    else:
        names = sorted(PAPER_JUDGES)

    total_shared = total_agree = 0
    skipped: list[str] = []
    for name in names:
        cfg = get(name)
        stats = compare_judge(cfg, args.baseline, args.model)
        if stats is None:
            skipped.append(name)
            continue
        total_shared += stats["n_shared"]
        total_agree += stats["n_agree"]
        agree_pct = (
            stats["n_agree"] / stats["n_shared"] if stats["n_shared"] else None
        )
        kappa = stats["kappa"]
        print(f"\n{name}")
        print(
            f"  graded: {args.baseline}={stats['n_baseline']} "
            f"{args.model}={stats['n_model']} shared={stats['n_shared']}"
        )
        print(
            f"  pass rate: {args.baseline}={_fmt_rate(stats['pass_rate_baseline'])} "
            f"{args.model}={_fmt_rate(stats['pass_rate_model'])}"
        )
        print(
            f"  agreement: {_fmt_rate(agree_pct)} "
            f"(kappa={'n/a' if kappa is None else f'{kappa:.3f}'})"
        )
        shown = stats["disagreements"]
        if args.max_disagreements is not None:
            shown = shown[: args.max_disagreements]
        for eid, (av, ar), (bv, br) in shown:
            print(f"  != {eid}")
            print(f"     {args.baseline}: pass={av} ({ar[:80]})")
            print(f"     {args.model}: pass={bv} ({br[:80]})")
        n_hidden = len(stats["disagreements"]) - len(shown)
        if n_hidden > 0:
            print(f"  ... {n_hidden} more disagreements not shown")

    print("\n== overall ==")
    if total_shared:
        print(
            f"micro-averaged agreement: {_fmt_rate(total_agree / total_shared)} "
            f"({total_agree}/{total_shared} shared verdicts)"
        )
    else:
        print("no shared verdicts found")
    if skipped:
        print(
            f"skipped (no {args.model} CSV yet): {', '.join(sorted(skipped))}"
        )


if __name__ == "__main__":
    main()
