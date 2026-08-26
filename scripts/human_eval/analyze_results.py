"""Score the MTurk human-validation study against the VLM judge verdicts.

``select_subset.py`` published 240 items across six MTurk projects; this reads
the downloaded batch results back and answers one question: **how often does a
majority of crowdworkers reach the same verdict the VLM judge did?**

The three joins that make that non-trivial, and why each exists:

* **Choice -> role.** A worker answers ``left`` / ``right`` / ``both`` /
  ``neither``, but the left/right order was randomised per item precisely so
  that a position habit cannot masquerade as a signal. The true role behind
  each slot lives in ``manifest.csv`` (``role_left`` / ``role_right``), so a
  raw choice is meaningless until it is resolved to ``baseline`` /
  ``intervention``. Everything downstream keys on the role.

* **Role -> verdict.** Each condition predicts a *different* answer, mirroring
  what its judge was asked (see the table in ``VERDICT_ANSWER``). A knockout
  ``ref->text`` pass means the property was destroyed, so the worker should
  pick the clean baseline; a ``ref->image`` pass means it survived, so the
  worker should say "both". Collapsing all four options to one bit per
  condition is what makes human and VLM comparable at all.

* **Verdict -> item.** ~3 workers rate each item, so the item-level human
  verdict is a majority vote. Padded HIT slots give some items 6 raters, i.e.
  an even count, i.e. a possible tie; ties are reported as unresolved and left
  out of the agreement numerator and denominator rather than broken by a coin
  flip.

Two deliberate methodological choices:

* **No kappa.** Chance-corrected agreement is not reported. The quantity of
  interest is the raw rate at which humans and the judge land on the same
  verdict, on these items, with these priors --- not a coefficient whose
  "chance" baseline depends on the marginal that the intervention itself moves.

* **The bootstrap resamples WORKERS, not items.** A worker contributes 16-20
  correlated ratings; treating those as independent draws understates the
  interval badly. Resampling whole workers (a cluster bootstrap) and recomputing
  the majority vote inside every resample propagates both the rater noise and
  the vote-flipping that noise causes. Seeds are sha256-derived ints, never
  ``hash()`` of a string, which is salted per interpreter.

Workers are screened on the check arms before any of this: those items have a
known-correct answer, so a worker who misses it repeatedly was not looking. Each
method builds that answer differently (see ``CHECK_ANSWER`` and the manifest's
``target_role``), and the arms are recognised by suffix, ``_check`` for the
original study and ``_paircheck`` for the paired redesign.

Outputs (under ``--out-dir``, default ``results_v4/human_eval/``):

* ``human_verdicts.csv``   -- one row per item: the majority verdict and the
                              full vote split.
* ``disagreements.csv``    -- every item where the human majority and the judge
                              disagree, with its three image names, for eyeballing.

Usage::

    uv run python -m scripts.human_eval.analyze_results --results ~/Downloads/mturk
    uv run python -m scripts.human_eval.analyze_results \\
        --results 'batches/Batch_*_results.csv' --bootstrap 20000
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HUMAN_EVAL = REPO_ROOT / "results_v4" / "human_eval"

CHOICES = ("left", "right", "both", "neither")
ROLES = ("baseline", "intervention", "both", "neither")

# Condition suffix -> the resolved answer that counts as a human "pass", i.e.
# the answer the judge's pass=1 predicts. Mirrors the judge prompts one for one:
#   _ref_to_text   knockout destroyed the property -> only the baseline has it
#   _ref_to_image  knockout left the property intact -> both images have it
#   _patched       the patch transferred the property -> the patched side has it
#   _check         attention check; the source pass's own output always carries
#                  the source property, so "intervention" is known-correct.
VERDICT_ANSWER = {
    "_ref_to_text": "baseline",
    "_ref_to_image": "both",
    "_patched": "intervention",
}

# The two check arms put their known-correct image on OPPOSITE sides, so a
# single "_check" rule would score one of them exactly backwards:
#   i2i2i  intervention = source_i2i_4step, the source pass's own output, which
#          always carries the source property   -> answer "intervention"
#   ko     intervention = t2i_clean_4step, generated with no reference at all,
#          so the reference-conditioned baseline is the one that matches
#                                              -> answer "baseline"
CHECK_ANSWER = {"ko": "baseline", "i2i2i": "intervention"}

# Answers on check items score the worker; they are never compared to a judge
# (select_subset leaves vlm_pass blank on those rows because the judge was only
# ever shown the *patched* image for that pair).
#
# Two suffixes, because the paired redesign added its own screening arm under a
# name that does NOT end in "_check": ``ko_<family>_paircheck``. Matching only
# "_check" silently classified those as real items, which disabled screening for
# the whole batch without failing anything. Kept identical to the tuple
# ``select_subset`` uses when it decides which rows get a blank ``vlm_pass``.
CHECK_SUFFIXES = ("_check", "_paircheck")


def seed_of(*parts: str) -> int:
    """Deterministic cross-process seed, identical to ``select_subset.seed_of``.
    Duplicated rather than imported so the analysis does not drag in PIL and the
    task tables just to hash three strings."""
    h = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


def target_role(m: dict) -> str:
    """The role whose selection counts as a human "pass" for this item.

    Taken from the manifest's own ``target_role`` column when the study that
    produced it wrote one. Deriving this from the condition *name* has been the
    source of every sign error in this pipeline --- the two ``_check`` arms put
    their known-good image on opposite sides, and the paired ``ref_to_image``
    redesign changed what a pass looks like without changing any name --- so a
    study now states the answer per item and the analysis reads it rather than
    re-deriving it. ``target_answer`` remains for manifests written before the
    column existed.
    """
    return m.get("target_role") or target_answer(m["condition"])


def target_answer(condition: str) -> str:
    """Legacy fallback: infer the passing answer from the condition name.

    Only correct for the pre-pairing study, where ``ref_to_image`` showed the
    knockout output beside its own baseline and both carrying the property was
    the prediction. Check arms are resolved by METHOD, not by the shared
    check suffix, because the two methods place their known-good image on
    opposite sides."""
    if condition.endswith(CHECK_SUFFIXES):
        method = condition.split("_", 1)[0]
        try:
            return CHECK_ANSWER[method]
        except KeyError:
            raise KeyError(f"no check mapping for method {method!r} "
                           f"(condition {condition!r})") from None
    for suffix, ans in VERDICT_ANSWER.items():
        if condition.endswith(suffix):
            return ans
    raise KeyError(f"no verdict mapping for condition {condition!r}")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {r["item_id"]: r for r in rows}


def find_results(spec: str) -> list[Path]:
    """``--results`` is either a directory of MTurk CSVs or a glob of them."""
    p = Path(spec).expanduser()
    if p.is_dir():
        return sorted(p.glob("*.csv"))
    return sorted(Path(q) for q in glob.glob(str(p)))


@dataclass
class Assignment:
    worker: str
    hit: str
    assignment: str
    status: str
    work_time: float
    source: str
    answers: list[tuple[str, str]] = field(default_factory=list)  # (item_id, choice)


def parse_task_answers(blob: str) -> dict[str, object]:
    """``crowd-form`` does NOT emit one CSV column per field. Every field in the
    form is serialised into a single ``Answer.taskAnswers`` cell holding a JSON
    *list* with one object in it::

        [{"q0_choice": {"both": true, "left": false,
                        "neither": false, "right": false},
          "q0_item_id": "11051495cfda", ...}]

    A radio group arrives as a boolean per option rather than the chosen string,
    so the answer is the single key that is true. Verified against a real MTurk
    export; the flat ``Answer.q0_choice`` column shape this script originally
    assumed does not exist."""
    blob = (blob or "").strip()
    if not blob:
        return {}
    data = json.loads(blob)
    if isinstance(data, list):
        merged: dict[str, object] = {}
        for part in data:
            if isinstance(part, dict):
                merged.update(part)
        return merged
    return data if isinstance(data, dict) else {}


def choice_of(value: object) -> str:
    """Resolve one ``q<k>_choice`` value to a single option string. A plain
    string passes through (older templates / native inputs outside crowd-form);
    a boolean dict resolves to its one true key. Anything with zero or several
    true keys is treated as unanswered rather than guessed at --- several true
    would mean mutual exclusion failed, which must never be silently averaged
    into a verdict."""
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, dict):
        on = [k for k, v in value.items() if v is True]
        return on[0].strip().lower() if len(on) == 1 else ""
    return ""


def read_assignments(paths: list[Path]) -> tuple[list[Assignment], dict[str, int]]:
    """One record per submitted assignment. ``per_hit`` differs across the six
    projects (16 / 18 / 20), so the question indices are discovered from each
    assignment's own answer payload instead of assumed."""
    out: list[Assignment] = []
    per_hit: dict[str, int] = {}
    for path in paths:
        with path.open(newline="") as f:
            rdr = csv.DictReader(f)
            cols = set(rdr.fieldnames or [])
            if "Answer.taskAnswers" not in cols:
                print(f"  ! {path.name}: no Answer.taskAnswers column, skipped")
                continue
            for row in rdr:
                ans = parse_task_answers(row.get("Answer.taskAnswers", ""))
                ks = sorted(int(m.group(1)) for k in ans
                            if (m := re.fullmatch(r"q(\d+)_choice", k)))
                if not ks:
                    continue
                per_hit[path.name] = max(per_hit.get(path.name, 0), len(ks))
                a = Assignment(
                    worker=row.get("WorkerId", "").strip(),
                    hit=row.get("HITId", "").strip(),
                    assignment=row.get("AssignmentId", "").strip(),
                    status=row.get("AssignmentStatus", "").strip(),
                    work_time=float(row.get("WorkTimeInSeconds") or 0),
                    source=path.name,
                )
                for k in ks:
                    a.answers.append((clean(str(ans.get(f"q{k}_item_id", ""))),
                                      choice_of(ans.get(f"q{k}_choice"))))
                out.append(a)
    return out, per_hit


def clean(v: str | None) -> str:
    """MTurk writes an unanswered crowd-form field as an empty string or the
    literal ``{}``; both mean "no answer", not an answer of "{}"."""
    v = (v or "").strip()
    return "" if v == "{}" else v


# ---------------------------------------------------------------------------
# Melt + resolve
# ---------------------------------------------------------------------------


@dataclass
class Rating:
    worker: str
    item_id: str
    choice: str          # left | right | both | neither
    role: str            # baseline | intervention | both | neither
    verdict: int         # 1 if role == the condition's predicted answer
    condition: str
    is_check: bool


def melt(assignments: list[Assignment], manifest: dict[str, dict],
         keep_rejected: bool) -> tuple[list[Rating], Counter]:
    """One row per (worker, item). Returns the ratings and a tally of everything
    that had to be discarded, which is printed rather than swallowed."""
    drops: Counter = Counter()
    seen: set[tuple[str, str]] = set()
    ratings: list[Rating] = []
    for a in assignments:
        if a.status == "Rejected" and not keep_rejected:
            drops["rejected assignment"] += 1
            drops["answers in rejected assignments"] += len(a.answers)
            continue
        for item_id, choice in a.answers:
            if not item_id:
                drops["blank item_id (short HIT / unfilled slot)"] += 1
                continue
            if not choice:
                drops["blank answer"] += 1
                continue
            if choice not in CHOICES:
                drops[f"unrecognised answer {choice!r}"] += 1
                continue
            m = manifest.get(item_id)
            if m is None:
                drops["item_id not in manifest"] += 1
                continue
            # A padded slot can put the same item in two HITs, so one worker can
            # legitimately be served the same item twice. Keep the first rating:
            # the second is not an independent judgement.
            if (a.worker, item_id) in seen:
                drops["repeat rating of same item by same worker"] += 1
                continue
            seen.add((a.worker, item_id))
            role = ({"left": m["role_left"], "right": m["role_right"]}
                    .get(choice, choice))
            cond = m["condition"]
            ratings.append(Rating(
                worker=a.worker, item_id=item_id, choice=choice, role=role,
                verdict=int(role == target_role(m)),
                condition=cond, is_check=cond.endswith(CHECK_SUFFIXES)))
    return ratings, drops


# ---------------------------------------------------------------------------
# Worker screening
# ---------------------------------------------------------------------------


@dataclass
class WorkerStat:
    worker: str
    n_items: int = 0
    n_checks: int = 0
    n_checks_ok: int = 0
    work_time: float = 0.0       # mean seconds per assignment
    n_assignments: int = 0
    dropped: bool = False
    maj_agree: int = 0           # agreement with the item-level majority
    maj_total: int = 0

    @property
    def check_acc(self) -> float | None:
        return self.n_checks_ok / self.n_checks if self.n_checks else None


def screen(ratings: list[Rating], assignments: list[Assignment],
           threshold: float, min_checks: int) -> dict[str, WorkerStat]:
    stats: dict[str, WorkerStat] = {}
    for r in ratings:
        w = stats.setdefault(r.worker, WorkerStat(r.worker))
        w.n_items += 1
        if r.is_check:
            w.n_checks += 1
            w.n_checks_ok += r.verdict
    times: dict[str, list[float]] = defaultdict(list)
    for a in assignments:
        if a.worker in stats:
            times[a.worker].append(a.work_time)
    for w, ts in times.items():
        stats[w].n_assignments = len(ts)
        stats[w].work_time = sum(ts) / len(ts) if ts else 0.0
    for w in stats.values():
        # A worker with too few checks has no measurable accuracy, so there is
        # nothing to fail: keep them rather than drop on one unlucky answer.
        w.dropped = (w.n_checks >= min_checks
                     and (w.check_acc or 0.0) < threshold)
    return stats


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------


@dataclass
class ItemResult:
    item_id: str
    row: dict                      # manifest row
    n_pass: int = 0
    n_fail: int = 0
    roles: Counter = field(default_factory=Counter)
    choices: Counter = field(default_factory=Counter)
    min_consensus: float = 0.0

    @property
    def n_raters(self) -> int:
        return self.n_pass + self.n_fail

    @property
    def consensus(self) -> float:
        """Share of raters who picked the same one of the FOUR options. This is
        stricter than the binary verdict: {baseline, neither, both} collapses to
        a 2-1 verdict majority even though the three raters answered three
        different questions' worth of things. 0.0 when nobody rated the item."""
        if not self.roles:
            return 0.0
        return max(self.roles.values()) / sum(self.roles.values())

    @property
    def verdict(self) -> int | None:
        """``None`` when unresolved. Two ways that happens:

        * A tie on the binary verdict. Impossible with 3 raters, but padded HIT
          slots hand some items 6, and a 3-3 split must not be silently rounded.
        * Below ``min_consensus`` on the four-way answer. With 3 raters and 4
          options a 1-1-1 split yields a "majority" of one vote, which is not a
          majority in any useful sense --- and in the colour data every such
          item turned out to be one where the raters simply disagreed with each
          other, not with the judge."""
        if self.min_consensus and self.consensus < self.min_consensus:
            return None
        if self.n_pass == self.n_fail:
            return None
        return int(self.n_pass > self.n_fail)

    @property
    def unanimous_verdict(self) -> bool:
        return self.n_raters > 0 and 0 in (self.n_pass, self.n_fail)

    @property
    def unanimous_role(self) -> bool:
        return self.n_raters > 0 and len(self.roles) == 1

    @property
    def vlm(self) -> int | None:
        v = self.row["vlm_pass"]
        return int(v) if v in ("0", "1") else None


def tally(ratings: list[Rating], manifest: dict[str, dict],
          min_consensus: float = 0.0) -> dict[str, ItemResult]:
    items = {i: ItemResult(i, m, min_consensus=min_consensus)
             for i, m in manifest.items()}
    for r in ratings:
        it = items[r.item_id]
        if r.verdict:
            it.n_pass += 1
        else:
            it.n_fail += 1
        it.roles[r.role] += 1
        it.choices[r.choice] += 1
    return items


# ---------------------------------------------------------------------------
# Bootstrap over workers
# ---------------------------------------------------------------------------


def bootstrap(ratings: list[Rating], items: dict[str, ItemResult],
              order: list[str], n_boot: int,
              min_consensus: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Cluster bootstrap: resample workers with replacement, replay all of a
    resampled worker's ratings, recompute every item's majority vote, and score
    the whole study again. Returns ``(agree, resolved)``, both ``n_boot`` x
    ``n_items`` boolean, so any subset of items (a condition, a family) can be
    aggregated afterwards without redoing the resampling.

    Only judge-comparable ratings enter: ``_check`` items have no ``vlm_pass``,
    so they can never contribute to an agreement rate.
    """
    idx = {i: k for k, i in enumerate(order)}
    # Items dropped by the consensus filter are excluded outright rather than
    # re-tested inside each resample: the resampler works on the binary verdict
    # and does not carry the four-way choice, so it cannot recompute consensus.
    # The interval is therefore conditional on the retained item set --- honest
    # for rater noise on those items, not for which items would survive under a
    # different draw.
    usable = [r for r in ratings
              if not r.is_check and items[r.item_id].vlm is not None
              and not (min_consensus
                       and items[r.item_id].consensus < min_consensus)]
    workers = sorted({r.worker for r in usable})
    n_items, n_w = len(order), len(workers)

    agree = np.zeros((n_boot, n_items), dtype=bool)
    resolved = np.zeros((n_boot, n_items), dtype=bool)
    if not usable or n_w == 0:
        return agree, resolved

    widx = {w: k for k, w in enumerate(workers)}
    r_item = np.array([idx[r.item_id] for r in usable])
    r_worker = np.array([widx[r.worker] for r in usable])
    r_pass = np.array([float(r.verdict) for r in usable])
    vlm = np.array([(items[i].vlm if items[i].vlm is not None else -1)
                    for i in order])
    has_vlm = vlm >= 0

    rng = np.random.default_rng(seed_of("human_eval", "agreement_bootstrap"))
    # Drawing worker multiplicities from a multinomial is exactly sampling
    # n_w workers with replacement, but vectorised: a worker drawn twice simply
    # weights all of their ratings by 2.
    mult = rng.multinomial(n_w, np.full(n_w, 1.0 / n_w), size=n_boot)
    for b in range(n_boot):
        wt = mult[b][r_worker].astype(float)
        n_p = np.bincount(r_item, weights=wt * r_pass, minlength=n_items)
        n_all = np.bincount(r_item, weights=wt, minlength=n_items)
        n_f = n_all - n_p
        res = (n_p != n_f) & has_vlm
        resolved[b] = res
        agree[b] = res & ((n_p > n_f).astype(int) == vlm)
    return agree, resolved


def ci_of(agree: np.ndarray, resolved: np.ndarray, mask: np.ndarray
          ) -> tuple[float, float] | None:
    num = agree[:, mask].sum(axis=1)
    den = resolved[:, mask].sum(axis=1)
    ok = den > 0
    if not ok.any():
        return None
    rates = num[ok] / den[ok]
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def rate(num: int, den: int) -> str:
    """Never a bare percentage: the denominators here are 16-24 items, where a
    single flipped item moves the number by 4-6 points."""
    if den == 0:
        return "        n/a"
    return f"{num:3d}/{den:<3d} = {100.0 * num / den:5.1f}%"


def table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (nothing to report)")
        return
    cells = [headers] + rows
    widths = [max(len(str(r[c])) for r in cells) for c in range(len(headers))]
    # First column left-aligned (names), the rest right-aligned (numbers).
    def line(r: list[str]) -> str:
        return "  ".join(str(v).ljust(w) if c == 0 else str(v).rjust(w)
                         for c, (v, w) in enumerate(zip(r, widths)))
    print(line(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        print(line(r))


def split_str(c: Counter) -> str:
    return " ".join(f"{k}={c[k]}" for k in ROLES if c[k])


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score MTurk human verdicts against the VLM judges.")
    ap.add_argument("--results", required=True,
                    help="directory of MTurk 'Download CSV' files, or a glob")
    ap.add_argument("--manifest", type=Path, default=HUMAN_EVAL / "manifest.csv")
    ap.add_argument("--out-dir", type=Path, default=HUMAN_EVAL)
    ap.add_argument("--check-threshold", type=float, default=0.75,
                    help="minimum accuracy on _check items to keep a worker")
    ap.add_argument("--min-checks", type=int, default=2,
                    help="a worker is only screened once they saw this many "
                         "_check items")
    ap.add_argument("--min-consensus", type=float, default=0.0,
                    help="drop an item unless this share of its raters chose "
                         "the SAME one of the four options; 0.5 makes a 1-1-1 "
                         "three-way split unresolved instead of letting one "
                         "vote decide a 'majority'")
    ap.add_argument("--bootstrap", type=int, default=10000,
                    help="worker-level bootstrap resamples for the 95%% CI")
    ap.add_argument("--base-url", default="",
                    help="prefix image filenames with this in the "
                         "disagreement listing")
    ap.add_argument("--no-write", action="store_true",
                    help="print the analysis without writing any CSV")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    paths = find_results(args.results)
    if not paths:
        sys.exit(f"no results CSVs matched {args.results!r}")

    print(f"Reading {len(paths)} results file(s):")
    assignments, per_hit = read_assignments(paths)
    for p in paths:
        n = sum(1 for a in assignments if a.source == p.name)
        print(f"  {p.name:44s} {n:4d} assignments x {per_hit.get(p.name, 0)} questions")
    print(f"  {'TOTAL':44s} {len(assignments):4d} assignments, "
          f"{len(manifest)} items in manifest")

    # ---- 1-3. melt, resolve to roles, map to verdicts --------------------
    ratings, drops = melt(assignments, manifest, keep_rejected=False)
    print(f"\nMelted to {len(ratings)} (worker, item) ratings "
          f"from {len({r.worker for r in ratings})} workers.")
    if drops:
        print("Dropped:")
        for k, v in drops.most_common():
            print(f"  {v:5d}  {k}")
    else:
        print("Dropped: nothing.")

    # ---- 4. worker screening --------------------------------------------
    stats = screen(ratings, assignments, args.check_threshold, args.min_checks)
    bad = sorted((w for w in stats.values() if w.dropped),
                 key=lambda w: (w.check_acc or 0.0))
    no_checks = [w for w in stats.values() if w.n_checks == 0]
    under = [w for w in stats.values()
             if 0 < w.n_checks < args.min_checks]
    table(f"Worker screening (drop below {args.check_threshold:.0%} on _check "
          f"items, with >= {args.min_checks} checks seen)",
          ["worker", "checks", "check acc", "items", "action"],
          [[w.worker, f"{w.n_checks_ok}/{w.n_checks}",
            f"{(w.check_acc or 0.0):.0%}", str(w.n_items), "DROPPED"]
           for w in bad] or [])
    kept = {w for w in stats if not stats[w].dropped}
    lost = [r for r in ratings if r.worker not in kept]
    ratings = [r for r in ratings if r.worker in kept]
    print(f"\n  dropped {len(bad)} worker(s), removing {len(lost)} ratings "
          f"({len({r.item_id for r in lost})} items affected)")
    print(f"  kept {len(kept)} worker(s); {len(no_checks)} of them saw NO "
          f"_check items and are kept unscreened, "
          f"{len(under)} saw fewer than {args.min_checks} and are exempt")

    # ---- 5. majority vote ------------------------------------------------
    items = tally(ratings, manifest, args.min_consensus)
    order = list(manifest)
    empty = [i for i in order if items[i].n_raters == 0]
    ties = [i for i in order if items[i].n_raters and items[i].verdict is None]
    if empty:
        print(f"  ! {len(empty)} item(s) have no surviving ratings at all")
    # Worth seeing explicitly: the design pays for 3 raters an item, but padded
    # slots push some to 6 and screening pushes others down to 1, and a 1-rater
    # "majority" is one person's opinion carrying the same weight below.
    hist = Counter(items[i].n_raters for i in order)
    print("  raters per item: " + "  ".join(f"{n}x{hist[n]}" for n in sorted(hist)))
    print(f"  {len(ties)} tie(s) (even rater counts from padded HIT slots) "
          f"-> unresolved, excluded from every agreement numerator")

    # ---- 6-7. agreement + bootstrap --------------------------------------
    agree_m, resolved_m = bootstrap(ratings, items, order, args.bootstrap, args.min_consensus)
    pos = {i: k for k, i in enumerate(order)}

    def report(name: str, keys: list[str], key_of) -> list[list[str]]:
        rows = []
        for k in keys:
            sel = [i for i in order if key_of(items[i].row) == k]
            # An item nobody rated is not a tie, it is missing data; it is
            # reported once above and left out of both columns here.
            comparable = [i for i in sel
                          if items[i].vlm is not None and items[i].n_raters]
            res = [i for i in comparable if items[i].verdict is not None]
            hit = [i for i in res if items[i].verdict == items[i].vlm]
            mask = np.zeros(len(order), dtype=bool)
            mask[[pos[i] for i in comparable]] = True
            ci = ci_of(agree_m, resolved_m, mask) if comparable else None
            rows.append([k, str(len(sel)), str(len(comparable)),
                         str(len(comparable) - len(res)),
                         rate(len(hit), len(res)),
                         f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"])
        return rows

    hdr = ["", "items", "judged", "ties", "human-VLM agreement", "95% CI"]
    conds = sorted({m["condition"] for m in manifest.values()})
    table("AGREEMENT BY CONDITION  (denominator = judged items minus ties; "
          "_check arms carry no vlm_pass and are excluded by construction)",
          hdr, report("condition", conds, lambda m: m["condition"]))
    table("AGREEMENT BY FAMILY", hdr,
          report("family", sorted({m["family"] for m in manifest.values()}),
                 lambda m: m["family"]))
    table("AGREEMENT BY METHOD", hdr,
          report("method", sorted({m["method"] for m in manifest.values()}),
                 lambda m: m["method"]))
    table("AGREEMENT OVERALL", hdr, report("overall", ["all"], lambda m: "all"))
    print(f"\n  CIs are percentile intervals over {args.bootstrap} bootstrap "
          f"resamples of WORKERS (the unit of independence), majority vote "
          f"recomputed inside each resample.\n"
          f"  A resample omits ~37% of workers, so items lose raters and some "
          f"fall to a 1-1 tie; the interval can therefore sit below the point\n"
          f"  estimate. That is the cost of honouring the worker-level "
          f"correlation, not a miscomputation --- read it as a spread, not as a "
          f"recentred estimate.")

    # ---- 8. diagnostics --------------------------------------------------
    rows = []
    for c in conds:
        sel = [items[i] for i in order
               if items[i].row["condition"] == c and items[i].n_raters]
        rows.append([c, str(len(sel)),
                     rate(sum(it.unanimous_role for it in sel), len(sel)),
                     rate(sum(it.unanimous_verdict for it in sel), len(sel))])
    table("INTER-RATER AGREEMENT (all raters gave the same answer)",
          ["condition", "items", "same 4-way answer", "same verdict"], rows)

    by_group: dict[str, Counter] = defaultdict(Counter)
    for r in ratings:
        by_group[manifest[r.item_id]["hit_group"]][r.choice] += 1
        by_group["ALL"][r.choice] += 1
    rows = []
    for g in sorted(by_group, key=lambda g: (g == "ALL", g)):
        c = by_group[g]
        lr = c["left"] + c["right"]
        rows.append([g, str(sum(c.values())), str(c["left"]), str(c["right"]),
                     rate(c["left"], lr)])
    table("POSITION BIAS (L/R is randomised per item, so expect ~50%)",
          ["hit group", "ratings", "left", "right", "left share"], rows)

    rows = []
    for c in conds:
        sel = [r for r in ratings if r.condition == c]
        n = len(sel)
        rows.append([c, str(n),
                     rate(sum(r.role == "both" for r in sel), n),
                     rate(sum(r.role == "neither" for r in sel), n)])
    table("'BOTH' / 'NEITHER' USAGE  (ref_to_image predicts 'both', so a high "
          "rate there is the signal, not noise)",
          ["condition", "ratings", "both", "neither"], rows)

    for r in ratings:
        v = items[r.item_id].verdict
        if v is not None:
            stats[r.worker].maj_total += 1
            stats[r.worker].maj_agree += int(r.verdict == v)
    rows = []
    for w in sorted(stats.values(), key=lambda w: (-w.n_items, w.worker)):
        rows.append([w.worker + (" *" if w.dropped else ""), str(w.n_items),
                     f"{w.n_checks_ok}/{w.n_checks}" if w.n_checks else "-",
                     f"{w.check_acc:.0%}" if w.check_acc is not None else "-",
                     f"{w.work_time:.0f}s", str(w.n_assignments),
                     rate(w.maj_agree, w.maj_total)])
    table("PER-WORKER  (* = screened out; their rows show pre-drop counts and "
          "no majority agreement, since their ratings are excluded)",
          ["worker", "items", "checks", "acc", "mean time", "HITs",
           "agrees w/ majority"], rows)

    # ---- the useful bit: every human/VLM disagreement --------------------
    disagree = [items[i] for i in order
                if items[i].vlm is not None and items[i].verdict is not None
                and items[i].verdict != items[i].vlm]
    url = (lambda f: f"{args.base_url}/{f}") if args.base_url else (lambda f: f)
    # Printed as a block per item rather than a table: entity_ids run to 50
    # characters and there are three image names per row, so a tabular layout
    # either wraps or truncates exactly the fields you need in order to go and
    # look at the pictures.
    print(f"\nDISAGREEMENTS: human majority != vlm_pass  "
          f"({len(disagree)} of {sum(1 for i in order if items[i].vlm is not None and items[i].verdict is not None)} "
          f"resolved, judged items)")
    print("=" * 78)
    if not disagree:
        print("  (none)")
    for n, it in enumerate(disagree, 1):
        m = it.row
        print(f"\n[{n:2d}] {m['condition']}   {m['entity_id']}")
        print(f"     vlm_pass={it.vlm}  human={it.verdict}  "
              f"({split_str(it.roles)}; {it.n_raters} "
              f"rater{'s' if it.n_raters != 1 else ''}; "
              f"predicted answer = {target_role(m)})")
        for slot in ("top", "left", "right"):
            print(f"     {slot:<5s} {m['role_' + slot]:<12s} {url(m['img_' + slot])}")

    if args.no_write:
        return

    # ---- 9. write --------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vpath = args.out_dir / "human_verdicts.csv"
    with vpath.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "condition", "family", "method", "entity_id",
                    "vlm_pass", "human_verdict", "n_pass", "n_fail", "n_raters",
                    "n_baseline", "n_intervention", "n_both", "n_neither",
                    "unanimous", "agrees_with_vlm"])
        for i in order:
            it = items[i]
            m = it.row
            v = it.verdict
            w.writerow([
                i, m["condition"], m["family"], m["method"], m["entity_id"],
                m["vlm_pass"],
                # On a _check row the "verdict" is whether the majority passed
                # the attention check, not a judge comparison: vlm_pass is blank
                # there so it can never leak into an agreement rate.
                "" if v is None else v,
                it.n_pass, it.n_fail, it.n_raters,
                it.roles["baseline"], it.roles["intervention"],
                it.roles["both"], it.roles["neither"],
                int(it.unanimous_verdict),
                "" if (v is None or it.vlm is None) else int(v == it.vlm),
            ])

    dpath = args.out_dir / "disagreements.csv"
    with dpath.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "condition", "family", "method", "entity_id",
                    "ref_key", "sub_key", "vlm_pass", "human_verdict",
                    "n_pass", "n_fail", "n_raters", "vote_split",
                    "img_top", "img_left", "role_left", "img_right",
                    "role_right"])
        for it in disagree:
            m = it.row
            w.writerow([it.item_id, m["condition"], m["family"], m["method"],
                        m["entity_id"], m["ref_key"], m["sub_key"], it.vlm,
                        it.verdict, it.n_pass, it.n_fail, it.n_raters,
                        split_str(it.roles), url(m["img_top"]),
                        url(m["img_left"]), m["role_left"],
                        url(m["img_right"]), m["role_right"]])

    print(f"\nWrote {vpath}")
    print(f"Wrote {dpath}")


if __name__ == "__main__":
    main()
