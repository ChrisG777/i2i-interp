"""Select a reference-diverse subset of paper cells for human (MTurk) grading.

The VLM judges in ``scripts/judge/`` grade every cell. This script picks a
subset for a human validation study, chosen to maximise coverage of *distinct
reference images* (8 solid colors, 18 style refs, 10 DreamBench++ humans)
rather than to be a uniform random sample --- a random draw of 36 style cells
would leave several of the 18 style references unrepresented.

Three nested counts, which are easy to confuse:

* **task**  -- one paper cell: a ``task_id`` (knockout) or a ``<src>__<tgt>``
  pair id (patching). 120 of them.
* **item**  -- one question put to a worker. Each task yields one item per
  *arm*: the two knockout directions, or the patched output vs. an
  attention check. 240 of them. An item shows 3 images at once, the same
  way a judge bundle does.
* **image** -- one exported JPEG. 480 of them; a task's reference and baseline
  are shared by both of its arms rather than exported twice.

Every arm of a group grades the SAME task list, so the arms are a within-task
contrast. Diversity priority inside ``select()``: distinct reference images
first, then distinct prompts, and only then repeated seeds.

BOTH methods use one identical layout and one identical question per family,
mirroring the judges: each judge also asks a comparative question against a
clean baseline. A reference on top, two outputs below in randomised
left/right order, four options (Left / Right / Both / Neither).

``ko`` (Attention Knockout)
    Top: the task reference. Below: the clean i2i baseline and the knocked-out
    output. ``ref->text`` ``pass=1`` (property destroyed) predicts workers pick
    the baseline; ``ref->image`` ``pass=1`` (property survives) predicts "Both".

``i2i2i`` (I2I-to-I2I Patching)
    Top: the SOURCE reference. Below: the clean target baseline and the patched
    output. ``pass=1`` predicts workers pick the patched side. The judge's
    Image 2 (target reference) is dropped --- it exists only to say what the
    baseline should look like, and the worker sees the baseline itself.
    The ``_check`` arm swaps the patched output for ``source_i2i_4step.png``,
    the source pass's own generation, which always carries the source property:
    an attention check with a guaranteed-correct answer.

Selection is deterministic across processes: every shuffle seed is a
sha256-derived int, never a Python ``hash()`` of a string.

Outputs (under ``--out-dir``, default ``results_v4/human_eval/``):

* ``images/<sha1>.jpg``          -- flat, opaquely-named, web-sized exports, so
                                    a worker cannot infer the arm from a URL.
* ``manifest.csv``               -- one row per item: condition, entity_id,
                                    reference key, per-slot image + the true
                                    role behind that slot, and the VLM verdict
                                    for later human-vs-VLM agreement.
* ``mturk_input_<group>.csv``    -- items batched ``--per-hit`` to a row, one
                                    CSV per (method, family) group.
* ``hit_<group>.html``           -- the matching layout for that group.

Usage::

    uv run python -m scripts.human_eval.select_subset
    uv run python -m scripts.human_eval.select_subset --per-hit 20 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_V4 = REPO_ROOT / "results_v4"
JUDGE_DIR = RESULTS_V4 / "vlm_judge"
PAIRS_DIR = REPO_ROOT / "experiments" / "i2i_to_i2i_patching" / "pairs"

KO_BASE = RESULTS_V4 / "attention_knockout" / "full_ko_4step"
I2I2I_COLOR = RESULTS_V4 / "i2i_to_i2i_patching" / "single9_4step_color"
I2I2I_STYLE = RESULTS_V4 / "i2i_to_i2i_patching" / "mm7_4step_style_to_real"
I2I2I_HUMANS = RESULTS_V4 / "i2i_to_i2i_patching" / "mm7_4step_dreambench_humans"
T2I_LENS_MM7 = RESULTS_V4 / "i2i_to_unconditional" / "mm7_4step"


def seed_of(*parts: str) -> int:
    """Deterministic cross-process seed. Python's ``hash()`` of a str is
    salted per interpreter, so it cannot be used to make a selection that
    reproduces on another machine or in another run."""
    h = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


# ---------------------------------------------------------------------------
# Task universes
# ---------------------------------------------------------------------------


def load_tasks(bucket: str) -> list[dict]:
    p = REPO_ROOT / "data" / "tasks" / bucket / "tasks.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def read_pairs(name: str) -> list[tuple[str, str]]:
    lines = (PAIRS_DIR / f"{name}.txt").read_text().splitlines()
    out = []
    for line in lines:
        if not line.strip():
            continue
        src, tgt = line.split("\t")
        out.append((src, tgt))
    return out


def human_idx(task_id: str) -> str:
    """``customize_dreambench_plus_human_shared_04_walking_city_street`` -> ``04``."""
    return task_id.split("_shared_")[1].split("_")[0]


def color_of(task_id: str) -> str:
    """``solid_green_mug_s0`` -> ``green``."""
    return task_id.split("_")[1]


def object_of(task_id: str) -> str:
    """``solid_green_mug_s0`` -> ``mug``."""
    return task_id.split("_")[2]


def style_ref_of(task_id: str) -> str:
    """``customize_property_style_free_sparrow_snow_fence_s3`` -> ``sparrow``.

    Recovered from the task table rather than parsed, because subject slugs
    contain underscores (``alice_in_wonderland``)."""
    return _STYLE_REF_BY_TASK[task_id]


_STYLE_REF_BY_TASK: dict[str, str] = {}
_PROMPT_SLUG_BY_TASK: dict[str, str] = {}
for _t in load_tasks("style"):
    _STYLE_REF_BY_TASK[_t["task_id"]] = _t["real_ref_name"]
    _PROMPT_SLUG_BY_TASK[_t["task_id"]] = _t["metadata"]["prompt_slug"]


# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    name: str
    family: str                       # color | style | human
    method: str                       # ko | i2i2i
    judge_csv: str                    # under results_v4/vlm_judge/
    base_dir: Path
    # Tasks (or pairs) selected for the whole group. Every arm in a group grades
    # the SAME tasks, so the arms are a within-task contrast and the reference +
    # baseline images are shared rather than exported twice.
    n_tasks: int
    universe: Callable[[], list[str]]           # entity ids
    ref_key: Callable[[str], str]               # primary diversity axis
    sub_key: Callable[[str], str]               # secondary diversity axis
    slots: dict[str, str]                       # slot name -> filename in entity dir
    # Which slot pair is randomised into the left / right positions
    randomise: tuple[str, str]
    # Per-slot override of base_dir. The knockout check arm needs the clean
    # no-reference generation, which the knockout sweep wrote for color and
    # style but not for the DreamBench humans; the T2I-lens sweep has it for all
    # of them. Same task_id, same prompt, same 4-step schedule -- only the
    # results root differs.
    slot_base: dict[str, Path] = field(default_factory=dict)
    # Per-slot full-path resolver, for slots whose image does not live in the
    # entity's own directory at all. The paired ref->image arms show a foil
    # drawn from a *different* cell -- another reference colour, another
    # subject, a generation with no reference -- so its path is a function of
    # the hand-labelled pairing, not of the entity id.
    slot_resolver: dict[str, Callable[[str], Path]] = field(default_factory=dict)
    # The role whose selection counts as a human "pass". Written per item into
    # the manifest so the analysis reads it instead of re-deriving it from the
    # condition name -- every sign error in this pipeline has come from a name
    # whose meaning changed without the name changing.
    target_role: str = ""
    # This arm's slice of the group's chosen task list. ``None`` means all of
    # it, which is the norm -- arms are a within-task contrast. Only the paired
    # screening arm narrows it: a check costs a worker's time but yields no
    # data, so it rides along on a fraction of the sample, and it can only use
    # cells whose foil is a non-carrier.
    arm_pool: Callable[[list[str]], list[str]] | None = None
    # HIT batching key. Defaults to (method, family); the paired arms override
    # it because their layout is a two-option forced choice and cannot share a
    # HIT with the four-option one.
    group: str = ""
    # Force an exact left/right split instead of an independent coin per item.
    # An independent coin on 53 items lands 35/18 often enough to matter, and a
    # side imbalance multiplies straight into any worker-side position bias.
    # Only the new arms set it; the published ones keep the exact randomisation
    # their HITs already went out with.
    balance_positions: bool = False
    # Whether to show the reference at the top. Color and identity need it --
    # "that color", "that person" are meaningless without something to point
    # at. Style does not: the question is whether an image looks drawn or
    # photographed, which is a property of the image alone. Showing the source
    # drawing invites workers to hunt for resemblance to *that* drawing, a
    # similarity judgement the style judge never asks for either.
    show_ref: bool = True

    def dir_for(self, slot: str) -> Path:
        return self.slot_base.get(slot, self.base_dir)

    def path_for(self, slot: str, eid: str) -> Path:
        r = self.slot_resolver.get(slot)
        return r(eid) if r else self.dir_for(slot) / eid / self.slots[slot]

    @property
    def hit_group(self) -> str:
        """HITs are batched per (method, family). Every question in one HIT is
        then the same question, so the wording can be concrete about the one
        property being judged instead of hedging across all three."""
        return self.group or f"{self.method}_{self.family}"


# ---------------------------------------------------------------------------
# Worker-facing wording. Deliberately avoids "reference image", "baseline",
# "model", and any other framing that only makes sense if you know what the
# intervention was: each family names the one concrete property being judged
# (a color, a drawing, a person) and points at a position on the page.
# ---------------------------------------------------------------------------


# "Both" and "Neither" are not padding. The ref->image arm's whole prediction is
# that the property SURVIVES the knockout, i.e. both images below carry it --- a
# plain Left/Right forced choice can only express that as a chance-level split,
# which is indistinguishable from "both LOST it". "Neither" is the other half of
# that: without it, two equally-wrong images also produce a chance split. With
# all four options each item yields a complete absolute judgement in one click.
KO_TEXT = {
    "color": {
        "title": "Match the color",
        "top_tag": "THIS COLOR",
        "intro": "At the top is a colored square. Below it are two photos of an "
                 "object.",
        # Scoped to the OBJECT. The reference-conditioned baseline floods the
        # whole canvas with the reference colour while the knocked-out output
        # keeps it only on the object, so "which photo matches the color"
        # rewards the image with more coloured pixels rather than the one whose
        # object is that colour --- workers answered the question as written and
        # it was the wrong question.
        "q": "Which of the objects below are the color at the top? Ignore the background.",
        "opt_both": "Both are that color",
        "opt_neither": "Neither is that color",
    },
    "style": {
        "title": "Drawing or photograph?",
        "top_tag": "",           # style shows no reference; see Condition.show_ref
        "intro": "Each question shows two pictures.",
        "q": "Which of these pictures look like drawings rather than real "
             "photographs?",
        "opt_both": "Both look like drawings",
        "opt_neither": "Neither does",
    },
    "human": {
        "title": "Same person, or not?",
        "top_tag": "THIS PERSON",
        "intro": "At the top is a photo of a person. Below it are two more "
                 "photos.",
        "q": "Which of the photos below show the same person as the one at the "
             "top?",
        "opt_both": "Both are that person",
        "opt_neither": "Neither is that person",
    },
}


# Wording for the paired arms. Every item now has exactly one correct answer by
# construction, so the four-option form is gone: no "Both", no "Neither", and
# the question is singular. Saying so up front is not a hint --- it is true of
# every item, and a worker who does not know it wastes time looking for a
# second right answer that is not there.
#
# Each family follows one shape: a sentence naming what the top image presents
# as the TARGET, then a question scoped to the two images below. The tolerance
# word ("roughly") is load-bearing --- these are generated images, none of them
# match a target exactly, and without it a worker reasonably reads the question
# as demanding an exact match and answers "neither" by picking arbitrarily.
KO_PAIR_TEXT = {
    "color": {
        "title": "Which image matches the target color?",
        "top_tag": "TARGET COLOR",
        # The intro is printed immediately before the question, so it sets the
        # scene and states that exactly one answer is right, and leaves the
        # property itself to the question rather than saying it twice.
        "intro": "The image at the top presents a uniform target color. Below "
                 "it are two images, each containing one object. Exactly one of "
                 "the two objects matches.",
        "q": "In the images on the bottom, which image contains an object that "
             "roughly follows the target color? Ignore the background.",
        "opt_left": "The left image",
        "opt_right": "The right image",
    },
    "style": {
        "title": "Which image is the drawing?",
        "top_tag": "",           # style shows no target; see Condition.show_ref
        "intro": "Each question below shows two images, one of them drawn and "
                 "the other photographed.",
        "q": "Which of the two images is the drawing or illustration, and NOT "
             "a real photograph?",
        "opt_left": "The left image",
        "opt_right": "The right image",
    },
    "human": {
        "title": "Which image shows the target person?",
        "top_tag": "TARGET PERSON",
        "intro": "The image at the top presents a target person. Below it are "
                 "two images, each showing a person. Exactly one of them is the "
                 "target person.",
        "q": "In the images on the bottom, which image shows the same person as "
             "the target person?",
        "opt_left": "The left image",
        "opt_right": "The right image",
    },
}


def _ko_conditions() -> list[Condition]:
    color_tasks = [t["task_id"] for t in load_tasks("solid_color")]
    style_tasks = [t["task_id"] for t in load_tasks("style")]
    # 3 of the 9 dreambench_humans prompts per subject ask for a watercolour, an
    # anime illustration or a retro drawing (p4/p5/p6), and two more name no
    # medium at all (p7/p8). Identity here is a photo-to-photo judgement, so the
    # human study samples only the prompts that explicitly ask for a photograph.
    human_tasks = [t["task_id"] for t in load_tasks("dreambench_humans")
                   if t["instruction"].lower().startswith(("a photo ", "a photograph "))]

    # n_tasks is the count of DISTINCT tasks graded per group; each is graded
    # twice (once per knockout direction), sharing its reference + baseline
    # images. Diversity priority is baked into select(): distinct reference
    # images first, then distinct prompts, and only then repeated seeds.
    specs = [
        ("color", color_tasks, color_of, object_of, 16),
        ("style", style_tasks, style_ref_of, lambda i: _PROMPT_SLUG_BY_TASK[i], 18),
        ("human", human_tasks, lambda i: i.split("_human_")[1].split("_p")[0],
         lambda i: i.rsplit("_", 1)[1], 20),
    ]
    judge_name = {
        ("color", "text"): "ko_color_ref_to_text",
        ("color", "image"): "ko_color_ref_to_image",
        ("style", "text"): "ko_style_ref_to_text",
        ("style", "image"): "ko_style_ref_to_image",
        ("human", "text"): "ko_dreambench_human_ref_to_text",
        ("human", "image"): "ko_dreambench_human_ref_to_image",
    }
    out = []
    for family, ids, rk, sk, n in specs:
        # Both knockout directions are genuine experimental questions -- their
        # answers are the result, so neither can score a worker. Without a third
        # arm the whole knockout half is unscreenable: the six projects are
        # separate, so a worker who only takes ko_* HITs never meets a check
        # item. ``t2i_clean_4step.png`` is the same prompt generated with NO
        # reference at all, so it cannot carry the reference property and the
        # baseline is the known-correct answer. Verified for color: across the
        # selected tasks the clean generation's dominant colour is 97-199 RGB
        # away from its reference swatch, never close enough to be ambiguous.
        # Style and identity are safer still -- the prompt asks for "A
        # photograph of ...", so the clean pass is photographic and shows a
        # stranger, while the baseline is stylised and shows the reference
        # person.
        # The knockout sweep wrote t2i_clean_4step.png for color and style but
        # not for the DreamBench humans; the T2I-lens sweep has it for all 40
        # photo-prompt human tasks. Same task_id and prompt, different root.
        clean_base = {} if family != "human" else {"intervention": T2I_LENS_MM7}
        arms = [("ref_to_text", "ref_to_text_full_ko.png",
                 judge_name[(family, "text")], {}),
                ("ref_to_image", "ref_to_image_full_ko.png",
                 judge_name[(family, "image")], {}),
                ("check", "t2i_clean_4step.png",
                 judge_name[(family, "text")], clean_base)]
        for arm, fname, judge, sbase in arms:
            out.append(Condition(
                name=f"ko_{family}_{arm}",
                family=family, method="ko",
                judge_csv=judge,
                base_dir=KO_BASE, n_tasks=n,
                universe=(lambda ids=ids: list(ids)),
                ref_key=rk, sub_key=sk,
                slots={"ref": "reference.png",
                       "baseline": "i2i_baseline_4step.png",
                       "intervention": fname},
                randomise=("baseline", "intervention"),
                slot_base=sbase,
                show_ref=(family != "style"),
            ))
    return out


REF_TO_IMAGE_LABELS = RESULTS_V4 / "human_eval" / "ref_to_image_labels.csv"


def load_ref_to_image_labels() -> dict[str, dict]:
    """The hand labels that decide each ref->image pairing, keyed by entity id.

    Rows marked unusable are absent: an item nobody could call is a hole, not a
    label, and pairing it would put two images of unknown status side by side
    and reduce the question to a coin flip. ``label_ui.py`` draws a replacement
    cell for each one, so the arm keeps its size."""
    if not REF_TO_IMAGE_LABELS.exists():
        return {}
    with REF_TO_IMAGE_LABELS.open() as f:
        return {r["entity_id"]: r for r in csv.DictReader(f)
                if r["answer"] and r["answer"] != "unusable"}


def _ko_pair_conditions() -> list[Condition]:
    """The redesigned ref->image arms: one knockout output against one foil.

    The old arm showed the knockout beside its own baseline, so when the
    property survived *both* images carried it and the worker was grading a
    matter of degree -- which is why round 2 landed at 56-59% with humans and
    the judge agreeing wherever an item was determinate and splitting only on
    where to put a threshold. Here the foil is chosen against the hand label,
    so exactly one of the two images ever carries the property and the question
    has a determinate answer.

    Note that the label does not enter the scoring. A pass is "the worker
    picked the knockout output", which is their verdict on whether the property
    survived, compared straight to the judge's. The label only decides which
    foil is shown.
    """
    labels = load_ref_to_image_labels()
    if not labels:
        print("  ! no ref_to_image_labels.csv -- skipping the paired arms; "
              "run scripts.human_eval.label_ui first")
        return []

    judge = {"color": "ko_color_ref_to_image", "style": "ko_style_ref_to_image",
             "human": "ko_dreambench_human_ref_to_image"}
    keys = {
        "color": (color_of, object_of),
        "style": (style_ref_of, lambda i: _PROMPT_SLUG_BY_TASK[i]),
        "human": (lambda i: i.split("_human_")[1].split("_p")[0],
                  lambda i: i.rsplit("_", 1)[1]),
    }
    out = []
    for family in ("color", "style", "human"):
        fam_labels = {e: r for e, r in labels.items() if r["family"] == family}
        rk, sk = keys[family]

        def foil(eid: str, fl=fam_labels) -> Path:
            r = fl[eid]
            return KO_BASE / r["foil_dir"] / r["foil_file"]

        # The screening arm reuses the pairing machinery with a known answer:
        # the task's own baseline against a foil that cannot carry the property.
        # It is restricted to the cells whose labelled foil is a non-carrier --
        # for a cell labelled "lost" the foil IS the baseline, which would pit
        # an image against itself.
        checkable = {e for e, r in fam_labels.items()
                     if r["foil_role"] == "noncarrier"}

        n_check = max(6, round(len(checkable) / 3))

        def check_pool(chosen: list[str], ok=checkable, n=n_check) -> list[str]:
            return [e for e in chosen if e in ok][:n]

        common = dict(family=family, method="ko", base_dir=KO_BASE,
                      ref_key=rk, sub_key=sk, show_ref=(family != "style"),
                      judge_csv=judge[family], n_tasks=len(fam_labels),
                      universe=(lambda fl=fam_labels: sorted(fl)),
                      slot_resolver={"foil": foil},
                      group=f"ko_{family}_pair", balance_positions=True)
        out.append(Condition(
            name=f"ko_{family}_pair",
            slots={"ref": "reference.png",
                   "intervention": "ref_to_image_full_ko.png",
                   "foil": ""},                      # resolved per entity
            randomise=("intervention", "foil"),
            target_role="intervention", **common))
        out.append(Condition(
            name=f"ko_{family}_paircheck",
            slots={"ref": "reference.png",
                   "baseline": "i2i_baseline_4step.png",
                   "foil": ""},
            randomise=("baseline", "foil"),
            target_role="baseline",
            arm_pool=check_pool, **common))
    return out


def _i2i2i_conditions() -> list[Condition]:
    color_pairs = read_pairs("single9_4step_color")
    style_pairs = read_pairs("mm7_4step_style_to_real")
    human_pairs = read_pairs("mm7_4step_dreambench_humans")

    def pid(p: tuple[str, str]) -> str:
        return f"{p[0]}__{p[1]}"

    # ref_key is the SOURCE reference (never the ordered pair): round-robin over
    # ordered pairs balances pairs but can leave an individual reference image
    # unused, and the whole point of the subset is per-reference coverage. The
    # target reference rides in sub_key so the interleave diversifies it too.
    specs = [
        ("color", I2I2I_COLOR, "i2i2i_color", color_pairs, 24,
         lambda i: color_of(i.split("__")[0]),
         lambda i: f"{color_of(i.split('__')[1])}|{object_of(i.split('__')[0])}"),
        ("style", I2I2I_STYLE, "i2i2i_style", style_pairs, 18,
         lambda i: style_ref_of(i.split("__")[0]),
         lambda i: _PROMPT_SLUG_BY_TASK[i.split("__")[0]]),
        ("human", I2I2I_HUMANS, "i2i2i_dreambench_humans", human_pairs, 24,
         lambda i: human_idx(i.split("__")[0]),
         lambda i: f"{human_idx(i.split('__')[1])}|"
                   f"{i.split('__')[0].split('_shared_')[1].split('_', 1)[1]}"),
    ]
    out = []
    for family, base, judge, pairs, n, rk, sk in specs:
        ids = [pid(p) for p in pairs]
        # Same three-image form as the knockout layout, against the SOURCE
        # reference: does the lower-right image carry the source property that
        # the clean target baseline does not?
        #
        #   patched -- the real question. ``patched.png`` vs the clean baseline.
        #   check   -- an attention check with a known answer. ``source_i2i``
        #              is the SOURCE pass's own output, so it always carries the
        #              source property; a worker who picks the target baseline
        #              over it is not looking.
        for arm, fname in [("patched", "patched.png"),
                           ("check", "source_i2i_4step.png")]:
            out.append(Condition(
                name=f"i2i2i_{family}_{arm}",
                family=family, method="i2i2i",
                judge_csv=judge, base_dir=base, n_tasks=n,
                universe=(lambda ids=ids: list(ids)),
                ref_key=rk, sub_key=sk,
                slots={"ref": "ref_source.png",
                       "baseline": "target_baseline_4step.png",
                       "intervention": fname},
                randomise=("baseline", "intervention"),
                show_ref=(family != "style"),
            ))
    return out


def all_conditions() -> list[Condition]:
    return _ko_conditions() + _ko_pair_conditions() + _i2i2i_conditions()


# ---------------------------------------------------------------------------
# Reference-diverse selection
# ---------------------------------------------------------------------------


def select(cond: Condition, on_disk: Callable[[str], bool]) -> list[str]:
    """Round-robin over distinct reference keys, then over the secondary key
    within each reference, so that every reference image is represented before
    any reference is used twice."""
    by_ref: dict[str, list[str]] = defaultdict(list)
    for eid in cond.universe():
        if on_disk(eid):
            by_ref[cond.ref_key(eid)].append(eid)

    if not by_ref:
        return []

    # Deterministic per-reference ordering, diversified on the secondary axis:
    # shuffle within each (ref, sub) group, then interleave the groups.
    ordered: dict[str, list[str]] = {}
    for ref, eids in by_ref.items():
        by_sub: dict[str, list[str]] = defaultdict(list)
        for eid in eids:
            by_sub[cond.sub_key(eid)].append(eid)
        for sub, group in by_sub.items():
            random.Random(seed_of(cond.hit_group, ref, sub)).shuffle(group)
        subs = sorted(by_sub)
        random.Random(seed_of(cond.hit_group, ref, "#subs")).shuffle(subs)
        flat: list[str] = []
        i = 0
        while len(flat) < len(eids):
            for sub in subs:
                if i < len(by_sub[sub]):
                    flat.append(by_sub[sub][i])
            i += 1
        ordered[ref] = flat

    refs = sorted(ordered)
    random.Random(seed_of(cond.hit_group, "#refs")).shuffle(refs)

    picked: list[str] = []
    depth = 0
    while len(picked) < cond.n_tasks:
        added = False
        for ref in refs:
            if depth < len(ordered[ref]):
                picked.append(ordered[ref][depth])
                added = True
                if len(picked) == cond.n_tasks:
                    break
        if not added:
            break            # universe exhausted
        depth += 1
    return picked


# ---------------------------------------------------------------------------
# Image export
# ---------------------------------------------------------------------------


def export(src: Path, out_dir: Path, size: int, quality: int, dry: bool) -> str:
    """Copy ``src`` to ``out_dir`` under a content-and-path-derived opaque name.
    Returns the exported filename."""
    key = str(src.relative_to(RESULTS_V4))
    name = hashlib.sha1(key.encode()).hexdigest()[:16] + ".jpg"
    dst = out_dir / name
    if not dry and not dst.exists():
        im = Image.open(src).convert("RGB")
        im.thumbnail((size, size), Image.LANCZOS)
        im.save(dst, "JPEG", quality=quality)
    return name


def load_verdicts(judge_csv: str) -> dict[str, str]:
    p = JUDGE_DIR / f"{judge_csv}.csv"
    if not p.exists():
        return {}
    with p.open() as f:
        return {r["entity_id"]: r["pass"] for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------


@dataclass
class Item:
    item_id: str
    condition: str
    family: str
    method: str
    hit_group: str
    entity_id: str
    ref_key: str
    sub_key: str
    images: dict[str, str] = field(default_factory=dict)   # display slot -> filename
    roles: dict[str, str] = field(default_factory=dict)    # display slot -> true role
    vlm_pass: str = ""
    target_role: str = ""


def build_items(conds: Iterable[Condition], out_images: Path,
                size: int, quality: int, dry: bool) -> list[Item]:
    by_group: dict[str, list[Condition]] = defaultdict(list)
    for c in conds:
        by_group[c.hit_group].append(c)

    items: list[Item] = []
    for group, arms in by_group.items():
        head = arms[0]
        # A task qualifies only if EVERY arm's image exists, so no arm silently
        # ends up grading a different task list than its sibling. Resolved per
        # (arm, slot): an arm may source a slot from another results root, or
        # from another cell entirely.
        wanted = [(a, slot) for a in arms for slot in a.slots
                  if slot != "ref" or a.show_ref]

        def on_disk(eid: str, wanted=wanted) -> bool:
            return all(a.path_for(slot, eid).exists() for a, slot in wanted)

        chosen = select(head, on_disk)
        if len(chosen) < head.n_tasks:
            print(f"  ! {group}: only {len(chosen)}/{head.n_tasks} tasks on disk")
        items.extend(_items_for(arms, chosen, out_images, size, quality, dry))
    return items


def _items_for(arms: list[Condition], chosen: list[str], out_images: Path,
               size: int, quality: int, dry: bool) -> list[Item]:
    """One item per (arm, task). Arms share the task list, so the reference and
    baseline exports are deduplicated by ``export``'s content-derived name."""
    items: list[Item] = []
    for cond in arms:
        verdicts = load_verdicts(cond.judge_csv)
        lo, hi = cond.randomise
        pool = cond.arm_pool(chosen) if cond.arm_pool else chosen
        # Deterministic either way: an independent per-item coin, or a shuffle
        # whose first half is flipped so the split is exactly even.
        flipped = set()
        if cond.balance_positions:
            order = sorted(pool, key=lambda e: seed_of(cond.name, e, "#flip"))
            flipped = set(order[: len(order) // 2])
        for eid in pool:
            flip = (eid in flipped if cond.balance_positions
                    else random.Random(seed_of(cond.name, eid, "#flip")).random() < 0.5)
            a, b = (hi, lo) if flip else (lo, hi)

            slots = ({"top": "ref", "left": a, "right": b} if cond.show_ref
                     else {"left": a, "right": b})

            it = Item(
                item_id=hashlib.sha1(f"{cond.name}|{eid}".encode()).hexdigest()[:12],
                condition=cond.name, family=cond.family, method=cond.method,
                hit_group=cond.hit_group,
                entity_id=eid, ref_key=cond.ref_key(eid), sub_key=cond.sub_key(eid),
                # The check arm swaps in the source pass's own output, which no
                # judge was ever asked about --- the judge CSV row for this pair
                # holds the verdict on the *patched* image. Leave it blank rather
                # than let a mislabelled verdict reach the agreement analysis.
                vlm_pass=("" if cond.name.endswith(("_check", "_paircheck"))
                          else verdicts.get(eid, "")),
                target_role=cond.target_role,
            )
            for display, role in slots.items():
                src = cond.path_for(role, eid)
                it.images[display] = export(src, out_images, size, quality, dry)
                it.roles[display] = role
            items.append(it)
    return items


def write_manifest(items: list[Item], path: Path) -> None:
    displays = sorted({d for it in items for d in it.images})
    cols = (["item_id", "condition", "family", "method", "hit_group",
             "entity_id", "ref_key", "sub_key", "vlm_pass", "target_role"]
            + [f"img_{d}" for d in displays] + [f"role_{d}" for d in displays])
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for it in items:
            w.writerow(
                [it.item_id, it.condition, it.family, it.method, it.hit_group,
                 it.entity_id, it.ref_key, it.sub_key, it.vlm_pass, it.target_role]
                + [it.images.get(d, "") for d in displays]
                + [it.roles.get(d, "") for d in displays])


def write_mturk_input(items: list[Item], path: Path, per_hit: int,
                      base_url: str, group: str) -> int:
    """One CSV row per HIT, ``per_hit`` items per row, all from one
    (method, family) group so every question in the HIT is the same question.
    The group's two arms are interleaved so a worker's ratings are never all
    from one arm."""
    sel = [it for it in items if it.hit_group == group]
    by_cond: dict[str, list[Item]] = defaultdict(list)
    for it in sel:
        by_cond[it.condition].append(it)
    conds = sorted(by_cond)
    interleaved: list[Item] = []
    i = 0
    while len(interleaved) < len(sel):
        for c in conds:
            if i < len(by_cond[c]):
                interleaved.append(by_cond[c][i])
        i += 1

    # No HIT may show a worker the same underlying cell twice. Two conditions
    # routinely share a cell: KO ref->text / ref->image are the same task, and
    # the patched / check arms of a pair share reference and baseline. Seeing
    # either back to back would let a worker anchor on their first answer, so
    # pack greedily under a no-repeated-entity constraint rather than slicing
    # the interleaved list into contiguous chunks.
    # A HIT can hold at most one item per distinct cell, so a group whose arms
    # share a small task pool caps out below --per-hit (18 style pairs cannot
    # fill a 20-slot HIT without showing one twice). Shrink this group's HIT
    # size to that ceiling; write_hit_html is handed the same number so the
    # layout always matches the CSV.
    per_hit = min(per_hit, len({it.entity_id for it in sel}))

    # Place into the emptiest non-colliding HIT, not the first one that fits.
    # First-fit packs HIT 0 solid, then HIT 1, and the second copy of a cell
    # that appears in both arms can arrive to find every non-colliding HIT
    # already full even though total capacity is fine.
    n_hits = -(-len(interleaved) // per_hit)
    hits: list[list[Item]] = [[] for _ in range(n_hits)]
    ents: list[set[str]] = [set() for _ in range(n_hits)]
    # Balance ARMS within each HIT, not just overall fullness. Entity-uniqueness
    # alone is not enough: when n_tasks == per_hit every HIT must hold each cell
    # exactly once, and the only packing a fullness-only rule finds is one arm
    # per HIT. That confounds condition with worker --- every ref->text rating
    # would come from one worker triple and every ref->image rating from
    # another, so an arm difference could just be a worker difference, which is
    # precisely the comparison this study exists to make. Prefer the HIT holding
    # fewest items of THIS item's condition, then the emptiest.
    arm_counts: list[Counter] = [Counter() for _ in range(n_hits)]
    for it in interleaved:
        order = sorted(range(n_hits),
                       key=lambda h: (arm_counts[h][it.condition], len(hits[h])))
        for h in order:
            if len(hits[h]) < per_hit and it.entity_id not in ents[h]:
                hits[h].append(it)
                ents[h].add(it.entity_id)
                arm_counts[h][it.condition] += 1
                break
        else:
            raise RuntimeError(f"{path.name}: cannot place {it.item_id} "
                               f"without repeating a cell; lower --per-hit")

    # A short final HIT would render empty <img src=""> slots, so pad it by
    # recycling items that don't collide. Recycled items just collect extra
    # ratings (and give an intra-worker consistency check); the analysis keys
    # on item_id, so a duplicate is another rating of the same item.
    for h in range(n_hits):
        if len(hits[h]) == per_hit:
            continue
        # Re-pick the least-represented arm for EVERY spare slot, not once for
        # the whole run: a single sort leaves the counts stale after the first
        # placement, so all the padding still piles onto whichever arm sorted
        # first, giving the screening arm more ratings than the arm that
        # actually feeds the agreement rate.
        while len(hits[h]) < per_hit:
            cand = min(
                (it for it in interleaved if it.entity_id not in ents[h]),
                key=lambda it: (arm_counts[h][it.condition], it.item_id),
                default=None,
            )
            if cand is None:
                break                       # no non-colliding donor left
            hits[h].append(cand)
            ents[h].add(cand.entity_id)
            arm_counts[h][cand.condition] += 1
        assert len(hits[h]) == per_hit, f"{path.name} HIT {h} is short"

    # Shuffle within each HIT. The packer above interleaves arms to balance
    # them, which leaves a strict A-B-A-B run down the page. Alternation does
    # balance order and fatigue effects, but it is also learnable: a worker who
    # picks up the rhythm can settle into an alternating response habit, and
    # that habit would correlate perfectly with condition -- manufacturing the
    # very arm difference the study exists to measure. Balance the arms, then
    # hide the pattern. Seeded per (group, HIT) so the order reproduces.
    for h, hit in enumerate(hits):
        random.Random(seed_of(group, "#order", str(h))).shuffle(hit)

    displays = sorted({d for it in sel for d in it.images})
    cols = ["item_id_%d" % k for k in range(per_hit)]
    for d in displays:
        cols += [f"img_{d}_{k}" for k in range(per_hit)]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for hit in hits:
            row = [it.item_id for it in hit]
            for d in displays:
                row += [f"{base_url}/{it.images[d]}" for it in hit]
            w.writerow(row)
    return len(hits), per_hit


# ---------------------------------------------------------------------------
# HIT layout. Emitted rather than committed as a static file so the number of
# trial blocks always matches --per-hit and the ${...} placeholders always
# match the column names in mturk_input_*.csv.
# ---------------------------------------------------------------------------


_HEAD = """<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>
<crowd-form>
<style>
  .wrap {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 900px;
           margin: 0 auto; line-height: 1.5; }}
  .trial {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 18px 0; }}
  .trial h3 {{ margin: 0 0 10px; font-size: 14px; color: #777; font-weight: 500; }}
  .row {{ display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;
          align-items: flex-end; }}
  .cell {{ text-align: center; }}
  .cell img {{ width: 260px; max-width: 40vw; border: 1px solid #ccc; border-radius: 4px;
               display: block; }}
  .cell .tag {{ font-weight: 700; margin-top: 6px; font-size: 13px; letter-spacing: .04em; }}
  .top img {{ width: 190px; }}
  .q {{ margin: 14px 0 4px; font-weight: 600; }}
  .opts {{ margin: 8px 0 0; }}
  .opt {{ display: block; padding: 5px 0; cursor: pointer; }}
  .opt input {{ margin-right: 8px; }}
  hr {{ border: 0; border-top: 1px dashed #ccc; margin: 14px 0; }}
  .intro {{ background: #f7f7f7; padding: 16px; border-radius: 8px; }}
</style>
<div class="wrap">
<div class="intro">
<h2>{title}</h2>
<p>{intro} {q}</p>
<p>These pictures were made by a computer, so some of them look odd. That is
expected. Please do not use any AI tool to answer.</p>
<p><b>Consent.</b> This is part of an academic research study. Your responses
are anonymous; we record only your worker ID and your answers. Participation is
voluntary and you may return the HIT at any time.</p>
<p><b>{n} questions, about {minutes} minutes.</b></p>
</div>
"""

_TAIL = """</div>
</crowd-form>
"""


# Native <input type="radio">, NOT crowd-radio-button. Two reasons, both from
# the AWS docs:
#   1. Mutual exclusion. crowd-radio-group only guarantees single-selection when
#      its children have DIFFERENT name values ("to ensure that only one button
#      in a group is selected ... use different name values"). Sibling buttons
#      sharing one name -- the natural HTML idiom -- can end up co-selected.
#   2. Output shape. crowd-radio-button emits {"<name>": {"<value>": true}},
#      i.e. a boolean per option, not the single value the analysis joins on.
# A native radio group keyed on `name` is browser-enforced, and crowd-form
# serialises it to exactly Answer.q<k>_choice = "left" | "right" | ... .
def _option(k: int, value: str, label: str, first: bool) -> str:
    req = " required" if first else ""
    return (f'    <label class="opt"><input type="radio" name="q{k}_choice" '
            f'value="{value}"{req}> {label}</label>')


def _ko_trial(k: int, t: dict, n: int, show_ref: bool = True,
              paired: bool = False) -> str:
    choices = ([("left", t["opt_left"]), ("right", t["opt_right"])] if paired else
               [("left", "Only the left"), ("right", "Only the right"),
                ("both", t["opt_both"]), ("neither", t["opt_neither"])])
    opts = "\n".join(_option(k, v, label, i == 0)
                     for i, (v, label) in enumerate(choices))
    # Style items show no reference at all, so the top block and its separator
    # are omitted rather than left empty -- an empty <img> renders as a broken
    # image icon, which reads to a worker as a loading failure.
    top = f"""  <div class="row top"><div class="cell">
    <img src="${{img_top_{k}}}"><div class="tag">{t['top_tag']}</div>
  </div></div>
  <hr>
""" if show_ref else ""
    return f"""
<div class="trial">
  <h3>Question {k + 1} of {n}</h3>
{top}  <div class="row">
    <div class="cell"><img src="${{img_left_{k}}}"><div class="tag">LEFT</div></div>
    <div class="cell"><img src="${{img_right_{k}}}"><div class="tag">RIGHT</div></div>
  </div>
  <p class="q">{t['q']}</p>
  <div class="opts">
{opts}
  </div>
  <input type="hidden" name="q{k}_item_id" value="${{item_id_{k}}}">
</div>"""


def write_hit_html(path: Path, group: str, per_hit: int) -> None:
    """One layout per (method, family) group, so the wording names the single
    property being judged instead of enumerating all three. Every item is a
    two-alternative forced choice: no "about the same" escape hatch, so chance
    is exactly 50% and the condition-level read is unambiguous."""
    paired = group.endswith("_pair")
    family = group.split("_")[1] if paired else group.split("_", 1)[1]
    t = (KO_PAIR_TEXT if paired else KO_TEXT)[family]
    show_ref = family != "style"
    body = "".join(_ko_trial(k, t, per_hit, show_ref, paired)
                   for k in range(per_hit))
    html = _HEAD.format(title=t["title"], intro=t["intro"], q=t["q"],
                        n=per_hit,
                        minutes=max(2, round(per_hit * 0.2))) + body + _TAIL
    path.write_text(html)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=RESULTS_V4 / "human_eval")
    ap.add_argument("--per-hit", type=int, default=20,
                    help="comparison items shown in a single HIT")
    ap.add_argument("--assignments", type=int, default=3,
                    help="independent workers per HIT (for the cost estimate)")
    ap.add_argument("--reward", type=float, default=1.00,
                    help="USD reward per assignment (for the cost estimate)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--base-url", default="https://YOUR-BUCKET.s3.amazonaws.com/images")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the selection without writing images")
    args = ap.parse_args()

    images_dir = args.out_dir / "images"
    if not args.dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)

    conds = all_conditions()
    print("Selecting...")
    items = build_items(conds, images_dir, args.size, args.quality, args.dry_run)

    # Coverage is reported per *reference image*, which for the pair conditions
    # means both endpoints of the pair, not the ordered pair itself.
    # Only the color / human pair conditions have a target reference that varies
    # independently of the source; style pairs are 1:1 (``fox`` <-> ``fox_real``),
    # so their target coverage is by construction equal to their source coverage.
    def tgt_of(c: Condition, eid: str) -> str | None:
        k = c.sub_key(eid)
        return k.split("|")[0] if "|" in k else None

    print(f"\n{'condition':34s} {'items':>6s} {'src refs':>9s} {'tgt refs':>9s} "
          f"{'variants':>9s} {'universe':>9s}")
    print("-" * 84)
    by_cond: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_cond[it.condition].append(it)
    for c in conds:
        got = by_cond.get(c.name, [])
        univ = c.universe()
        refs = f"{len({it.ref_key for it in got})}/{len({c.ref_key(e) for e in univ})}"
        tgts = ({tgt_of(c, it.entity_id) for it in got},
                {tgt_of(c, e) for e in univ}) if tgt_of(c, univ[0]) else None
        tgt_s = f"{len(tgts[0])}/{len(tgts[1])}" if tgts else "-"
        print(f"{c.name:34s} {len(got):6d} {refs:>9s} {tgt_s:>9s} "
              f"{len({it.sub_key for it in got}):9d} {len(univ):9d}")
    n_img = len({fn for it in items for fn in it.images.values()})
    print("-" * 60)
    print(f"{'TOTAL':34s} {len(items):6d} {'':6s} {n_img:6d} images")

    if args.dry_run:
        return

    write_manifest(items, args.out_dir / "manifest.csv")

    # One project per (method, family): six HTML layouts, six input CSVs. The
    # split is what lets each layout name one concrete property instead of
    # hedging "color / drawing style / the person's identity, whichever applies".
    groups = sorted({it.hit_group for it in items})
    # MTurk fee: 20% of the reward, +20% more when a HIT has >=10 assignments,
    # $0.01 minimum per assignment.
    fee_rate = 0.40 if args.assignments >= 10 else 0.20
    total_hits = 0
    total_slots = 0
    total_cost = 0.0
    print(f"\n{'project':16s} {'items':>6s} {'HITs':>5s} {'assign':>7s} "
          f"{'reward':>7s} {'cost':>8s}")
    print("-" * 54)
    for g in groups:
        # per_hit can shrink for a group with a small task pool; the layout is
        # built from the value the CSV actually used.
        n, eff = write_mturk_input(items, args.out_dir / f"mturk_input_{g}.csv",
                                   args.per_hit, args.base_url, g)
        write_hit_html(args.out_dir / f"hit_{g}.html", g, eff)
        # The knockout layout asks three questions per item, the patching
        # layout one, so they cannot carry the same reward.
        reward = args.reward
        assign = n * args.assignments
        cost = assign * (reward + max(0.01, reward * fee_rate))
        total_hits += n
        # Slots, not n * args.per_hit: a group with a small task pool shrinks
        # below the requested --per-hit, so the global value overcounts.
        total_slots += n * eff
        total_cost += cost
        print(f"{g:16s} {sum(1 for it in items if it.hit_group == g):6d} "
              f"{n:5d} {assign:7d} {reward:7.2f} {cost:8.2f}")
    print("-" * 54)
    print(f"{'TOTAL':16s} {len(items):6d} {total_hits:5d} "
          f"{total_hits * args.assignments:7d} {'':7s} {total_cost:8.2f}")
    pad = total_slots - len(items)
    print(f"\n{len(items)} items x {args.assignments} workers = "
          f"{len(items) * args.assignments} judgements on distinct items; "
          f"{total_slots * args.assignments} answers in total "
          f"({pad} padded slot{'s' if pad != 1 else ''} x {args.assignments})")
    print(f"\nWrote {args.out_dir}/")


if __name__ == "__main__":
    main()
