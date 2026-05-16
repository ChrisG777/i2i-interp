"""Registry of all paper-scale judges.

Each :class:`JudgeConfig` declares:

* ``name`` — identifier for the CLI ``--judge NAME`` flag.
* ``csv_path`` — output CSV under ``results_v4/vlm_judge/``.
* ``base_dir`` — per-entity result root (e.g. ``results_v4/i2i_to_unconditional/mm7_4step``).
* ``entity_ids()`` — function returning the full intended entity-id list
  (task_ids or pair-ids ``<src>__<tgt>``). The orchestrator filters this
  against the on-disk ``base_dir`` to skip entities that haven't been
  generated yet.
* ``bundle_builder`` — function ``(entity_id, base_dir) -> Bundle``.
* ``description`` — one-line summary used in the auto-generated README.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from experiments.common.tasks import load_tasks
from experiments.i2i_to_i2i_patching.pair_io import read_pair_list

from scripts.judge import bundles
from scripts.judge.bundles import Bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_V4 = REPO_ROOT / "results_v4"
JUDGE_DIR = RESULTS_V4 / "vlm_judge"
PAIR_LIST_DIR = REPO_ROOT / "slurm" / "i2i_to_i2i_patching"

T2I_UNC_MM7 = RESULTS_V4 / "i2i_to_unconditional" / "mm7_4step"
T2I_UNC_S9 = RESULTS_V4 / "i2i_to_unconditional" / "single9_1step"
KO_FULL = RESULTS_V4 / "attention_knockout" / "full_ko_4step"
I2I2I_COLOR = RESULTS_V4 / "i2i_to_i2i_patching" / "single9_4step_color"
I2I2I_STYLE = RESULTS_V4 / "i2i_to_i2i_patching" / "mm7_4step_style_to_real"
I2I2I_HUMANS = RESULTS_V4 / "i2i_to_i2i_patching" / "mm7_4step_dreambench_humans"


@dataclass(frozen=True)
class JudgeConfig:
    name: str
    description: str
    csv_path: Path
    base_dir: Path
    entity_ids: Callable[[], list[str]]
    bundle_builder: Callable[[str, Path], Bundle]


# ---------------------------------------------------------------------------
# Entity-id sources
# ---------------------------------------------------------------------------


def _solid_color_ids() -> list[str]:
    return sorted(t.task_id for t in load_tasks("solid_color"))


def _style_ids() -> list[str]:
    """Property-manual style ids (450). The stylized DreamBench++ subjects
    are NOT routed here — many of their prompts request a stylized output
    style ("anime style", "watercolor"), which contaminates the style
    judge: the patched output looks style-y because the prompt asked for
    it, not because of the patch."""
    return sorted(t.task_id for t in load_tasks("style"))


def _dreambench_human_ids() -> list[str]:
    """Real-photo human dreambench subjects only: 10 subjects * 9 prompts = 90 ids.
    The other 10 stylized subjects are routed through ``_style_ids`` instead."""
    return sorted(t.task_id for t in load_tasks("dreambench_humans"))


def _bucket_ids(bucket: str) -> Callable[[], list[str]]:
    """All task_ids for a bucket — what the reproduce scripts iterate via
    ``--bucket``."""
    return lambda: sorted(t.task_id for t in load_tasks(bucket))


def _pair_list_ids(path: Path) -> Callable[[], list[str]]:
    """Read pair-list file; entity_id is ``<src>__<tgt>`` per row."""
    def _f() -> list[str]:
        if not path.exists():
            return []
        return [f"{src}__{tgt}" for src, tgt in read_pair_list(path)]
    return _f


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

JUDGES: list[JudgeConfig] = [
    # 1. T2I-unc / Single-9 / 1-step / color
    JudgeConfig(
        name="i2i_unc_color_text_lens",
        description="Did patching the i2i text tokens into a clean t2i pass surface "
                    "the reference solid color in the generation? (1-step, single block 9.)",
        csv_path=JUDGE_DIR / "i2i_unc_color_text_lens.csv",
        base_dir=T2I_UNC_S9,
        entity_ids=_solid_color_ids,
        bundle_builder=bundles.i2i_unc_color_text_lens,
    ),
    # 2. T2I-unc / MM-7 / 4-step / style
    JudgeConfig(
        name="i2i_unc_style_text_lens",
        description="Did patching the i2i text tokens (MM 7, 4-step) into a clean t2i "
                    "pass shift the generation toward a style / cartoon style?",
        csv_path=JUDGE_DIR / "i2i_unc_style_text_lens.csv",
        base_dir=T2I_UNC_MM7,
        entity_ids=_style_ids,
        bundle_builder=bundles.i2i_unc_style_text_lens,
    ),
    # 3. T2I-unc / MM-7 / 4-step / dreambench humans
    JudgeConfig(
        name="i2i_unc_dreambench_human_identity",
        description="When patching i2i text tokens (MM 7, 4-step) for a dreambench "
                    "human reference, does the person in the patched t2i look like a "
                    "DIFFERENT individual than the reference?",
        csv_path=JUDGE_DIR / "i2i_unc_dreambench_human_identity.csv",
        base_dir=T2I_UNC_MM7,
        entity_ids=_dreambench_human_ids,
        bundle_builder=bundles.i2i_unc_dreambench_human_identity,
    ),
    # 4-9. Generic reference judge per edit_type / category.
    JudgeConfig(
        name="i2i_unc_add",
        description="Generic reference-info judge over the add bucket.",
        csv_path=JUDGE_DIR / "i2i_unc_add.csv",
        base_dir=T2I_UNC_MM7,
        entity_ids=_bucket_ids("add"),
        bundle_builder=bundles.i2i_unc_generic_reference,
    ),
    JudgeConfig(
        name="i2i_unc_remove",
        description="Generic reference-info judge over the remove bucket.",
        csv_path=JUDGE_DIR / "i2i_unc_remove.csv",
        base_dir=T2I_UNC_MM7,
        entity_ids=_bucket_ids("remove"),
        bundle_builder=bundles.i2i_unc_generic_reference,
    ),
    # 10-13. Attention-knockout judges (per direction, per task family).
    JudgeConfig(
        name="ko_color_ref_to_text",
        description="When ref->text is fully blocked (all layers, 4-step), did the "
                    "i2i lose the reference solid color?",
        csv_path=JUDGE_DIR / "ko_color_ref_to_text.csv",
        base_dir=KO_FULL,
        entity_ids=_solid_color_ids,
        bundle_builder=bundles.ko_color_ref_to_text,
    ),
    JudgeConfig(
        name="ko_color_ref_to_image",
        description="When ref->image is fully blocked (all layers, 4-step), did the "
                    "i2i KEEP the reference solid color?",
        csv_path=JUDGE_DIR / "ko_color_ref_to_image.csv",
        base_dir=KO_FULL,
        entity_ids=_solid_color_ids,
        bundle_builder=bundles.ko_color_ref_to_image,
    ),
    JudgeConfig(
        name="ko_style_ref_to_text",
        description="When ref->text is fully blocked (all layers, 4-step), did the "
                    "i2i become more realistic / less style-y?",
        csv_path=JUDGE_DIR / "ko_style_ref_to_text.csv",
        base_dir=KO_FULL,
        entity_ids=_style_ids,
        bundle_builder=bundles.ko_style_ref_to_text,
    ),
    JudgeConfig(
        name="ko_style_ref_to_image",
        description="When ref->image is fully blocked (all layers, 4-step), did the "
                    "i2i KEEP its style / cartoon style?",
        csv_path=JUDGE_DIR / "ko_style_ref_to_image.csv",
        base_dir=KO_FULL,
        entity_ids=_style_ids,
        bundle_builder=bundles.ko_style_ref_to_image,
    ),
    JudgeConfig(
        name="ko_dreambench_human_ref_to_text",
        description="When ref->text is fully blocked (all layers, 4-step) on a "
                    "dreambench human i2i, does the person in the i2i lose the "
                    "reference identity?",
        csv_path=JUDGE_DIR / "ko_dreambench_human_ref_to_text.csv",
        base_dir=KO_FULL,
        entity_ids=_dreambench_human_ids,
        bundle_builder=bundles.ko_dreambench_human_ref_to_text,
    ),
    JudgeConfig(
        name="ko_dreambench_human_ref_to_image",
        description="When ref->image is fully blocked (all layers, 4-step) on a "
                    "dreambench human i2i, does the person in the i2i KEEP the "
                    "reference identity?",
        csv_path=JUDGE_DIR / "ko_dreambench_human_ref_to_image.csv",
        base_dir=KO_FULL,
        entity_ids=_dreambench_human_ids,
        bundle_builder=bundles.ko_dreambench_human_ref_to_image,
    ),
    # Padding-only ref->text knockouts. Same answer key as ko_*_ref_to_text;
    # CSVs are directly comparable to test whether the padding positions of the
    # text are sufficient for the ref->text effect.
    JudgeConfig(
        name="ko_color_ref_to_text_padding",
        description="When ref->text[padding] is fully blocked (all layers, 4-step), "
                    "did the i2i lose the reference solid color?",
        csv_path=JUDGE_DIR / "ko_color_ref_to_text_padding.csv",
        base_dir=KO_FULL,
        entity_ids=_solid_color_ids,
        bundle_builder=bundles.ko_color_ref_to_text_padding,
    ),
    JudgeConfig(
        name="ko_style_ref_to_text_padding",
        description="When ref->text[padding] is fully blocked (all layers, 4-step), "
                    "did the i2i become more realistic / less style-y?",
        csv_path=JUDGE_DIR / "ko_style_ref_to_text_padding.csv",
        base_dir=KO_FULL,
        entity_ids=_style_ids,
        bundle_builder=bundles.ko_style_ref_to_text_padding,
    ),
    JudgeConfig(
        name="ko_dreambench_human_ref_to_text_padding",
        description="When ref->text[padding] is fully blocked (all layers, 4-step) "
                    "on a dreambench human i2i, does the person in the i2i lose "
                    "the reference identity?",
        csv_path=JUDGE_DIR / "ko_dreambench_human_ref_to_text_padding.csv",
        base_dir=KO_FULL,
        entity_ids=_dreambench_human_ids,
        bundle_builder=bundles.ko_dreambench_human_ref_to_text_padding,
    ),
    # Content-only ref->text knockouts. Same answer key as ko_*_ref_to_text;
    # CSVs are directly comparable to the full and padding-only siblings to
    # decompose the ref->text effect across content vs. padding positions.
    JudgeConfig(
        name="ko_color_ref_to_text_content",
        description="When ref->text[content] is fully blocked (all layers, 4-step), "
                    "did the i2i lose the reference solid color?",
        csv_path=JUDGE_DIR / "ko_color_ref_to_text_content.csv",
        base_dir=KO_FULL,
        entity_ids=_solid_color_ids,
        bundle_builder=bundles.ko_color_ref_to_text_content,
    ),
    JudgeConfig(
        name="ko_style_ref_to_text_content",
        description="When ref->text[content] is fully blocked (all layers, 4-step), "
                    "did the i2i become more realistic / less style-y?",
        csv_path=JUDGE_DIR / "ko_style_ref_to_text_content.csv",
        base_dir=KO_FULL,
        entity_ids=_style_ids,
        bundle_builder=bundles.ko_style_ref_to_text_content,
    ),
    JudgeConfig(
        name="ko_dreambench_human_ref_to_text_content",
        description="When ref->text[content] is fully blocked (all layers, 4-step) "
                    "on a dreambench human i2i, does the person in the i2i lose "
                    "the reference identity?",
        csv_path=JUDGE_DIR / "ko_dreambench_human_ref_to_text_content.csv",
        base_dir=KO_FULL,
        entity_ids=_dreambench_human_ids,
        bundle_builder=bundles.ko_dreambench_human_ref_to_text_content,
    ),
    # Padding-only and content-only T2I-unc lenses for the style axis only.
    # The color/dreambench/add/remove families don't have these variants in
    # the paper, so we only keep the style pair.
    JudgeConfig(
        name="i2i_unc_style_text_lens_padding",
        description="Did patching ONLY the text-padding tokens (MM 7, 4-step) into a "
                    "clean t2i pass shift the generation toward a style / cartoon style?",
        csv_path=JUDGE_DIR / "i2i_unc_style_text_lens_padding.csv",
        base_dir=T2I_UNC_MM7,
        entity_ids=_style_ids,
        bundle_builder=bundles.i2i_unc_style_text_lens_padding,
    ),
    JudgeConfig(
        name="i2i_unc_style_text_lens_content",
        description="Did patching ONLY the text-content tokens (MM 7, 4-step) into a "
                    "clean t2i pass shift the generation toward a style / cartoon style?",
        csv_path=JUDGE_DIR / "i2i_unc_style_text_lens_content.csv",
        base_dir=T2I_UNC_MM7,
        entity_ids=_style_ids,
        bundle_builder=bundles.i2i_unc_style_text_lens_content,
    ),
    # 14-16. i2i->i2i pair judges.
    JudgeConfig(
        name="i2i2i_color",
        description="Did patching text tokens from a SOURCE color i2i shift a "
                    "TARGET color i2i toward the source's color? (Single 9, 4-step.)",
        csv_path=JUDGE_DIR / "i2i2i_color.csv",
        base_dir=I2I2I_COLOR,
        entity_ids=_pair_list_ids(
            PAIR_LIST_DIR / "single9_4step_color" / "pairs.txt"
        ),
        bundle_builder=bundles.i2i2i_color,
    ),
    JudgeConfig(
        name="i2i2i_style",
        description="Did patching text tokens from a STYLE source into a real-photo "
                    "TARGET shift the target toward a style style? (MM 7, 4-step.)",
        csv_path=JUDGE_DIR / "i2i2i_style.csv",
        base_dir=I2I2I_STYLE,
        entity_ids=_pair_list_ids(
            PAIR_LIST_DIR / "mm7_4step_style_to_real" / "pairs.txt"
        ),
        bundle_builder=bundles.i2i2i_style,
    ),
    # Padding-only i2i->i2i pair judges (color, style). Same answer key.
    JudgeConfig(
        name="i2i2i_color_text_padding",
        description="Did patching ONLY the text-padding tokens from a SOURCE color "
                    "i2i shift a TARGET color i2i toward the source's color?",
        csv_path=JUDGE_DIR / "i2i2i_color_text_padding.csv",
        base_dir=I2I2I_COLOR,
        entity_ids=_pair_list_ids(
            PAIR_LIST_DIR / "single9_4step_color" / "pairs.txt"
        ),
        bundle_builder=bundles.i2i2i_color_text_padding,
    ),
    JudgeConfig(
        name="i2i2i_style_text_padding",
        description="Did patching ONLY the text-padding tokens from a STYLE source "
                    "into a real-photo TARGET shift the target toward a style style?",
        csv_path=JUDGE_DIR / "i2i2i_style_text_padding.csv",
        base_dir=I2I2I_STYLE,
        entity_ids=_pair_list_ids(
            PAIR_LIST_DIR / "mm7_4step_style_to_real" / "pairs.txt"
        ),
        bundle_builder=bundles.i2i2i_style_text_padding,
    ),
    JudgeConfig(
        name="i2i2i_dreambench_humans",
        description="When patching text tokens from human-A's i2i into human-B's "
                    "target, does B end up looking like A? (Hopeful answer: 0.)",
        csv_path=JUDGE_DIR / "i2i2i_dreambench_humans.csv",
        base_dir=I2I2I_HUMANS,
        entity_ids=_pair_list_ids(
            PAIR_LIST_DIR / "mm7_4step_dreambench_humans" / "pairs.txt"
        ),
        bundle_builder=bundles.i2i2i_dreambench_humans,
    ),
    JudgeConfig(
        name="i2i2i_dreambench_humans_text_padding",
        description="When patching ONLY the text-padding tokens from human-A's i2i "
                    "into human-B's target, does B end up looking like A?",
        csv_path=JUDGE_DIR / "i2i2i_dreambench_humans_text_padding.csv",
        base_dir=I2I2I_HUMANS,
        entity_ids=_pair_list_ids(
            PAIR_LIST_DIR / "mm7_4step_dreambench_humans" / "pairs.txt"
        ),
        bundle_builder=bundles.i2i2i_dreambench_humans_text_padding,
    ),
    # Content-only i2i->i2i pair judges. Same answer key as i2i2i_*; CSVs are
    # directly comparable to the full and padding-only siblings.
    JudgeConfig(
        name="i2i2i_color_text_content",
        description="Did patching ONLY the text-content tokens from a SOURCE color "
                    "i2i shift a TARGET color i2i toward the source's color?",
        csv_path=JUDGE_DIR / "i2i2i_color_text_content.csv",
        base_dir=I2I2I_COLOR,
        entity_ids=_pair_list_ids(
            PAIR_LIST_DIR / "single9_4step_color" / "pairs.txt"
        ),
        bundle_builder=bundles.i2i2i_color_text_content,
    ),
    JudgeConfig(
        name="i2i2i_style_text_content",
        description="Did patching ONLY the text-content tokens from a STYLE source "
                    "into a real-photo TARGET shift the target toward a style style?",
        csv_path=JUDGE_DIR / "i2i2i_style_text_content.csv",
        base_dir=I2I2I_STYLE,
        entity_ids=_pair_list_ids(
            PAIR_LIST_DIR / "mm7_4step_style_to_real" / "pairs.txt"
        ),
        bundle_builder=bundles.i2i2i_style_text_content,
    ),
    JudgeConfig(
        name="i2i2i_dreambench_humans_text_content",
        description="When patching ONLY the text-content tokens from human-A's i2i "
                    "into human-B's target, does B end up looking like A?",
        csv_path=JUDGE_DIR / "i2i2i_dreambench_humans_text_content.csv",
        base_dir=I2I2I_HUMANS,
        entity_ids=_pair_list_ids(
            PAIR_LIST_DIR / "mm7_4step_dreambench_humans" / "pairs.txt"
        ),
        bundle_builder=bundles.i2i2i_dreambench_humans_text_content,
    ),
]

JUDGES_BY_NAME: dict[str, JudgeConfig] = {j.name: j for j in JUDGES}


def get(name: str) -> JudgeConfig:
    assert name in JUDGES_BY_NAME, (
        f"Unknown judge {name!r}. Known: {sorted(JUDGES_BY_NAME)}"
    )
    return JUDGES_BY_NAME[name]


# ---------------------------------------------------------------------------
# Judge groups — sets of judges that share a leading (label, image) prefix
# per entity_id. Running members back-to-back per entity lets Anthropic's
# prompt cache absorb the shared images. ``prefix_len`` is the number of
# (label, image) pairs that are byte-identical across members; the
# orchestrator marks ``image_block(..., cache=True)`` on the last one so the
# cache key spans the system prompt + those first prefix_len images.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeGroup:
    name: str
    members: tuple[str, ...]
    prefix_len: int


JUDGE_GROUPS: list[JudgeGroup] = [
    JudgeGroup(
        name="ko_color",
        members=(
            "ko_color_ref_to_text",
            "ko_color_ref_to_text_padding",
            "ko_color_ref_to_text_content",
            "ko_color_ref_to_image",
        ),
        prefix_len=2,
    ),
    JudgeGroup(
        name="ko_style",
        members=(
            "ko_style_ref_to_text",
            "ko_style_ref_to_text_padding",
            "ko_style_ref_to_text_content",
            "ko_style_ref_to_image",
        ),
        prefix_len=2,
    ),
    JudgeGroup(
        name="ko_dreambench_humans",
        members=(
            "ko_dreambench_human_ref_to_text",
            "ko_dreambench_human_ref_to_text_padding",
            "ko_dreambench_human_ref_to_text_content",
            "ko_dreambench_human_ref_to_image",
        ),
        prefix_len=2,
    ),
    JudgeGroup(
        name="t2i_unc_style",
        members=(
            "i2i_unc_style_text_lens",
            "i2i_unc_style_text_lens_padding",
            "i2i_unc_style_text_lens_content",
        ),
        prefix_len=2,
    ),
    JudgeGroup(
        name="i2i2i_color",
        members=("i2i2i_color", "i2i2i_color_text_padding", "i2i2i_color_text_content"),
        prefix_len=3,
    ),
    JudgeGroup(
        name="i2i2i_style",
        members=("i2i2i_style", "i2i2i_style_text_padding", "i2i2i_style_text_content"),
        prefix_len=3,
    ),
    JudgeGroup(
        name="i2i2i_dreambench_humans",
        members=(
            "i2i2i_dreambench_humans",
            "i2i2i_dreambench_humans_text_padding",
            "i2i2i_dreambench_humans_text_content",
        ),
        prefix_len=3,
    ),
]

JUDGE_GROUPS_BY_NAME: dict[str, JudgeGroup] = {g.name: g for g in JUDGE_GROUPS}


def get_group(name: str) -> JudgeGroup:
    assert name in JUDGE_GROUPS_BY_NAME, (
        f"Unknown judge group {name!r}. Known: {sorted(JUDGE_GROUPS_BY_NAME)}"
    )
    return JUDGE_GROUPS_BY_NAME[name]
