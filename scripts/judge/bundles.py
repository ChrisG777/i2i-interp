"""Per-judge bundle builders.

Each ``Bundle`` packages the (image_labels, image_paths, question) tuple
that the API call needs. Bundle builders take an ``entity_id`` (a task_id
for per-task judges, a pair_id ``<source>__<target>`` for pair judges)
plus a ``base_dir`` that points at the per-entity result directory.

The exact prompt wording for each judge is documented in
:mod:`scripts.judge.configs` (and copied into the auto-generated
``results_v2/vlm_judge/README.md``).

The ``ref->text`` knockout judges and the i2i->i2i pair judges each come
in three sibling variants — full, padding-only, content-only — that share
prompt + question wording and differ only in which patched / knocked-out
cell they read. The two factories below collapse the per-family triplet
into a single closure switched on a ``variant`` tag; module-level names
(``ko_color_ref_to_text``, ``ko_color_ref_to_text_padding``, etc.) are
preserved as factory-bound callables so :mod:`scripts.judge.configs` can
keep referencing them by name without change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal


@dataclass(frozen=True)
class Bundle:
    image_labels: list[str]
    image_paths: list[Path]
    question: str


# ---------------------------------------------------------------------------
# Variant tables — one entry per (variant) for KO ref->text and i2i->i2i pair
# patching. The two families have parallel triplet shapes (full / padding /
# content), differing only in the patched-cell filename and a short clause
# embedded in the prompt label / question.
# ---------------------------------------------------------------------------


Variant = Literal["full", "padding", "content"]


@dataclass(frozen=True)
class _KOVariant:
    filename: str
    label_clause: str   # interpolated into "Image 3 - same i2i, but {clause}:"
    setting_name: str   # interpolated into the question parenthetical


_KO_VARIANTS: dict[Variant, _KOVariant] = {
    "full": _KOVariant(
        filename="ref_to_text_full_ko.png",
        label_clause="with the ref->text attention path completely blocked",
        setting_name="ref->text",
    ),
    "padding": _KOVariant(
        filename="ref_to_text_padding_full_ko.png",
        label_clause="with ref->text attention blocked ONLY at the Qwen3 text-padding positions",
        setting_name="ref->text[padding]",
    ),
    "content": _KOVariant(
        filename="ref_to_text_content_full_ko.png",
        label_clause="with ref->text attention blocked ONLY at the Qwen3 text-content positions",
        setting_name="ref->text[content]",
    ),
}


@dataclass(frozen=True)
class _I2I2IVariant:
    filename: str
    action_clause: str   # interpolated into "Image 4 - same target generation, but {clause} <source-suffix>:"


_I2I2I_VARIANTS: dict[Variant, _I2I2IVariant] = {
    "full": _I2I2IVariant(
        filename="patched.png",
        action_clause="with text-token activations patched",
    ),
    "padding": _I2I2IVariant(
        filename="patched_text_padding.png",
        action_clause="with ONLY the text-padding tokens patched",
    ),
    "content": _I2I2IVariant(
        filename="patched_text_content.png",
        action_clause="with ONLY the text-content tokens patched",
    ),
}


@dataclass(frozen=True)
class _T2ILensVariant:
    filename_suffix: str   # appended to per-family base, before .png
    token_phrase: str      # leading phrase: "with text tokens" / "with ONLY text-padding tokens" / "with ONLY text-content tokens"


_T2I_LENS_VARIANTS: dict[Variant, _T2ILensVariant] = {
    "full": _T2ILensVariant(
        filename_suffix="",
        token_phrase="with text tokens",
    ),
    "padding": _T2ILensVariant(
        filename_suffix="_text_padding",
        token_phrase="with ONLY text-padding tokens",
    ),
    "content": _T2ILensVariant(
        filename_suffix="_text_content",
        token_phrase="with ONLY text-content tokens",
    ),
}


# ---------------------------------------------------------------------------
# i2i->T2I-unconditional judges (one patched cell per task)
# ---------------------------------------------------------------------------


def _i2i_unc_color_text_lens(variant: Variant) -> Callable[[str, Path], Bundle]:
    v = _T2I_LENS_VARIANTS[variant]
    def builder(entity_id: str, base_dir: Path) -> Bundle:
        d = base_dir / entity_id
        return Bundle(
            image_labels=[
                "Image 1 - solid color reference:",
                "Image 2 - text-to-image baseline (no patching) of the prompt:",
                f"Image 3 - same generation, but {v.token_phrase} patched in "
                "from a reference-conditioned i2i pass:",
            ],
            image_paths=[
                d / "reference.png", d / "t2i_clean.png",
                d / f"patched_single_mm9{v.filename_suffix}.png",
            ],
            question=(
                "Compared to Image 2, does Image 3 take on the predominant solid "
                "color of Image 1? Reply 1 if the color of Image 1 is now visibly "
                "present in Image 3 (and was not in Image 2)."
            ),
        )
    return builder


i2i_unc_color_text_lens = _i2i_unc_color_text_lens("full")


def _i2i_unc_style_text_lens(variant: Variant) -> Callable[[str, Path], Bundle]:
    v = _T2I_LENS_VARIANTS[variant]
    def builder(entity_id: str, base_dir: Path) -> Bundle:
        d = base_dir / entity_id
        return Bundle(
            image_labels=[
                "Image 1 - style / cartoon reference:",
                "Image 2 - clean text-to-image baseline:",
                f"Image 3 - same generation, but {v.token_phrase} patched from "
                "a reference-conditioned i2i pass:",
            ],
            image_paths=[
                d / "reference.png", d / "t2i_clean_4step.png",
                d / f"patched_mm7{v.filename_suffix}.png",
            ],
            question=(
                "Compared to Image 2, does Image 3 adopt a style / cartoon / "
                "illustrated / unrealistic style similar to Image 1? Look at the "
                "subject and the background / rest of the image - style-y style "
                "anywhere in the image counts as evidence. Reply 1 if Image 3 "
                "looks more style-like / less photographic than Image 2."
            ),
        )
    return builder


i2i_unc_style_text_lens = _i2i_unc_style_text_lens("full")
i2i_unc_style_text_lens_padding = _i2i_unc_style_text_lens("padding")
i2i_unc_style_text_lens_content = _i2i_unc_style_text_lens("content")


def _i2i_unc_dreambench_human_identity(variant: Variant) -> Callable[[str, Path], Bundle]:
    v = _T2I_LENS_VARIANTS[variant]
    def builder(entity_id: str, base_dir: Path) -> Bundle:
        d = base_dir / entity_id
        return Bundle(
            image_labels=[
                "Image 1 - reference photo of a person:",
                "Image 2 - text-to-image baseline of the same scene description:",
                f"Image 3 - same generation, but {v.token_phrase} patched from "
                "the reference-conditioned i2i pass:",
            ],
            image_paths=[
                d / "reference.png", d / "t2i_clean_4step.png",
                d / f"patched_mm7{v.filename_suffix}.png",
            ],
            question=(
                "Focus on the person in Image 3. Is the person in Image 3 a "
                "recognizably DIFFERENT individual (different face, hair, build, "
                "identity) from the person in Image 1? Reply 1 if a viewer would "
                "say it is a different person; reply 0 only if it is the same person."
            ),
        )
    return builder


i2i_unc_dreambench_human_identity = _i2i_unc_dreambench_human_identity("full")


def _i2i_unc_generic_reference(variant: Variant) -> Callable[[str, Path], Bundle]:
    """Shared template for add / remove / replace / customize_dreambench_nonhuman /
    customize_dreambooth / customize_property_misc judges. Per user feedback
    these all use the same wording — one CSV per category for parallel
    SLURM writes, same question text."""
    v = _T2I_LENS_VARIANTS[variant]
    def builder(entity_id: str, base_dir: Path) -> Bundle:
        d = base_dir / entity_id
        return Bundle(
            image_labels=[
                "Image 1 - reference image used to condition the source i2i pass:",
                "Image 2 - clean text-to-image baseline of the prompt:",
                f"Image 3 - same generation, but {v.token_phrase} patched from "
                "the reference-conditioned i2i pass:",
            ],
            image_paths=[
                d / "reference.png", d / "t2i_clean_4step.png",
                d / f"patched_mm7{v.filename_suffix}.png",
            ],
            question=(
                "Compared to Image 2, does Image 3 contain ANY visible information "
                "drawn from Image 1 - things like colors, textures, layout, style, "
                "distinctive shapes, or specific subject features? Reply 1 if you "
                "can point to anything in Image 3 that came from Image 1 and was "
                "not in Image 2."
            ),
        )
    return builder


i2i_unc_generic_reference = _i2i_unc_generic_reference("full")


# ---------------------------------------------------------------------------
# Attention-knockout judges (full-KO 4-step on each direction)
# ---------------------------------------------------------------------------


def _ko_color_ref_to_text(variant: Variant) -> Callable[[str, Path], Bundle]:
    v = _KO_VARIANTS[variant]
    def builder(entity_id: str, base_dir: Path) -> Bundle:
        d = base_dir / entity_id
        return Bundle(
            image_labels=[
                "Image 1 - solid color reference:",
                "Image 2 - clean reference-conditioned i2i baseline:",
                f"Image 3 - same i2i, but {v.label_clause}:",
            ],
            image_paths=[
                d / "reference.png", d / "i2i_baseline_4step.png",
                d / v.filename,
            ],
            question=(
                "Compared to Image 2, has Image 3 LOST the predominant solid color "
                "of Image 1? Reply 1 if the color is significantly removed (color "
                f"depended on {v.setting_name})."
            ),
        )
    return builder


ko_color_ref_to_text = _ko_color_ref_to_text("full")
ko_color_ref_to_text_padding = _ko_color_ref_to_text("padding")
ko_color_ref_to_text_content = _ko_color_ref_to_text("content")


def ko_color_ref_to_image(entity_id: str, base_dir: Path) -> Bundle:
    d = base_dir / entity_id
    return Bundle(
        image_labels=[
            "Image 1 - solid color reference:",
            "Image 2 - clean reference-conditioned i2i baseline:",
            "Image 3 - same i2i, but with the ref->image attention path completely blocked:",
        ],
        image_paths=[
            d / "reference.png", d / "i2i_baseline_4step.png",
            d / "ref_to_image_full_ko.png",
        ],
        question=(
            "Compared to Image 2, does Image 3 STILL show the predominant "
            "solid color of Image 1? Reply 1 if the color is preserved (color "
            "survives blocking ref->image)."
        ),
    )


def _ko_style_ref_to_text(variant: Variant) -> Callable[[str, Path], Bundle]:
    v = _KO_VARIANTS[variant]
    def builder(entity_id: str, base_dir: Path) -> Bundle:
        d = base_dir / entity_id
        return Bundle(
            image_labels=[
                "Image 1 - style reference:",
                "Image 2 - clean reference-conditioned i2i baseline:",
                f"Image 3 - same i2i, but {v.label_clause}:",
            ],
            image_paths=[
                d / "reference.png", d / "i2i_baseline_4step.png",
                d / v.filename,
            ],
            question=(
                "Compared to Image 2, has Image 3 LOST the style / cartoon style "
                "of Image 1 and become more photographic / realistic? Reply 1 if "
                f"Image 3 became more realistic when {v.setting_name} was blocked."
            ),
        )
    return builder


ko_style_ref_to_text = _ko_style_ref_to_text("full")
ko_style_ref_to_text_padding = _ko_style_ref_to_text("padding")
ko_style_ref_to_text_content = _ko_style_ref_to_text("content")


def ko_style_ref_to_image(entity_id: str, base_dir: Path) -> Bundle:
    d = base_dir / entity_id
    return Bundle(
        image_labels=[
            "Image 1 - style reference:",
            "Image 2 - clean reference-conditioned i2i baseline:",
            "Image 3 - same i2i, but with the ref->image attention path completely blocked:",
        ],
        image_paths=[
            d / "reference.png", d / "i2i_baseline_4step.png",
            d / "ref_to_image_full_ko.png",
        ],
        question=(
            "Compared to Image 2, does Image 3 STILL look style / cartoon-like "
            "(similar style to Image 1)? Look at the subject and the background "
            "/ rest of the image - style-y style anywhere in the image counts "
            "as evidence. Reply 1 if the style style was preserved when "
            "ref->image was blocked."
        ),
    )


def _ko_dreambench_human_ref_to_text(variant: Variant) -> Callable[[str, Path], Bundle]:
    v = _KO_VARIANTS[variant]
    def builder(entity_id: str, base_dir: Path) -> Bundle:
        d = base_dir / entity_id
        return Bundle(
            image_labels=[
                "Image 1 - reference photo of a person:",
                "Image 2 - clean reference-conditioned i2i baseline (this should look "
                "like the person in Image 1):",
                f"Image 3 - same i2i, but {v.label_clause}:",
            ],
            image_paths=[
                d / "reference.png", d / "i2i_baseline_4step.png",
                d / v.filename,
            ],
            question=(
                "Focus on the person in Image 3. Compared to Image 2, has Image 3 "
                "LOST the identity of the person in Image 1 - i.e. does the person "
                "in Image 3 look like a recognizably DIFFERENT individual (different "
                "face, hair, build) from the person in Image 1? Reply 1 if blocking "
                f"{v.setting_name} destroyed the reference identity; reply 0 if Image 3 "
                "still looks like the same person as Image 1."
            ),
        )
    return builder


ko_dreambench_human_ref_to_text = _ko_dreambench_human_ref_to_text("full")
ko_dreambench_human_ref_to_text_padding = _ko_dreambench_human_ref_to_text("padding")
ko_dreambench_human_ref_to_text_content = _ko_dreambench_human_ref_to_text("content")


def ko_dreambench_human_ref_to_image(entity_id: str, base_dir: Path) -> Bundle:
    d = base_dir / entity_id
    return Bundle(
        image_labels=[
            "Image 1 - reference photo of a person:",
            "Image 2 - clean reference-conditioned i2i baseline (this should look "
            "like the person in Image 1):",
            "Image 3 - same i2i, but with the ref->image attention path completely blocked:",
        ],
        image_paths=[
            d / "reference.png", d / "i2i_baseline_4step.png",
            d / "ref_to_image_full_ko.png",
        ],
        question=(
            "Focus on the person in Image 3. Does Image 3 STILL look like the "
            "same individual as Image 1 (same face, hair, build)? Reply 1 if "
            "the reference identity is preserved despite blocking ref->image; "
            "reply 0 if Image 3 now looks like a different person from Image 1."
        ),
    )


# ---------------------------------------------------------------------------
# i2i->i2i patching judges (per-pair). Three sibling variants per family —
# full / padding / content — share question + Image 1-3 labels and differ
# only in the patched-cell filename and a clause inside the Image 4 label.
# ---------------------------------------------------------------------------


def _i2i2i_color(variant: Variant) -> Callable[[str, Path], Bundle]:
    v = _I2I2I_VARIANTS[variant]
    def builder(entity_id: str, base_dir: Path) -> Bundle:
        d = base_dir / entity_id
        return Bundle(
            image_labels=[
                "Image 1 - solid color reference for the SOURCE i2i pass:",
                "Image 2 - solid color reference for the TARGET i2i pass:",
                "Image 3 - clean i2i generation conditioned on Image 2 (no patching):",
                f"Image 4 - same target generation, but {v.action_clause} "
                "from the SOURCE i2i pass:",
            ],
            image_paths=[
                d / "ref_source.png", d / "ref_target.png",
                d / "target_baseline_4step.png", d / v.filename,
            ],
            question=(
                "Compared to Image 3 (which should show the color of Image 2), "
                "does Image 4 take on the color of Image 1 (the source) instead? "
                "Reply 1 if Image 4 is more like Image 1's color than Image 2's."
            ),
        )
    return builder


i2i2i_color = _i2i2i_color("full")
i2i2i_color_text_padding = _i2i2i_color("padding")
i2i2i_color_text_content = _i2i2i_color("content")


def _i2i2i_style(variant: Variant) -> Callable[[str, Path], Bundle]:
    v = _I2I2I_VARIANTS[variant]
    def builder(entity_id: str, base_dir: Path) -> Bundle:
        d = base_dir / entity_id
        return Bundle(
            image_labels=[
                "Image 1 - style SOURCE reference:",
                "Image 2 - real-photo TARGET reference (same subject):",
                "Image 3 - clean i2i generation conditioned on Image 2:",
                f"Image 4 - same target generation, but {v.action_clause} "
                "from the SOURCE style i2i pass:",
            ],
            image_paths=[
                d / "ref_source.png", d / "ref_target.png",
                d / "target_baseline_4step.png", d / v.filename,
            ],
            question=(
                "Compared to Image 3, has Image 4 become MORE style / cartoon / "
                "unrealistic in style (matching Image 1)? Look at the subject and "
                "the background / rest of the image - style-y style anywhere in "
                "the image counts as evidence. Reply 1 if Image 4 looks more "
                "style-like than Image 3."
            ),
        )
    return builder


i2i2i_style = _i2i2i_style("full")
i2i2i_style_text_padding = _i2i2i_style("padding")
i2i2i_style_text_content = _i2i2i_style("content")


def _i2i2i_dreambench_humans(variant: Variant) -> Callable[[str, Path], Bundle]:
    v = _I2I2I_VARIANTS[variant]
    def builder(entity_id: str, base_dir: Path) -> Bundle:
        d = base_dir / entity_id
        return Bundle(
            image_labels=[
                "Image 1 - photo of person A (SOURCE i2i reference):",
                "Image 2 - photo of person B (TARGET i2i reference):",
                "Image 3 - clean i2i generation conditioned on person B:",
                f"Image 4 - same target generation, but {v.action_clause} "
                "from person A's i2i pass:",
            ],
            image_paths=[
                d / "ref_source.png", d / "ref_target.png",
                d / "target_baseline_4step.png", d / v.filename,
            ],
            question=(
                "Focus on the person in Image 4. Does the person in Image 4 look "
                "more like person A (Image 1, the source) than like person B "
                "(Image 2, the target)? Reply 1 if A's identity transferred over."
            ),
        )
    return builder


i2i2i_dreambench_humans = _i2i2i_dreambench_humans("full")
i2i2i_dreambench_humans_text_padding = _i2i2i_dreambench_humans("padding")
i2i2i_dreambench_humans_text_content = _i2i2i_dreambench_humans("content")
