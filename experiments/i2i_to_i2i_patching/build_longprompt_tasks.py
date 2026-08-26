"""Build the long-prompt task buckets for the padding-token ablation.

The i2i->i2i experiment suggests the *padding tokens* of the 512-long Qwen3
text sequence carry the reference's color/style. To test that, we re-run the
known-working color and style pairs with **long** instructions that fill the
text budget (≈0 padding at seq-len 512), then again at seq-len 1024 (padding
restored). This script materializes the long-prompt tasks by reusing the exact
``real_ref_name`` / ``real_ref_dir`` / ``noise_seed`` / dims of the existing
short-prompt tasks and swapping in a long instruction.

Each long instruction = the original short instruction (kept verbatim, at the
front so it survives truncation at 512) + a long, deliberately *neutral* filler
that references "the color of this reference image" / "the style of this
reference image" generically but never names the actual color or style.

Outputs (committed artifacts; regenerate with this script):

* ``data/tasks/solid_color_longprompt/tasks.jsonl``
* ``data/tasks/style_longprompt/tasks.jsonl``

Usage::

    uv run python -m experiments.i2i_to_i2i_patching.build_longprompt_tasks
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.common.tasks import TASKS_ROOT, load_tasks

# ---------------------------------------------------------------------------
# Smoke set selection
# ---------------------------------------------------------------------------

# Color: two objects, three colors, all five seeds. Directed (color_a, color_b)
# cross-pairs within an object (built by build_pairs_longprompt.py).
COLOR_OBJECTS = ("ball", "mug")
COLOR_NAMES = ("solid_red", "solid_green", "solid_blue")
NUM_SEEDS = 5

# Style: two (subject, prompt_slug) combos. Each has a style source bucket
# entry (real_ref_name=<subject>) and a manual real entry
# (real_ref_name=<subject>_real); both get the same long instruction.
STYLE_SPECS = (
    ("alice_in_wonderland", "tea_party"),
    ("dog", "frisbee"),
)

# ---------------------------------------------------------------------------
# Long-instruction filler (neutral: never names the actual color / style)
# ---------------------------------------------------------------------------

# {object} is substituted per color task. No specific color words appear.
COLOR_FILLER = (
    "The single most important requirement of this task is to look carefully at "
    "the reference image provided and identify the color that it shows, and then "
    "reproduce that exact same color faithfully across the entire surface of the "
    "{object} that you draw. Treat the reference image purely as a color sample: "
    "it exists only to tell you which color to use. Match the hue, the saturation, and the overall "
    "tone of the reference image so that anyone comparing the two would agree "
    "that the {object} clearly takes on the color of the reference image. Apply "
    "the color evenly and uniformly, covering the whole object from edge to edge "
    "without leaving any part untouched, and keep the color of the reference "
    "image consistent across every visible region of the {object}. "
    "Beyond the color requirement, the rest of the picture should be kept clean, "
    "simple, and uncluttered so that nothing distracts from the object itself. "
    "Compose the image as a straightforward studio product photograph of a "
    "single {object} placed near the center of the frame, resting on a plain, "
    "smooth, seamless backdrop with an unbroken surface beneath it. Use soft, "
    "even, diffuse lighting that wraps gently around the form, so that the shape "
    "and contours of the {object} read clearly and the surface is easy to "
    "perceive, with only a subtle, soft contact shadow grounding the object on "
    "the floor. Photograph the {object} at a comfortable three-quarter viewing "
    "angle that shows its overall shape and proportions in a natural and honest "
    "way, using a moderate focal length that avoids any distortion and keeps the "
    "entire object in sharp focus from the nearest edge to the farthest edge. "
    "Keep the background completely free of any other objects, props, text, "
    "writing, logos, patterns, or decorations, so that absolutely nothing "
    "competes with the subject for attention. Maintain realistic proportions, a "
    "stable and level horizon, and an honest sense of scale, exactly as you "
    "would expect from a careful catalogue photograph that is meant to present a "
    "single product on its own. The material of the {object} should look smooth "
    "and believable, with a gentle and natural falloff of illumination from the "
    "more illuminated side toward the more shaded side, and a single soft "
    "highlight that suggests one broad, gentle light source positioned above. "
    "Above everything else, remember and repeat to yourself the central "
    "instruction of this task: take the color shown in the reference image and "
    "paint the entire {object} in that very same color, so that the finished "
    "{object} unmistakably carries the color of the reference image while still "
    "looking like a natural, photographic depiction of a real {object}."
)

# {subject_kind} is substituted per style task. No specific style words appear.
STYLE_FILLER = (
    "Pay very close attention to the reference image that has been provided, "
    "because the most important part of this task is to capture and reproduce "
    "the overall visual style of that reference image. Study the way the "
    "reference image is rendered, including its artistic look, its level of "
    "detail, the character of its lines and edges, the way its surfaces and "
    "textures are depicted, and the general aesthetic and visual treatment it "
    "uses, and then carry that same style faithfully into the picture you "
    "produce. "
    "While preserving the style of the reference image, keep the identity and "
    "recognizable features of the {subject_kind} consistent and believable, so "
    "that the {subject_kind} remains the clear subject of the scene. Place the "
    "{subject_kind} naturally within the setting described, with a sensible and "
    "stable composition, a coherent background, and a believable sense of depth "
    "and space. Keep the framing comfortable and uncluttered, with the "
    "{subject_kind} positioned clearly in the scene and the surrounding "
    "environment supporting the action without overwhelming it. Maintain "
    "consistent lighting across the whole image, with a single coherent light "
    "direction and natural, gentle falloff, so that the {subject_kind} and the "
    "environment feel like they belong together in one unified picture. "
    "Avoid adding any text, writing, captions, watermarks, logos, borders, or "
    "other distracting elements that would not belong in the scene. Keep "
    "proportions believable and the overall arrangement natural and easy to "
    "read, with the {subject_kind} and the setting rendered at an honest and "
    "consistent scale. The whole composition should feel intentional and "
    "harmonious, with every part of the image rendered in the same consistent "
    "manner. Be deliberate and thorough about applying the style of the "
    "reference image to every region of the picture, including the {subject_kind}, "
    "the foreground, the background, and any incidental details, so that no part "
    "of the image looks like it was rendered in a different manner from the rest. "
    "Take your time to study the reference image again before you begin, and let "
    "its overall visual style guide every choice you make about how to render the "
    "scene, from the largest shapes down to the smallest details, keeping "
    "everything unified, coherent, and faithful to the appearance of the "
    "reference image throughout. "
    "Above all else, remember the central instruction of this task and "
    "repeat it to yourself: look at the reference image, identify its overall "
    "visual style, and reproduce that very same style throughout the entire "
    "picture, so that the finished image unmistakably carries the style of the "
    "reference image while still clearly showing the {subject_kind} in the "
    "described scene."
)


def _lp_id(task_id: str) -> str:
    """Insert a ``_lp`` marker before the trailing ``_s<i>`` seed slot."""
    base, seed = task_id.rsplit("_s", 1)
    return f"{base}_lp_s{seed}"


def _long_instruction(short: str, filler: str, **subs: str) -> str:
    """Original short instruction (kept at the front) + neutral filler."""
    return f"{short}. {filler.format(**subs)}" if not short.endswith(".") \
        else f"{short} {filler.format(**subs)}"


def _row(task: object, instruction: str, metadata_extra: dict) -> dict:
    """Serialize a TaskDefinition-derived long-prompt row, reusing the source
    task's ref + seed exactly so the long-prompt run mirrors the short one."""
    t = task
    md = dict(t.metadata)  # type: ignore[attr-defined]
    md.update(metadata_extra)
    md["prompt_variant"] = "longprompt"
    md["short_instruction"] = t.instruction  # type: ignore[attr-defined]
    return {
        "task_id": _lp_id(t.task_id),  # type: ignore[attr-defined]
        "edit_type": t.edit_type,  # type: ignore[attr-defined]
        "source": t.source,  # type: ignore[attr-defined]
        "instruction": instruction,
        "source_image_path": t.source_image_path,  # type: ignore[attr-defined]
        "source_caption": t.source_caption,  # type: ignore[attr-defined]
        "ref_seed": t.ref_seed,  # type: ignore[attr-defined]
        "noise_seed": t.noise_seed,  # type: ignore[attr-defined]
        "real_ref_name": t.real_ref_name,  # type: ignore[attr-defined]
        "real_ref_dir": t.real_ref_dir,  # type: ignore[attr-defined]
        "height": t.height,  # type: ignore[attr-defined]
        "width": t.width,  # type: ignore[attr-defined]
        "metadata": md,
    }


def build_color_rows() -> list[dict]:
    by_id = {t.task_id: t for t in load_tasks("solid_color")}
    rows: list[dict] = []
    for obj in COLOR_OBJECTS:
        instr_long = _long_instruction(
            f"draw a {obj} in this color", COLOR_FILLER, object=obj,
        )
        for color in COLOR_NAMES:
            for i in range(NUM_SEEDS):
                src_id = f"{color}_{obj}_s{i}"
                assert src_id in by_id, f"missing solid_color task: {src_id}"
                rows.append(_row(by_id[src_id], instr_long, {"object": obj}))
    return rows


def build_style_rows() -> list[dict]:
    style = {t.task_id: t for t in load_tasks("style")}
    manual = {t.task_id: t for t in load_tasks("manual")}
    rows: list[dict] = []
    for subject, slug in STYLE_SPECS:
        # Pull the shared short instruction + subject_kind from the s0 entries.
        src0 = style[f"customize_property_style_free_{subject}_{slug}_s0"]
        subject_kind = src0.metadata["subject_kind"]
        instr_long = _long_instruction(
            src0.instruction, STYLE_FILLER, subject_kind=subject_kind,
        )
        for i in range(NUM_SEEDS):
            src_id = f"customize_property_style_free_{subject}_{slug}_s{i}"
            tgt_id = f"manual_free_{subject}_real_{slug}_s{i}"
            assert src_id in style, f"missing style task: {src_id}"
            assert tgt_id in manual, f"missing manual task: {tgt_id}"
            rows.append(_row(style[src_id], instr_long, {}))
            rows.append(_row(manual[tgt_id], instr_long, {}))
    return rows


def _write(bucket: str, rows: list[dict]) -> None:
    out = TASKS_ROOT / bucket / "tasks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} tasks to {out}")


def main() -> None:
    _write("solid_color_longprompt", build_color_rows())
    _write("style_longprompt", build_style_rows())


if __name__ == "__main__":
    main()
