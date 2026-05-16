"""VLM-authored edit proposals: prompt + validation + task-row builder.

Used by ``data/datasets/sun397/extract.py`` to author add / remove tasks
for images whose underlying datasets don't ship matching instructions.
One Anthropic call per image returns short object names; this module
translates that response into TaskDefinition-shaped row dicts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import anthropic

from data.tasks._seed import task_seed
from experiments.common.tasks import TaskDefinition
from utils.vlm import DEFAULT_MODEL, call_vision, image_block, text_block

SYSTEM_PROMPT = """\
You design test cases for an image-editing experiment. Given ONE image,
propose two short object names:

1. ADD: a single object NOT currently in the image but plausibly fits
   the scene. If no scene-agnostic addition fits, set "add_object": null.
2. REMOVE: a single object IS visible and could be plausibly removed
   (not the entire subject; the scene would still read coherently
   without it). If nothing meets that bar, set "remove_object": null.

SCENE-AGNOSTIC RULE: the proposed object names must NOT reveal the
specific scene/location depicted. For a volcano photo, "lava plume" or
"volcanic crater" is forbidden -- those give away the scene. Prefer
generic objects that could plausibly fit many different scenes. If no
scene-agnostic object fits a given field, set it to null.

FORBIDDEN OBJECTS: do NOT propose any of these (or any phrase
containing one of these words as the head noun): bird, backpack,
person, bicycle, bench, bottle, plant, dog. They are over-represented
in our existing dataset. Pick a different scene-agnostic object, or
set the field to null if nothing else fits.

Return ONLY a single JSON object on one line, no markdown:
{"add_object": "<noun>" | null,
 "remove_object": "<noun>" | null}

Rules:
- Object phrases: 1-4 words, lowercase, no punctuation, singular.
- All non-null object names must be pairwise distinct.
- No proper nouns, no named people.
- All non-null object names must obey the SCENE-AGNOSTIC RULE.
- All non-null object names must NOT be in the FORBIDDEN OBJECTS list.\
"""

# 1-4 lowercase ASCII words separated by single spaces; no digits, no
# punctuation. Hyphens are accepted within a word (e.g. "stop-sign") but
# not as standalone tokens.
_WORD_RE = re.compile(r"^[a-z]+(?:-[a-z]+)?(?: [a-z]+(?:-[a-z]+)?){0,3}$")


def _validate_proposal(p: object) -> str | None:
    """Return error string on failure, ``None`` on success."""
    if not isinstance(p, dict):
        return f"top-level not a dict: {type(p).__name__}"
    expected_keys = {"add_object", "remove_object"}
    if set(p.keys()) != expected_keys:
        return f"unexpected keys: {sorted(p.keys())}"
    add_obj = p["add_object"]
    rem_obj = p["remove_object"]
    if add_obj is not None:
        if not isinstance(add_obj, str) or not _WORD_RE.match(add_obj):
            return f"add_object invalid: {add_obj!r}"
    if rem_obj is not None:
        if not isinstance(rem_obj, str) or not _WORD_RE.match(rem_obj):
            return f"remove_object invalid: {rem_obj!r}"
    if add_obj is not None and rem_obj is not None and add_obj == rem_obj:
        return f"object names not pairwise distinct: add={add_obj!r}, remove={rem_obj!r}"
    if add_obj is None and rem_obj is None:
        return "all of add_object/remove_object are null -- no usable task"
    return None


async def propose_edit_objects(
    client: anthropic.AsyncAnthropic,
    image_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    max_attempts: int = 3,
) -> tuple[dict | None, str, int, int]:
    """Returns ``(proposal, error, total_in_tokens, total_out_tokens)``.

    On JSON-parse failure or invariant violation, retries up to
    ``max_attempts`` total attempts. Token counts accumulate across attempts.
    Returns ``(None, last_error, ...)`` if every attempt fails.
    """
    user_blocks = [
        text_block("Image:"),
        image_block(image_path),
        text_block("Return JSON now."),
    ]
    total_in, total_out = 0, 0
    last_err = ""
    for attempt in range(1, max_attempts + 1):
        text, err, in_tok, out_tok = await call_vision(
            client,
            system=SYSTEM_PROMPT,
            user_blocks=user_blocks,
            model=model,
            max_tokens=200,
        )
        total_in += in_tok
        total_out += out_tok
        if text is None:
            last_err = f"TRANSPORT (attempt {attempt}): {err}"
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            last_err = f"PARSE (attempt {attempt}): {e}: {text!r}"
            continue
        validation = _validate_proposal(parsed)
        if validation is None:
            return parsed, "", total_in, total_out
        last_err = f"INVARIANT (attempt {attempt}): {validation}: got {text!r}"
    return None, last_err, total_in, total_out


def _slug(s: str) -> str:
    return s.replace(" ", "_").replace("-", "_").replace("/", "-")


def _article(noun: str) -> str:
    return "an" if noun[0] in "aeiou" else "a"


def proposal_to_task_rows(
    *,
    proposal: dict,
    task_id_stem: str,
    source: str,
    rel_image_paths: dict[str, str],
    image_size: tuple[int, int],
    metadata_extra: dict,
) -> list[dict]:
    """Build 0-2 TaskDefinition-shaped row dicts from a validated proposal.

    ``task_id_stem`` is everything after the ``add_/remove_`` prefix and
    before the per-task object slug — e.g. ``sun397_a_abbey__sun_aaaa``.

    ``rel_image_paths`` is keyed by edit_type (``"add"``/``"remove"``) and
    maps to the in-bucket image path (because each bucket gets its own copy
    of the cropped JPG).

    Either add_object or remove_object in ``proposal`` may be null (None) —
    those rows are simply not emitted. Validation guarantees not-all-null.
    """
    add_obj = proposal["add_object"]
    rem_obj = proposal["remove_object"]
    w, h = image_size
    rows: list[dict] = []

    if add_obj is not None:
        add_id = f"add_{task_id_stem}_{_slug(add_obj)}"
        add_row = {
            "task_id": add_id,
            "edit_type": "add",
            "source": source,
            "instruction": f"add {_article(add_obj)} {add_obj}",
            "source_image_path": rel_image_paths["add"],
            "source_caption": None,
            "ref_seed": None,
            "noise_seed": task_seed(add_id),
            "real_ref_name": None,
            "height": h,
            "width": w,
            "metadata": {**metadata_extra, "add_object": add_obj},
        }
        TaskDefinition(
            **{k: v for k, v in add_row.items() if k != "metadata"},
            metadata=add_row["metadata"],
        )
        rows.append(add_row)

    if rem_obj is not None:
        rem_id = f"remove_{task_id_stem}_{_slug(rem_obj)}"
        rem_row = {
            "task_id": rem_id,
            "edit_type": "remove",
            "source": source,
            "instruction": f"remove the {rem_obj}",
            "source_image_path": rel_image_paths["remove"],
            "source_caption": None,
            "ref_seed": None,
            "noise_seed": task_seed(rem_id),
            "real_ref_name": None,
            "height": h,
            "width": w,
            "metadata": {**metadata_extra, "remove_object": rem_obj},
        }
        TaskDefinition(
            **{k: v for k, v in rem_row.items() if k != "metadata"},
            metadata=rem_row["metadata"],
        )
        rows.append(rem_row)

    return rows
