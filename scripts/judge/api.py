"""Vision-judge call: builds labeled-image parts, parses JSON.

Provider-neutral: composes :class:`utils.vlm.TextPart` /
:class:`utils.vlm.ImagePart` lists and sends them through any client
implementing the ``call_vision(system=, parts=, max_tokens=)`` contract
(see :func:`utils.vlm.make_vlm`). This module just composes the
multi-image labeled content and parses the JSON response.
"""

from __future__ import annotations

import json
from pathlib import Path

from utils.vlm import ImagePart, Part, TextPart

__all__ = [
    "SYSTEM_PROMPT",
    "call_judge",
]

SYSTEM_PROMPT = (
    "You are a strict visual judge for an image-editing interpretability "
    "experiment. You will be shown several labeled images and asked one "
    "yes/no question about whether a stated prediction is satisfied.\n"
    "Respond ONLY with a single JSON object on one line, no markdown, "
    "no preamble:\n"
    '{"pass": 0 or 1, "reason": "<one short sentence, <=25 words>"}\n'
    'The "pass" field is 1 if the prediction is satisfied, 0 otherwise. '
    "If the question cannot be answered from the images, reply with "
    '{"pass": 0, "reason": "cannot determine"}.'
)


def _content(
    image_labels: list[str],
    image_paths: list[Path],
    question: str,
    *,
    cache_prefix_len: int | None = None,
) -> list[Part]:
    parts: list[Part] = []
    # Why: cache marks a prefix breakpoint; placing it on the LAST image in
    # the shared prefix means siblings whose first cache_prefix_len
    # (label, image) pairs are identical will hit the ephemeral cache.
    # OpenAI ignores the flag (automatic prefix caching).
    cache_at = (cache_prefix_len - 1) if cache_prefix_len else -1
    for i, (label, path) in enumerate(zip(image_labels, image_paths)):
        parts.append(TextPart(label))
        parts.append(ImagePart(path, cache=(i == cache_at)))
    parts.append(TextPart(question))
    return parts


async def call_judge(
    vlm,
    image_labels: list[str],
    image_paths: list[Path],
    question: str,
    *,
    cache_prefix_len: int | None = None,
    cache_key: str | None = None,
) -> tuple[int | None, str, int, int]:
    """Returns ``(pass, reason, in_tokens, out_tokens)``.

    ``vlm`` is any provider client from :func:`utils.vlm.make_vlm` (duck
    typed: ``.model`` plus async ``call_vision``). ``pass`` is ``None`` on
    transport or parse error; ``reason`` carries the error message in that
    case.

    If ``cache_prefix_len`` is set, the last image in the first
    ``cache_prefix_len`` (label, image) pairs is marked as an ephemeral
    cache breakpoint so back-to-back calls sharing that prefix can hit
    the provider's prompt cache. ``cache_key`` names the shared prefix
    (one value per cache bucket); OpenAI uses it as ``prompt_cache_key``
    to keep sibling calls on the same cache shard, Anthropic ignores it.
    """
    text, err, in_tok, out_tok = await vlm.call_vision(
        system=SYSTEM_PROMPT,
        parts=_content(
            image_labels, image_paths, question,
            cache_prefix_len=cache_prefix_len,
        ),
        max_tokens=200,
        cache_key=cache_key,
    )
    if text is None:
        return None, err, in_tok, out_tok
    try:
        parsed = json.loads(text)
        return int(parsed["pass"]), str(parsed.get("reason", ""))[:200], in_tok, out_tok
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return None, f"PARSE_ERROR: {e}: {text!r}", in_tok, out_tok
