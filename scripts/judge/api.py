"""Vision-judge call: builds labeled-image content blocks, parses JSON.

Reuses the transport layer in :mod:`utils.vlm` (async client, system-prompt
caching, base64 image encoding). This module just composes the multi-image
labeled content blocks and parses the JSON response.
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic

from utils.vlm import (
    DEFAULT_MODEL as MODEL_ID,
    PRICE_IN_PER_MTOK,
    PRICE_OUT_PER_MTOK,
    call_vision,
    image_block,
    text_block,
)

__all__ = [
    "MODEL_ID",
    "PRICE_IN_PER_MTOK",
    "PRICE_OUT_PER_MTOK",
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
) -> list[dict]:
    blocks: list[dict] = []
    n = len(image_labels)
    # Why: cache_control marks a prefix breakpoint; placing it on the LAST
    # image in the shared prefix means siblings whose first cache_prefix_len
    # (label, image) pairs are identical will hit the ephemeral cache.
    cache_at = (cache_prefix_len - 1) if cache_prefix_len else -1
    for i, (label, path) in enumerate(zip(image_labels, image_paths)):
        blocks.append(text_block(label))
        blocks.append(image_block(path, cache=(i == cache_at)))
    blocks.append(text_block(question))
    return blocks


async def call_judge(
    client: anthropic.AsyncAnthropic,
    image_labels: list[str],
    image_paths: list[Path],
    question: str,
    *,
    cache_prefix_len: int | None = None,
) -> tuple[int | None, str, int, int]:
    """Returns ``(pass, reason, in_tokens, out_tokens)``.

    ``pass`` is ``None`` on transport or parse error; ``reason`` carries
    the error message in that case.

    If ``cache_prefix_len`` is set, the last image in the first
    ``cache_prefix_len`` (label, image) pairs is marked with
    ``cache_control: ephemeral`` so back-to-back calls sharing that prefix
    can hit Anthropic's prompt cache.
    """
    text, err, in_tok, out_tok = await call_vision(
        client,
        system=SYSTEM_PROMPT,
        user_blocks=_content(
            image_labels, image_paths, question,
            cache_prefix_len=cache_prefix_len,
        ),
        model=MODEL_ID,
        max_tokens=200,
    )
    if text is None:
        return None, err, in_tok, out_tok
    try:
        parsed = json.loads(text)
        return int(parsed["pass"]), str(parsed.get("reason", ""))[:200], in_tok, out_tok
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return None, f"PARSE_ERROR: {e}: {text!r}", in_tok, out_tok
