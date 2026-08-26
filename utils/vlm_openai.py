"""Async OpenAI Responses-API vision client.

Mirrors the provider-neutral contract of :mod:`utils.vlm` (``.model`` plus
``call_vision(system=, parts=, max_tokens=)``); construct via
:func:`utils.vlm.make_vlm`. Reuses the shared image encoder and retry
driver so both providers behave identically at the orchestration layer.
"""

from __future__ import annotations

import hashlib
import os

import openai

from utils.vlm import (
    ImagePart,
    Part,
    TextPart,
    _encode_image,
    call_with_retries,
)

# Judges emit a one-line JSON verdict; no deep reasoning needed.
REASONING_EFFORT = "low"

# Reasoning tokens bill against max_output_tokens on gpt-5.6; a 200-token cap
# can be consumed entirely by reasoning, yielding an empty output_text. Floor
# the cap so the visible reply always has room.
_MIN_OUTPUT_TOKENS = 2048

# OpenAI rejects a prompt_cache_key longer than this with a 400.
_MAX_CACHE_KEY = 64


def _fold_cache_key(key: str) -> str:
    """Fit ``key`` into OpenAI's 64-char prompt_cache_key limit.

    Judge cache keys are ``<judge>:<entity_id>``, and entity ids for the style
    and pair judges routinely exceed 64 chars, which is a hard 400 rather than
    a silent truncation. Oversized keys keep a readable prefix and get a
    sha256 tail, so distinct ids stay on distinct shards and equal ids still
    collide (which is the whole point -- siblings must share a shard to hit
    the prefix cache).

    sha256 rather than ``hash()``: the run is resumable across processes, and
    PYTHONHASHSEED randomization would hand the same entity a different key on
    every restart, quietly scattering siblings across shards.
    """
    if len(key) <= _MAX_CACHE_KEY:
        return key
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"{key[:_MAX_CACHE_KEY - len(digest) - 1]}-{digest}"


def render_openai_content(parts: list[Part]) -> list[dict]:
    """Render provider-neutral parts into Responses-API content dicts.

    ``ImagePart.cache`` is intentionally ignored: OpenAI prompt caching is
    automatic and prefix-based, with no explicit breakpoints.
    """
    content: list[dict] = []
    for p in parts:
        if isinstance(p, TextPart):
            content.append({"type": "input_text", "text": p.text})
        else:
            assert isinstance(p, ImagePart), p
            media, b64 = _encode_image(p.path)
            content.append(
                {"type": "input_image", "image_url": f"data:{media};base64,{b64}"}
            )
    return content


class OpenAIVlm:
    def __init__(
        self,
        model: str = "gpt-5.6-terra",
        *,
        reasoning_effort: str = REASONING_EFFORT,
    ):
        api_key = os.environ.get("OPENAI_API_KEY")
        assert api_key, (
            "OPENAI_API_KEY not set. Add it to your shell env or a .env file."
        )
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def call_vision(
        self, *, system: str, parts: list[Part], max_tokens: int = 200,
        cache_key: str | None = None,
    ) -> tuple[str | None, str, int, int]:
        content = render_openai_content(parts)
        # prompt_cache_key steers cache-shard routing: calls sharing a prefix
        # (sibling judges on one entity) should carry the same key so they
        # land on the same shard and the automatic prefix cache actually hits.
        extra = (
            {} if cache_key is None
            else {"prompt_cache_key": _fold_cache_key(cache_key)}
        )

        async def do_call() -> tuple[str, str, int, int]:
            resp = await self._client.responses.create(
                model=self.model,
                instructions=system,
                input=[{"role": "user", "content": content}],
                max_output_tokens=max(max_tokens, _MIN_OUTPUT_TOKENS),
                reasoning={"effort": self.reasoning_effort},
                **extra,
            )
            # An "incomplete" response (e.g. reasoning ate the token budget)
            # yields empty output_text; the judge's JSON parse then records a
            # PARSE_ERROR row with empty `pass`, which reruns retry.
            text = (resp.output_text or "").strip()
            in_tok = getattr(resp.usage, "input_tokens", 0) or 0
            out_tok = getattr(resp.usage, "output_tokens", 0) or 0
            return text, "", in_tok, out_tok

        return await call_with_retries(
            do_call,
            rate_limit_type=openai.RateLimitError,
            status_error_type=openai.APIStatusError,
            api_error_type=openai.APIError,
            retryable_statuses=(429, 500, 503),
        )
