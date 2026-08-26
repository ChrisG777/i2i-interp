"""Generic async VLM vision client, provider-agnostic at the call layer.

Single shared module for VLM-as-Judge and VLM-as-task-author callsites.
Callers compose provider-neutral :class:`TextPart` / :class:`ImagePart`
lists and go through a client object (see :func:`make_vlm`) exposing
``call_vision(system=, parts=, max_tokens=)``. The Anthropic implementation
lives here (:class:`AnthropicVlm` over the legacy block-builder functions);
the OpenAI Responses-API implementation is in :mod:`utils.vlm_openai`.

Pricing constants are Claude Opus 4.7 list prices (USD per 1M tokens).
Update here in one place if pricing changes.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import anthropic

PRICE_IN_PER_MTOK = 15.0
PRICE_OUT_PER_MTOK = 75.0
DEFAULT_MODEL = "claude-opus-4-7"

VERBOSE_CACHE = False

_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

# Anthropic's vision endpoint rejects any single image whose base64 payload
# exceeds 5 MiB. We re-encode oversized inputs in-memory (JPEG q=92, halving
# dimensions until under the cap) and cache the result by path so sibling
# judge calls don't re-encode the same file.
_BASE64_LIMIT_BYTES = 5 * 1024 * 1024
_RAW_LIMIT_BYTES = (_BASE64_LIMIT_BYTES // 4) * 3  # base64 inflates 4/3x
_encoded_cache: dict[tuple[str, int], tuple[str, str]] = {}


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    path: Path
    # Anthropic ephemeral cache breakpoint; ignored by OpenAI, whose prompt
    # caching is automatic and prefix-based.
    cache: bool = False


Part = TextPart | ImagePart


def make_client() -> anthropic.AsyncAnthropic:
    """Construct an async Anthropic client using ``ANTHROPIC_API_KEY`` from env."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    assert api_key, (
        "ANTHROPIC_API_KEY not set. Add it to your shell env or a .env file."
    )
    return anthropic.AsyncAnthropic(api_key=api_key)


def text_block(text: str, *, cache: bool = False) -> dict:
    block: dict = {"type": "text", "text": text}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def _encode_image(path: Path) -> tuple[str, str]:
    """Return ``(media_type, base64_data)`` for ``path``, shrinking if needed.

    Only the shrink path is memoized (keyed by path + mtime). Reading and
    base64-encoding a small file is fast; PIL resize + JPEG encode is not.
    """
    raw = path.read_bytes()
    media = _MEDIA_TYPES[path.suffix.lower()]
    if len(raw) <= _RAW_LIMIT_BYTES:
        return media, base64.standard_b64encode(raw).decode("ascii")

    key = (str(path), path.stat().st_mtime_ns)
    if key in _encoded_cache:
        return _encoded_cache[key]

    from PIL import Image  # local import; pillow is a project dep

    img = Image.open(io.BytesIO(raw))
    img.load()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    scale = 1.0
    while True:
        if scale < 1.0:
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            candidate = img.resize(new_size, Image.LANCZOS)
        else:
            candidate = img
        buf = io.BytesIO()
        candidate.save(buf, format="JPEG", quality=92, optimize=True)
        data = buf.getvalue()
        if len(data) <= _RAW_LIMIT_BYTES or scale < 0.1:
            break
        scale *= 0.7
    print(
        f"[vlm] shrunk {path.name}: {len(raw)} -> {len(data)} bytes "
        f"(scale={scale:.2f}, format=JPEG)",
        file=sys.stderr,
    )
    result = ("image/jpeg", base64.standard_b64encode(data).decode("ascii"))
    _encoded_cache[key] = result
    return result


def image_block(path: Path, *, cache: bool = False) -> dict:
    media, data = _encode_image(path)
    block: dict = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media,
            "data": data,
        },
    }
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return block


# Why retry: without this, a single 429/529 from the provider stamps an
# unrecoverable "ERROR: ..." into the row's `pass` field. The runner's CSV is
# append-only so later runs do shadow the bad row, but the raw file ends up
# full of stale error rows that read as if the grading is incomplete.
_MAX_RETRIES = 6
_BACKOFF_CAP_S = 60.0


async def call_with_retries(
    do_call: Callable[[], Awaitable[tuple[str, str, int, int]]],
    *,
    rate_limit_type: type[Exception],
    status_error_type: type[Exception],
    api_error_type: type[Exception],
    retryable_statuses: tuple[int, ...],
) -> tuple[str | None, str, int, int]:
    """Retry driver shared by the Anthropic and OpenAI clients.

    ``do_call`` performs one API call and returns ``(text, "", in_tok,
    out_tok)``, raising the provider SDK's exceptions on failure. Both SDKs
    are stainless-generated, so the ``status_code`` / ``response.headers``
    shapes match and only the exception types differ per provider.

    Retries rate-limit errors and ``retryable_statuses`` up to
    ``_MAX_RETRIES`` times with capped exponential backoff + jitter,
    honoring the ``retry-after`` header when present. Other API errors
    return ``(None, "ERROR: ...", 0, 0)``.
    """
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return await do_call()
        except (rate_limit_type, status_error_type) as e:
            status = getattr(e, "status_code", None)
            if not isinstance(e, rate_limit_type) and status not in retryable_statuses:
                return None, f"ERROR: {e}", 0, 0
            last_err = e
            retry_after = _retry_after_seconds(e)
            sleep_s = retry_after if retry_after is not None else min(
                _BACKOFF_CAP_S, 2.0**attempt + random.uniform(0.0, 1.0)
            )
            await asyncio.sleep(sleep_s)
        except api_error_type as e:
            return None, f"ERROR: {e}", 0, 0
    return None, f"ERROR: rate-limited after {_MAX_RETRIES} retries: {last_err}", 0, 0


async def call_vision(
    client: anthropic.AsyncAnthropic,
    *,
    system: str,
    user_blocks: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 200,
) -> tuple[str | None, str, int, int]:
    """Single async Anthropic vision call.

    Returns ``(text_response, error, in_tokens, out_tokens)``. On transport
    error, ``text_response`` is ``None`` and ``error`` carries the message.
    """

    async def do_call() -> tuple[str, str, int, int]:
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            # Judging is a mechanical binary check; some models (Sonnet 5)
            # run adaptive thinking by default, which bills extra output
            # tokens and can eat the max_tokens budget before the verdict.
            thinking={"type": "disabled"},
            system=[text_block(system, cache=True)],
            messages=[{"role": "user", "content": user_blocks}],
        )
        in_tok = getattr(resp.usage, "input_tokens", 0) or 0
        out_tok = getattr(resp.usage, "output_tokens", 0) or 0
        if VERBOSE_CACHE:
            cr = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
            cw = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
            print(
                f"[cache] in={in_tok} read={cr} write={cw} out={out_tok}",
                file=sys.stderr,
            )
        # First text-type block: models with thinking enabled may lead with
        # a ThinkingBlock, which has no .text.
        text = next(
            (b.text for b in resp.content if getattr(b, "type", "") == "text"), ""
        ).strip()
        return text, "", in_tok, out_tok

    return await call_with_retries(
        do_call,
        rate_limit_type=anthropic.RateLimitError,
        status_error_type=anthropic.APIStatusError,
        api_error_type=anthropic.APIError,
        retryable_statuses=(429, 529),
    )


def _retry_after_seconds(err: Exception) -> float | None:
    """Pull a ``retry-after`` value (seconds) off an SDK APIStatusError."""
    resp = getattr(err, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def render_anthropic_blocks(parts: list[Part]) -> list[dict]:
    """Render provider-neutral parts into Anthropic content-block dicts."""
    blocks: list[dict] = []
    for p in parts:
        if isinstance(p, TextPart):
            blocks.append(text_block(p.text))
        else:
            assert isinstance(p, ImagePart), p
            blocks.append(image_block(p.path, cache=p.cache))
    return blocks


class AnthropicVlm:
    """Anthropic implementation of the provider-neutral vision contract:
    ``.model`` plus ``call_vision(system=, parts=, max_tokens=)``."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client = make_client()

    async def call_vision(
        self, *, system: str, parts: list[Part], max_tokens: int = 200,
        cache_key: str | None = None,
    ) -> tuple[str | None, str, int, int]:
        # cache_key is an OpenAI routing hint; Anthropic's cache is keyed on
        # content breakpoints (ImagePart.cache), so it's ignored here.
        return await call_vision(
            self._client,
            system=system,
            user_blocks=render_anthropic_blocks(parts),
            model=self.model,
            max_tokens=max_tokens,
        )


@dataclass(frozen=True)
class ModelSpec:
    provider: str  # "anthropic" | "openai"
    price_in_per_mtok: float
    price_out_per_mtok: float


MODELS: dict[str, ModelSpec] = {
    "claude-opus-4-7": ModelSpec("anthropic", PRICE_IN_PER_MTOK, PRICE_OUT_PER_MTOK),
    # Intro pricing through 2026-08-31; reverts to the $3/$15 list price after.
    "claude-sonnet-5": ModelSpec("anthropic", 2.0, 10.0),
    # TODO: fill in Sol/Luna list prices when needed; no callsite reads
    # pricing today, so 0.0 is a safe placeholder.
    "gpt-5.6": ModelSpec("openai", 0.0, 0.0),
    "gpt-5.6-terra": ModelSpec("openai", 2.50, 15.0),  # 2026-07 list prices
    "gpt-5.6-luna": ModelSpec("openai", 0.0, 0.0),
}


def make_vlm(model: str = DEFAULT_MODEL):
    """Construct the provider client for ``model`` (see ``MODELS``)."""
    assert model in MODELS, f"Unknown VLM model {model!r}. Known: {sorted(MODELS)}"
    if MODELS[model].provider == "anthropic":
        return AnthropicVlm(model=model)
    from utils.vlm_openai import OpenAIVlm  # lazy: import openai only when used

    return OpenAIVlm(model=model)
