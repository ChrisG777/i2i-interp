"""Additive attention masks for category-level knockouts.

A knockout setting is a ``(sources, destination)`` pair: it blocks
information flow from any region in ``sources`` to ``destination``.
Regions are token bands within the joint stream — full categories
(``"text"``, ``"image"``, ``"ref"``) or text-token subsets
(``"text[content]"``, ``"text[padding]"``). Category slices come from
``utils.flux2_klein.get_category_slices``; content/padding subsets
within text are resolved against a 1D bool mask passed by the caller.

In the attention view, information flows from keys to queries: query
position ``q`` reads value vectors ``v_k`` weighted by the softmax of
``q·k``. So "block info from A to B" translates to "queries in B cannot
attend to keys in A". We implement that by setting the attention mask
to ``-inf`` at rows ``destination`` (queries) × columns ``sources``
(keys), and ``0`` elsewhere.

The mask is additive; SDPA applies it before softmax, so ``0`` entries
are numerically a no-op and ``-inf`` entries are killed pre-softmax
inside SDPA — we never implement attention ourselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch

from utils.flux2_klein import TokenLayout, get_category_slices

__all__ = [
    "Category",
    "Region",
    "VALID_REGIONS",
    "KnockoutSetting",
    "KNOCKOUT_SETTINGS",
    "COMPOSITE_KNOCKOUT_SETTINGS",
    "LayerMode",
    "LAYER_MODES",
    "build_knockout_mask",
    "build_combined_knockout_mask",
    "combine_masks",
    "masked_indices",
    "apply_mask_to_layers",
    "apply_split_mask_to_layers",
    "clear_all_masks",
    "resolve_settings",
]


Category = Literal["text", "image", "ref"]
_VALID_CATEGORIES: tuple[Category, ...] = ("text", "image", "ref")

# Region strings name a band of tokens within the joint stream. A bare
# category (``"text"|"image"|"ref"``) means the full category. ``"text[content]"``
# and ``"text[padding]"`` mean the text-content / text-padding subsets — the
# caller resolves which positions are content via the Qwen3 attention mask
# and passes a single ``text_content_mask`` bool tensor to the builder.
Region = Literal["text", "image", "ref", "text[content]", "text[padding]"]
VALID_REGIONS: tuple[Region, ...] = (
    "text", "image", "ref", "text[content]", "text[padding]",
)
_TEXT_SUBSET_REGIONS: frozenset[str] = frozenset({"text[content]", "text[padding]"})

LayerMode = Literal["suffix", "prefix", "individual", "window"]
LAYER_MODES: tuple[LayerMode, ...] = ("suffix", "prefix", "individual", "window")


def _parse_region(region: Region) -> tuple[Category, Literal["content", "padding"] | None]:
    """Return ``(base_category, subset_kind)`` for a region string.

    ``subset_kind`` is ``None`` for full categories, ``"content"`` /
    ``"padding"`` for text subsets. Fails fast on unknown region strings.
    """
    assert region in VALID_REGIONS, (
        f"Unknown region {region!r}. Valid: {sorted(VALID_REGIONS)}"
    )
    if region == "text[content]":
        return ("text", "content")
    if region == "text[padding]":
        return ("text", "padding")
    return (region, None)  # type: ignore[return-value]


@dataclass(frozen=True)
class KnockoutSetting:
    """Block information flow from every ``sources`` region to ``destination``.

    Under the hood, this means: queries in ``destination`` cannot
    attend to keys in any of the ``sources`` regions. Each region is a
    full category (``"text"|"image"|"ref"``) or a text subset
    (``"text[content]"|"text[padding]"``).

    Same-base-category source/destination pairs are rejected — masking a
    category against itself (or its content/padding subset against the
    other text subset) crosses into territory where validating
    softmax-row safety becomes per-instance, and no current production
    setting needs it. Cross-category pairs (``"text[padding]"->image``,
    ``"ref"->text[padding]"``) are the supported shape.
    """

    sources: tuple[Region, ...]
    destination: Region

    def __post_init__(self) -> None:
        assert len(self.sources) > 0, "Need at least one source region"
        assert len(set(self.sources)) == len(self.sources), (
            f"Duplicate sources: {self.sources}"
        )
        dest_cat, _ = _parse_region(self.destination)
        for source in self.sources:
            src_cat, _ = _parse_region(source)
            assert src_cat != dest_cat, (
                f"Cannot mask {source}->{self.destination}: source and "
                f"destination share base category {src_cat!r}"
            )

    @property
    def name(self) -> str:
        """Short info-flow label, e.g. ``"image+text->ref"`` or ``"text[padding]+ref->image"``."""
        return f"{'+'.join(self.sources)}->{self.destination}"


KNOCKOUT_SETTINGS: list[KnockoutSetting] = [
    # Cross-modal pair knockouts — block info flow source -> destination.
    KnockoutSetting(("text",), "ref"),
    KnockoutSetting(("image",), "ref"),
    KnockoutSetting(("ref",), "text"),
    KnockoutSetting(("image",), "text"),
    KnockoutSetting(("ref",), "image"),
    KnockoutSetting(("text",), "image"),
    # Group knockouts.
    KnockoutSetting(("image", "text"), "ref"),
    KnockoutSetting(("image", "ref"), "text"),
]


COMPOSITE_KNOCKOUT_SETTINGS: dict[str, tuple[KnockoutSetting, ...]] = {
    # Bidirectional ref<->image: sever both directions of cross-stream
    # attention between the reference-image tokens and the generated-image
    # tokens, while leaving text<->image, text<->ref, and all self-attention
    # blocks (text->text, image->image, ref->ref) untouched.
    "image<->ref": (
        KnockoutSetting(("image",), "ref"),
        KnockoutSetting(("ref",), "image"),
    ),
}


def _region_indexer(
    region: Region,
    layout: TokenLayout,
    slices: dict[str, slice],
    *,
    text_content_mask: torch.Tensor | None,
    device: torch.device | str,
) -> slice | torch.Tensor:
    """Return the joint-stream indexer for a region.

    Full categories return a ``slice`` so advanced-indexing stays
    rectangular against another indexer (avoiding the 1D collapse you'd
    get if both rows and cols were 1D long tensors). Text subsets return
    a 1D long tensor on ``device`` whose entries are sorted.
    """
    base, subset = _parse_region(region)
    if base == "ref":
        assert layout.has_ref, (
            f"Region {region!r} references the 'ref' category but layout "
            f"has no ref tokens (ref_seq_len=0)"
        )
    if subset is None:
        return slices[base]
    # Text subset.
    assert base == "text", (
        f"Internal error: only text supports subsets, got base={base!r}"
    )
    assert text_content_mask is not None, (
        f"Region {region!r} requires text_content_mask, got None"
    )
    assert text_content_mask.shape == (layout.text_seq_len,), (
        f"text_content_mask shape {tuple(text_content_mask.shape)} "
        f"!= ({layout.text_seq_len},)"
    )
    assert text_content_mask.dtype == torch.bool, (
        f"text_content_mask dtype {text_content_mask.dtype} != torch.bool"
    )
    selector = text_content_mask if subset == "content" else ~text_content_mask
    assert selector.any(), (
        f"Region {region!r} resolved to an empty index set "
        f"(text_content_mask has no {'True' if subset == 'content' else 'False'} entries)"
    )
    return selector.nonzero(as_tuple=True)[0].to(device) + slices["text"].start


def build_knockout_mask(
    layout: TokenLayout,
    setting: KnockoutSetting,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    text_content_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a ``(1, 1, S, S)`` additive mask for a single knockout setting.

    Queries in ``setting.destination`` cannot attend to keys in any of
    ``setting.sources``: rows × cols at each ``(destination, source)``
    region pair are ``-inf``, everything else is ``0``. Construction uses
    float32 for numerical clarity, then casts to ``dtype`` (typically bf16)
    to match the Q/K dtype expected by SDPA on GPU.

    ``text_content_mask`` is a 1D bool tensor of length ``text_seq_len``
    where ``True`` marks content tokens and ``False`` marks padding. Required
    iff any region in ``setting`` is ``"text[content]"`` or ``"text[padding]"``.
    """
    slices = get_category_slices(layout)
    total = layout.total
    dest_indexer = _region_indexer(
        setting.destination, layout, slices,
        text_content_mask=text_content_mask, device=device,
    )
    mask = torch.zeros(
        (1, 1, total, total),
        device=device,
        dtype=torch.float32,
    )
    # One write per source. If both dest_indexer and src_indexer are 1D
    # long tensors, advanced indexing would silently collapse to a 1D
    # output instead of writing a rectangle. The same-base-category ban
    # in KnockoutSetting prevents that pairing today (text-subset dest
    # never co-occurs with a text-subset source); assert anyway to fail
    # fast if a future region kind breaks the invariant.
    dest_is_tensor = isinstance(dest_indexer, torch.Tensor)
    for source in setting.sources:
        src_indexer = _region_indexer(
            source, layout, slices,
            text_content_mask=text_content_mask, device=device,
        )
        assert not (dest_is_tensor and isinstance(src_indexer, torch.Tensor)), (
            f"Both destination {setting.destination!r} and source "
            f"{source!r} are 1D index tensors — advanced indexing would "
            f"collapse to 1D. Pair a subset region with a full-category "
            f"region on the other side."
        )
        mask[:, :, dest_indexer, src_indexer] = float("-inf")

    # Softmax over a fully-masked query row produces NaN. KnockoutSetting
    # rejects same-base-category source/destination pairs, so an atomic
    # setting can never fully mask a query row (the destination's own
    # self-attention is always preserved); assert to fail fast if that
    # invariant ever breaks.
    finite_any = (mask[0, 0] > float("-inf")).any(dim=-1)
    assert finite_any.all(), (
        f"Setting {setting.name!r} masks an entire query row — softmax would NaN"
    )
    return mask.to(dtype)


def combine_masks(*masks: torch.Tensor) -> torch.Tensor:
    """OR-union additive attention masks.

    For masks containing only ``{0, -inf}``, the elementwise minimum is the
    OR-union — a position is ``-inf`` iff at least one input mask blocks it.
    All masks must share shape, dtype, and device. Runs the same softmax-row
    safety check as the atomic builders.
    """
    assert len(masks) >= 1, "combine_masks needs at least one mask"
    ref = masks[0]
    for m in masks[1:]:
        assert m.shape == ref.shape, (
            f"combine_masks: shape mismatch {tuple(m.shape)} vs {tuple(ref.shape)}"
        )
        assert m.dtype == ref.dtype, (
            f"combine_masks: dtype mismatch {m.dtype} vs {ref.dtype}"
        )
        assert m.device == ref.device, (
            f"combine_masks: device mismatch {m.device} vs {ref.device}"
        )
    out = ref
    for m in masks[1:]:
        out = torch.minimum(out, m)
    finite_any = (out[0, 0] > float("-inf")).any(dim=-1)
    assert finite_any.all(), (
        "Combined mask fully masks a query row — softmax would NaN"
    )
    return out


def build_combined_knockout_mask(
    layout: TokenLayout,
    settings: Iterable[KnockoutSetting],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    text_content_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """OR-union of multiple atomic settings via :func:`combine_masks`.

    A position is ``-inf`` iff at least one setting marks it; otherwise ``0``.
    """
    settings = tuple(settings)
    assert len(settings) > 0, "Need at least one setting to combine"
    atomic = [
        build_knockout_mask(
            layout, s, device=device, dtype=dtype,
            text_content_mask=text_content_mask,
        )
        for s in settings
    ]
    return combine_masks(*atomic)


def masked_indices(
    mode: LayerMode,
    L: int,
    num_blocks: int,
    *,
    window_size: int | None = None,
) -> set[int]:
    """Return the set of block indices that should be masked for ``(mode, L)``.

    - ``suffix``: blocks ``[L..num_blocks)``. ``L=0`` masks all; ``L=num_blocks``
      masks none.
    - ``prefix``: blocks ``[0..L]`` inclusive. ``L=0`` masks only block 0;
      ``L=num_blocks-1`` masks all.
    - ``individual``: only block ``L``.
    - ``window``: blocks ``[L..L+window_size)``. Requires ``window_size`` and
      ``0 <= L <= num_blocks - window_size`` so every call masks exactly
      ``window_size`` blocks.
    """
    assert 0 <= L <= num_blocks, f"L={L} out of range for num_blocks={num_blocks}"
    if mode == "window":
        assert window_size is not None, "window mode requires window_size"
        assert 1 <= window_size <= num_blocks, (
            f"window_size={window_size} out of range for num_blocks={num_blocks}"
        )
        assert 0 <= L <= num_blocks - window_size, (
            f"window mode requires 0 <= L <= num_blocks - window_size "
            f"({num_blocks - window_size}), got L={L}"
        )
        return set(range(L, L + window_size))
    assert window_size is None, (
        f"window_size is only valid for mode='window', got mode={mode!r}"
    )
    if mode == "suffix":
        return set(range(L, num_blocks))
    if mode == "prefix":
        assert L < num_blocks, f"prefix mode requires L < num_blocks, got L={L}"
        return set(range(0, L + 1))
    if mode == "individual":
        assert L < num_blocks, f"individual mode requires L < num_blocks, got L={L}"
        return {L}
    raise AssertionError(f"Unknown layer mode: {mode!r}")


def apply_mask_to_layers(
    procs: dict[str, object],
    mode: LayerMode,
    L: int,
    ordered_block_names: Iterable[str],
    mask: torch.Tensor | None,
    *,
    window_size: int | None = None,
) -> None:
    """Install ``mask`` on the blocks selected by ``(mode, L)``; clear the rest.

    Blocks not in the selected set are explicitly set to ``_mask=None`` so
    masks from a previous iteration never leak. Passing ``mask=None`` clears
    every block regardless of ``mode``/``L`` — useful for equivalence checks
    against the stock processor.
    """
    ordered = list(ordered_block_names)
    selected: set[int] = (
        masked_indices(mode, L, len(ordered), window_size=window_size)
        if mask is not None
        else set()
    )
    for i, name in enumerate(ordered):
        assert name in procs, f"Block {name!r} not found in installed procs"
        procs[name]._mask = mask if i in selected else None


def apply_split_mask_to_layers(
    procs: dict[str, object],
    split: int,
    ordered_block_names: Iterable[str],
    prefix_mask: torch.Tensor | None,
    suffix_mask: torch.Tensor,
) -> None:
    """Install a split schedule: ``prefix_mask`` on blocks ``[0, split)`` and
    ``suffix_mask`` on blocks ``[split, num_blocks)``, in a single pass.

    Every block is assigned on every call, so a mask from a previous iteration
    can never leak. ``prefix_mask=None`` leaves the prefix blocks stock
    (``_mask`` set to ``None``) — that is the "suffix-only" schedule.

    Two sequential :func:`apply_mask_to_layers` calls cannot express this:
    each clears every non-selected block, so the second would wipe the first
    half. This assigns both halves at once.
    """
    ordered = list(ordered_block_names)
    num_blocks = len(ordered)
    assert 0 < split < num_blocks, (
        f"split={split} must be in (0, {num_blocks}) so both halves are "
        f"non-empty; a split at 0 or {num_blocks} is a single-mask run "
        f"(use apply_mask_to_layers)"
    )
    if prefix_mask is not None:
        assert prefix_mask.shape == suffix_mask.shape, (
            f"prefix_mask shape {tuple(prefix_mask.shape)} != suffix_mask "
            f"shape {tuple(suffix_mask.shape)}"
        )
    for i, name in enumerate(ordered):
        assert name in procs, f"Block {name!r} not found in installed procs"
        procs[name]._mask = prefix_mask if i < split else suffix_mask


def clear_all_masks(procs: dict[str, object]) -> None:
    """Clear ``_mask`` on every installed knockout processor."""
    for proc in procs.values():
        proc._mask = None


_TEXT_SUBSET_NAMED_SETTINGS: dict[str, KnockoutSetting] = {
    "ref->text[padding]": KnockoutSetting(("ref",), "text[padding]"),
    "ref->text[content]": KnockoutSetting(("ref",), "text[content]"),
    "text[padding]+ref->image": KnockoutSetting(("text[padding]", "ref"), "image"),
}


def resolve_settings(names: Iterable[str]) -> list[KnockoutSetting]:
    """Look up ``KnockoutSetting`` instances by ``.name`` (e.g. ``"ref->text"``).

    Accepts both the eight base settings in ``KNOCKOUT_SETTINGS`` and the three
    text-subset named settings (``ref->text[padding]``, ``ref->text[content]``,
    ``text[padding]+ref->image``). Composites such as ``image<->ref`` (which
    expand to multiple directives) are NOT handled here — callers that need
    them should consult ``COMPOSITE_KNOCKOUT_SETTINGS`` directly.
    """
    by_name = {s.name: s for s in KNOCKOUT_SETTINGS}
    resolved: list[KnockoutSetting] = []
    for name in names:
        if name in by_name:
            resolved.append(by_name[name])
            continue
        if name in _TEXT_SUBSET_NAMED_SETTINGS:
            resolved.append(_TEXT_SUBSET_NAMED_SETTINGS[name])
            continue
        known = sorted(list(by_name) + list(_TEXT_SUBSET_NAMED_SETTINGS))
        raise AssertionError(
            f"Unknown knockout setting {name!r}. Known: {known}"
        )
    return resolved
