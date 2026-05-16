"""Custom attention processors that inject an additive mask into SDPA.

Each processor is a trivial subclass of the default diffusers Flux2
processor — its ``__call__`` forwards to ``super().__call__`` with
``attention_mask`` set from an instance attribute. This means:

- When ``_mask is None``, the parent processor runs with
  ``attention_mask=None`` — numerically identical to the default
  processor.
- When ``_mask`` is a tensor, the parent calls
  ``dispatch_attention_fn(..., attn_mask=mask)``, which forwards to
  ``F.scaled_dot_product_attention``. SDPA adds the mask to
  ``QK^T / sqrt(d)`` before softmax. ``0`` entries are no-ops;
  ``-inf`` entries are killed pre-softmax by SDPA itself.

Consequence: we never write an attention formula. The only custom code
is one argument to SDPA, so there is no way to re-introduce the
hand-rolled-attention numerical mismatch that motivated this design.
"""

from __future__ import annotations

from typing import Callable

import torch

from diffusers.models.transformers.transformer_flux2 import (
    Flux2AttnProcessor,
    Flux2ParallelSelfAttnProcessor,
)


__all__ = [
    "KnockoutFlux2AttnProcessor",
    "KnockoutFlux2ParallelSelfAttnProcessor",
    "install_knockout_processors",
    "install_processors_by_factory",
    "restore_processors",
]


def install_processors_by_factory(
    transformer,
    factory: Callable[[str, bool], object],
) -> dict:
    """Install custom attention processors on every block of ``transformer``.

    ``factory(block_name, is_single) -> processor`` is called per block; the
    short ``block_name`` matches ``ALL_BLOCK_NAMES`` (e.g.
    ``"transformer_blocks.3"``); ``is_single`` is True for
    ``single_transformer_blocks.*``.

    Returns the original processors dict for use with ``restore_processors``.
    """
    original = transformer.attn_processors.copy()
    new_processors = {}
    for name in original:
        block_name = name.rsplit(".processor", 1)[0]
        is_single = "single_transformer_blocks" in name
        new_processors[name] = factory(block_name, is_single)
    transformer.set_attn_processor(new_processors)
    return original


def restore_processors(transformer, original: dict) -> None:
    """Restore the original attention processors."""
    transformer.set_attn_processor(original)


class KnockoutFlux2AttnProcessor(Flux2AttnProcessor):
    """MM-block processor that injects ``self._mask`` into SDPA."""

    def __init__(self, block_name: str):
        super().__init__()
        self.block_name = block_name
        self._mask: torch.Tensor | None = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert attention_mask is None, (
            f"{self.block_name}: external attention_mask collides with knockout"
        )
        return super().__call__(
            attn,
            hidden_states,
            encoder_hidden_states,
            self._mask,
            image_rotary_emb,
        )


class KnockoutFlux2ParallelSelfAttnProcessor(Flux2ParallelSelfAttnProcessor):
    """Single-block processor that injects ``self._mask`` into SDPA."""

    def __init__(self, block_name: str):
        super().__init__()
        self.block_name = block_name
        self._mask: torch.Tensor | None = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert attention_mask is None, (
            f"{self.block_name}: external attention_mask collides with knockout"
        )
        return super().__call__(
            attn,
            hidden_states,
            self._mask,
            image_rotary_emb,
        )


def install_knockout_processors(transformer) -> tuple[dict[str, object], dict]:
    """Install knockout processors on every block of ``transformer``.

    Args:
        transformer: The Flux2 transformer module (``pipe.transformer``).

    Returns:
        A tuple ``(procs_by_block_name, original)``:
          * ``procs_by_block_name`` maps short block names (matching
            ``utils.flux2_klein.ALL_BLOCK_NAMES`` — e.g.
            ``"transformer_blocks.3"``) to the installed processor
            instances. Mutate ``._mask`` on these to enable/disable
            knockouts per block.
          * ``original`` is the pre-install processors dict, to pass
            back to ``restore_processors``.

    The shared helper passes block names with a trailing ``.attn``
    (its own convention, used by attention_analysis processors). We
    strip that here so our keys match ``ALL_BLOCK_NAMES``.
    """
    procs_by_block_name: dict[str, object] = {}

    def factory(block_name: str, is_single: bool):
        short = block_name.removesuffix(".attn")
        cls = (
            KnockoutFlux2ParallelSelfAttnProcessor
            if is_single
            else KnockoutFlux2AttnProcessor
        )
        inst = cls(short)
        procs_by_block_name[short] = inst
        return inst

    original = install_processors_by_factory(transformer, factory)
    return procs_by_block_name, original
