"""Abstract base class for diffusion model wrappers.

Subclasses (Flux2KleinModel, QwenImageEditModel) supply the architecture
metadata and the ``pipe`` / ``transformer`` / ``device`` triple; the
generation and activation-capture methods are shared.
"""

from typing import Dict, List, Tuple

import torch
from baukit.nethook import TraceDict
from PIL import Image


class DiffusionModel:
    """Unified interface for diffusion model wrappers.

    Subclasses must set the architecture properties and ``pipe`` /
    ``transformer`` / ``device`` in their ``__init__``.  The generation
    methods are shared — only loading differs between models.
    """

    # -- Set by subclass __init__ ------------------------------------------
    pipe: object               # diffusers pipeline (Flux2KleinPipeline, …)
    transformer: torch.nn.Module
    device: str

    # Architecture metadata (read by HookBuilder)
    name: str                  # "flux2_klein"
    num_heads: int             # attention heads per block
    head_dim: int              # dimension per head
    inner_dim: int             # num_heads * head_dim
    text_seq_len: int | None   # fixed text token count, or None if per-task
    has_bias: bool             # whether Linear layers have bias
    has_fused_single_qkv: bool # True if single blocks use a fused to_qkv_mlp_proj

    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        seed: int,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 0.0,
        **kwargs,
    ) -> Image.Image:
        """Generate a single image with a deterministic seed."""
        generator = torch.Generator(self.device).manual_seed(seed)
        output = self.pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            **kwargs,
        )
        return output.images[0]

    def capture_activations(
        self,
        prompt: str,
        seed: int,
        capture_layers: List[str],
        *,
        captures_to_cpu: bool = False,
        **gen_kwargs,
    ) -> Tuple[Image.Image, Dict[str, list]]:
        """Generate an image while capturing activations at specified layers.

        Runs ``self.generate(prompt, seed=seed, **gen_kwargs)`` under a baukit
        ``TraceDict``, so subclass generation defaults/overrides apply to the
        captured run too.

        Args:
            captures_to_cpu: If True, move captured tensors to CPU inside the
                hook to conserve GPU memory when capturing many layers.

        Returns:
            Tuple of (image, activations_dict) where activations_dict maps
            layer names to lists of captured outputs (one per forward pass).
        """
        captured: Dict[str, list] = {name: [] for name in capture_layers}

        def capture_fn(output, layer):
            if isinstance(output, tuple):
                tensors = tuple(o.detach().cpu().clone() if captures_to_cpu else o.detach().clone() for o in output)
                captured[layer].append(tensors)
            else:
                t = output.detach().cpu().clone() if captures_to_cpu else output.detach().clone()
                captured[layer].append(t)
            return output

        with TraceDict(
            self.transformer,
            layers=capture_layers,
            edit_output=capture_fn,
        ):
            image = self.generate(prompt, seed=seed, **gen_kwargs)

        return image, captured
