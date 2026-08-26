"""Qwen-Image-Edit model wrapper and sequence-layout helpers.

Architecture (diffusers ``QwenImageTransformer2DModel``): 60 uniform
dual-stream MMDiT blocks in a single ``transformer_blocks`` list — no
single-stream blocks. Each block returns ``(encoder_hidden_states,
hidden_states)``, the same ``(txt, img)`` contract as klein's MM blocks, so
the MM patch hooks in ``experiments/patching/hooks.py`` apply unchanged.

Conditioning differs from klein in that the reference image enters twice:

* **appearance route** — VAE latents concatenated after the noise latents in
  the image stream (``hidden_states = [noise | ref]``), same ordering as
  FLUX.2;
* **semantic route** — the reference image is also fed to the Qwen2.5-VL text
  encoder together with the prompt, so the text stream contains a contiguous
  band of vision tokens (``TokenLayout.vision_start/vision_end``). The VL
  encoder is causal with the image *before* the prompt, so prompt tokens are
  ref-contaminated at block 0 while vision tokens never see the prompt.

Default generation regime is the Lightning distilled LoRA at
``true_cfg_scale=1.0`` (one conditional forward per step — the regime the
capture/patch machinery assumes) with 8 steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from PIL import Image

from utils.model_base import DiffusionModel
from utils.token_layout import TokenLayout, seq_len_for_image

MODEL_ID = "Qwen/Qwen-Image-Edit"

# Lightning distillation LoRA (lightx2v). Distilled specifically for
# true_cfg_scale=1.0 at few steps; the repo ROOT holds the LoRAs for the
# original Qwen-Image-Edit (subfolders hold 2509 variants; 2511 has its own
# repo, see VARIANTS below).
LIGHTNING_REPO = "lightx2v/Qwen-Image-Lightning"
LIGHTNING_WEIGHT = "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"
LIGHTNING_NUM_STEPS = 8
BASE_NUM_STEPS = 50  # pipeline default for the undistilled model

# Lightning samples with a constant exponential time shift of 3
# (mu = log(3) regardless of resolution). From the Qwen-Image-Lightning
# diffusers example.
LIGHTNING_SCHEDULER_OVERRIDES = {
    "base_shift": math.log(3),
    "max_shift": math.log(3),
    "use_dynamic_shifting": True,
    "time_shift_type": "exponential",
}

NUM_BLOCKS = 60

ALL_BLOCK_NAMES = [f"transformer_blocks.{i}" for i in range(NUM_BLOCKS)]
ALL_BLOCK_LABELS = [f"Dual {i}" for i in range(NUM_BLOCKS)]


def block_suffix(block_name: str) -> str:
    """Compact filename suffix for a block name: ``transformer_blocks.7`` ->
    ``dual7``. The klein convention (``mm7``/``single_mm9``) is not reused so
    on-disk artifacts from the two models can never be confused."""
    assert block_name in _NAME_TO_INDEX, (
        f"unknown block name {block_name!r} (expected transformer_blocks.0..{NUM_BLOCKS - 1})"
    )
    return block_name.replace("transformer_blocks.", "dual")


_NAME_TO_INDEX = {name: i for i, name in enumerate(ALL_BLOCK_NAMES)}
_SUFFIX_TO_INDEX = {f"dual{i}": i for i in range(NUM_BLOCKS)}


def block_index_from_suffix(suffix: str) -> int:
    """Inverse of :func:`block_suffix`: ``dual7`` -> 7."""
    assert suffix in _SUFFIX_TO_INDEX, (
        f"unknown block suffix {suffix!r}; expected dual0..dual{NUM_BLOCKS - 1}"
    )
    return _SUFFIX_TO_INDEX[suffix]


# Pixels per image token side: vae_scale_factor (8) * patchify (2) — same
# value as klein's VAE_PATCH.
VAE_PATCH = 16

# Reference images are ALWAYS resized to ~1024^2 area (aspect preserved, dims
# rounded — not floored — to multiples of 32). Mirrors the module-level
# ``calculate_dimensions(1024 * 1024, w / h)`` call in
# ``QwenImageEditPipeline.__call__``; unlike klein there is no
# "only if larger" condition.
_REF_TARGET_AREA = 1024 * 1024

# Mirror of ``QwenImageEditPipeline.prompt_template_encode`` /
# ``prompt_template_encode_start_idx``. The wrapper asserts these against the
# loaded pipeline so diffusers drift fails fast instead of silently
# desynchronizing the layout math. Note the ordering: system prompt (dropped
# from the final embeds), then vision tokens, then the user prompt.
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image "
    "(color, shape, size, texture, objects, background), then explain how "
    "the user's text instruction should alter or modify the image. Generate "
    "a new image that meets the user's requirements while maintaining "
    "consistency with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "{}<|im_end|>\n<|im_start|>assistant\n"
)
DROP_IDX = 64  # leading template tokens dropped from the final text embeds

# ---------------------------------------------------------------------------
# Checkpoint variants
#
# "2511" (Qwen-Image-Edit-2511, ``QwenImageEditPlusPipeline``) has a
# config-identical 60-block transformer — every DiT-level hook, knockout
# region, and block name is shared with "v1". The differences are
# conditioning-only:
#
# * the VL condition image is resized to ~384^2 area (vs the full ~1024^2),
#   so the vision band shrinks from ~1369 to ~200 tokens;
# * the template no longer bakes in the vision markers — the pipeline
#   prepends ``"Picture 1: <|vision_start|><|image_pad|><|vision_end|>"`` to
#   the user content (multi-image support; we use exactly one ref);
# * the VAE route is unchanged (~1024^2 area -> same ref latent band);
# * the Lightning LoRA lives in its own repo.
# ---------------------------------------------------------------------------

# Plus template: identical system prompt, vision markers NOT baked in.
PLUS_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image "
    "(color, shape, size, texture, objects, background), then explain how "
    "the user's text instruction should alter or modify the image. Generate "
    "a new image that meets the user's requirements while maintaining "
    "consistency with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
# Mirror of ``img_prompt_template`` in the Plus pipeline's
# ``_get_qwen_prompt_embeds`` (single-ref case).
PLUS_VISION_PREFIX = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"

# Plus: VL condition image target area (VAE route keeps _REF_TARGET_AREA).
_PLUS_CONDITION_AREA = 384 * 384


@dataclass(frozen=True)
class VariantSpec:
    key: str
    model_id: str
    pipeline_class_name: str
    lightning_repo: str
    lightning_weight: str
    condition_area: int   # VL condition image target area
    template: str         # mirror of pipe.prompt_template_encode
    vision_prefix: str    # prepended to the user content ("" for v1)


VARIANTS = {
    "v1": VariantSpec(
        key="v1",
        model_id=MODEL_ID,
        pipeline_class_name="QwenImageEditPipeline",
        lightning_repo=LIGHTNING_REPO,
        lightning_weight=LIGHTNING_WEIGHT,
        condition_area=_REF_TARGET_AREA,
        template=PROMPT_TEMPLATE,
        vision_prefix="",
    ),
    "2511": VariantSpec(
        key="2511",
        model_id="Qwen/Qwen-Image-Edit-2511",
        pipeline_class_name="QwenImageEditPlusPipeline",
        lightning_repo="lightx2v/Qwen-Image-Edit-2511-Lightning",
        lightning_weight="Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
        condition_area=_PLUS_CONDITION_AREA,
        template=PLUS_PROMPT_TEMPLATE,
        vision_prefix=PLUS_VISION_PREFIX,
    ),
}


def _calculate_dimensions(target_area: int, ref_h: int, ref_w: int) -> tuple[int, int]:
    """Mirror of the pipelines' ``calculate_dimensions``: scale to the target
    area preserving aspect ratio, then ``round`` both dims to a multiple of
    32."""
    assert ref_h > 0 and ref_w > 0, f"ref dims must be positive, got ({ref_h}, {ref_w})"
    ratio = ref_w / ref_h
    w = math.sqrt(target_area * ratio)
    h = w / ratio
    w = round(w / 32) * 32
    h = round(h / 32) * 32
    assert h > 0 and w > 0, f"degenerate effective dims for ({ref_h}, {ref_w})"
    return h, w


def effective_ref_dims(ref_h: int, ref_w: int) -> tuple[int, int]:
    """Dims of the reference on the VAE (appearance) route — ~1024^2 area.
    Identical for both variants (the Plus pipeline's ``VAE_IMAGE_SIZE``)."""
    return _calculate_dimensions(_REF_TARGET_AREA, ref_h, ref_w)


def effective_condition_dims(
    ref_h: int, ref_w: int, variant: str = "v1"
) -> tuple[int, int]:
    """Dims of the reference as fed to the Qwen2.5-VL encoder (semantic
    route). v1 shares the VAE dims (~1024^2 area); 2511 shrinks the condition
    image to ~384^2 (the Plus pipeline's ``CONDITION_IMAGE_SIZE``)."""
    return _calculate_dimensions(VARIANTS[variant].condition_area, ref_h, ref_w)


def load_processor(model_id: str | None = None, variant: str = "v1"):
    """Load only the Qwen2.5-VL processor (tokenizer + image grid logic).

    Enough for all layout/span math — no transformer or VL weights are
    downloaded, so this runs on the CPU-only dev machine.
    """
    # Explicit class: AutoProcessor mis-resolves the ``processor`` subfolder
    # (which holds tokenizer files alongside the VL preprocessor) to a bare
    # Qwen2Tokenizer.
    from transformers import Qwen2VLProcessor

    if model_id is None:
        model_id = VARIANTS[variant].model_id
    return Qwen2VLProcessor.from_pretrained(model_id, subfolder="processor")


@dataclass(frozen=True)
class QwenPromptSpans:
    """Token spans within the text band the transformer actually sees
    (i.e. after the leading ``DROP_IDX`` template tokens are dropped).

    ``vision_start:vision_end`` is the ``<|image_pad|>`` band (the
    ``<|vision_start|>``/``<|vision_end|>`` marker tokens belong to neither
    subset). ``instruction_start:instruction_end`` is the raw instruction —
    the tokens strictly between ``<|vision_end|>`` and the closing
    ``<|im_end|>``.
    """

    text_seq_len: int
    vision_start: int
    vision_end: int
    instruction_start: int
    instruction_end: int

    def __post_init__(self) -> None:
        assert 0 <= self.vision_start < self.vision_end, (
            f"empty vision span [{self.vision_start}, {self.vision_end})"
        )
        assert self.vision_end < self.instruction_start, (
            f"instruction must start after the vision band "
            f"({self.instruction_start} <= {self.vision_end})"
        )
        assert self.instruction_start < self.instruction_end <= self.text_seq_len, (
            f"instruction span [{self.instruction_start}, {self.instruction_end}) "
            f"out of range for text_seq_len={self.text_seq_len}"
        )


def qwen_prompt_spans(
    processor,
    prompt: str,
    ref_image: Image.Image,
    variant: str = "v1",
) -> QwenPromptSpans:
    """Tokenize ``prompt`` + ``ref_image`` exactly as the pipeline does and
    return the resulting text-band spans.

    Mirrors the pipeline's preprocessing: the reference is resized to
    :func:`effective_condition_dims` before the VL processor sees it (the
    vision token count depends only on the image dims fed to the processor,
    not the resampling filter), and for "2511" the pipeline's
    ``"Picture 1: ..."`` vision prefix is prepended to the user content.
    """
    assert prompt.strip(), f"prompt must be non-empty, got {prompt!r}"
    spec = VARIANTS[variant]
    calc_h, calc_w = effective_condition_dims(
        ref_image.height, ref_image.width, variant
    )
    img = ref_image.convert("RGB").resize((calc_w, calc_h), Image.LANCZOS)

    inputs = processor(
        text=[spec.template.format(spec.vision_prefix + prompt)],
        images=[img],
        padding=True,
        return_tensors="pt",
    )
    ids = inputs.input_ids[0].tolist()
    assert bool(inputs.attention_mask[0].all()), (
        "single-prompt tokenization should have no padding"
    )

    tok = processor.tokenizer
    image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    vision_end_id = tok.convert_tokens_to_ids("<|vision_end|>")
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

    pad_pos = [i for i, t in enumerate(ids) if t == image_pad_id]
    assert pad_pos, "no <|image_pad|> tokens in the tokenized template"
    assert pad_pos == list(range(pad_pos[0], pad_pos[-1] + 1)), (
        "vision-token band is not contiguous"
    )
    v0, v1 = pad_pos[0], pad_pos[-1] + 1
    assert ids[v1] == vision_end_id, (
        f"expected <|vision_end|> right after the pad band, got id {ids[v1]}"
    )
    assert v0 >= DROP_IDX, (
        f"vision band starts at {v0}, inside the dropped template prefix "
        f"(DROP_IDX={DROP_IDX}) — template drift?"
    )

    instr_start = v1 + 1
    im_end_after = [i for i in range(instr_start, len(ids)) if ids[i] == im_end_id]
    assert im_end_after, "no closing <|im_end|> after the instruction"
    instr_end = im_end_after[0]

    return QwenPromptSpans(
        text_seq_len=len(ids) - DROP_IDX,
        vision_start=v0 - DROP_IDX,
        vision_end=v1 - DROP_IDX,
        instruction_start=instr_start - DROP_IDX,
        instruction_end=instr_end - DROP_IDX,
    )


def layout_for(
    target_h: int,
    target_w: int,
    *,
    ref_image: Image.Image,
    prompt: str,
    processor,
    variant: str = "v1",
) -> TokenLayout:
    """Build the joint-attention :class:`TokenLayout` for one i2i generation.

    Unlike klein, the text band length is per-task (it contains the VL vision
    tokens plus the tokenized prompt), so building a layout requires the
    processor, the prompt, and the reference image. The ref latent band uses
    the VAE dims for BOTH variants; only the vision band depends on the
    variant's condition-image area.
    """
    assert target_h % VAE_PATCH == 0 and target_w % VAE_PATCH == 0, (
        f"target dims ({target_h}, {target_w}) must be multiples of "
        f"{VAE_PATCH} (the pipeline floors them silently; pass aligned dims)"
    )
    spans = qwen_prompt_spans(processor, prompt, ref_image, variant)
    calc_h, calc_w = effective_ref_dims(ref_image.height, ref_image.width)
    return TokenLayout(
        text_seq_len=spans.text_seq_len,
        noise_seq_len=seq_len_for_image(target_h, target_w, patch=VAE_PATCH),
        ref_seq_len=seq_len_for_image(calc_h, calc_w, patch=VAE_PATCH),
        vision_start=spans.vision_start,
        vision_end=spans.vision_end,
    )


class QwenImageEditModel(DiffusionModel):
    """Wrapper for QwenImageEditPipeline with baukit hook-based interventions.

    ``lightning=True`` (default) loads and fuses the Lightning distilled LoRA
    and installs its constant-shift scheduler; generation then runs 8 steps at
    ``true_cfg_scale=1.0``. Fusing (rather than keeping the adapter) matters
    for interpretability work: hooks and attention processors see an ordinary
    transformer with unchanged module names and no peft indirection.
    """

    def __init__(
        self,
        model_id: str | None = None,
        torch_dtype=torch.bfloat16,
        device: str = "cuda:0",
        lightning: bool = True,
        variant: str = "v1",
    ):
        import diffusers
        from diffusers import FlowMatchEulerDiscreteScheduler

        spec = VARIANTS[variant]
        if model_id is None:
            model_id = spec.model_id
        pipeline_cls = getattr(diffusers, spec.pipeline_class_name)

        self.pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=torch_dtype)
        # The layout math in this module mirrors the pipeline's template and
        # drop index; fail fast if a diffusers upgrade changes either.
        assert self.pipe.prompt_template_encode == spec.template, (
            f"{spec.pipeline_class_name}.prompt_template_encode drifted from "
            f"the utils.qwen_image_edit mirror for variant {variant!r} — "
            f"update the mirror"
        )
        assert self.pipe.prompt_template_encode_start_idx == DROP_IDX, (
            f"prompt_template_encode_start_idx="
            f"{self.pipe.prompt_template_encode_start_idx} != DROP_IDX={DROP_IDX}"
        )

        if lightning:
            self.pipe.load_lora_weights(
                spec.lightning_repo, weight_name=spec.lightning_weight
            )
            self.pipe.fuse_lora()
            self.pipe.unload_lora_weights()
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                {**self.pipe.scheduler.config, **LIGHTNING_SCHEDULER_OVERRIDES}
            )

        self.pipe.to(device)
        self.device = device
        self.transformer = self.pipe.transformer
        self.pipe.set_progress_bar_config(disable=True)

        cfg = self.transformer.config
        assert cfg.num_layers == NUM_BLOCKS, (
            f"expected {NUM_BLOCKS} blocks, got {cfg.num_layers}"
        )
        assert not cfg.guidance_embeds, (
            "Qwen-Image-Edit should not be guidance-distilled; the generate() "
            "contract (no guidance_scale) assumes guidance_embeds=False"
        )

        self.lightning = lightning
        self.variant = variant
        self.default_num_inference_steps = (
            LIGHTNING_NUM_STEPS if lightning else BASE_NUM_STEPS
        )

        # Architecture metadata
        self.name = "qwen_image_edit" if variant == "v1" else f"qwen_image_edit_{variant}"
        self.num_heads = cfg.num_attention_heads  # 24
        self.head_dim = cfg.attention_head_dim    # 128
        self.inner_dim = self.num_heads * self.head_dim  # 3072
        self.text_seq_len = None  # per-task: VL vision tokens + prompt tokens
        self.has_bias = True
        self.has_fused_single_qkv = False

    @property
    def processor(self):
        """The Qwen2.5-VL processor (for layout/span math)."""
        return self.pipe.processor

    def generate(
        self,
        prompt: str,
        seed: int,
        num_inference_steps: int | None = None,
        height: int | None = None,
        width: int | None = None,
        **kwargs,
    ) -> Image.Image:
        """Generate one edited image with a deterministic seed.

        ``height``/``width`` of ``None`` use the pipeline's default (the
        reference image's effective dims). ``image`` is required —
        QwenImageEditPipeline has no pure-t2i mode.
        """
        assert kwargs.get("image") is not None, (
            "QwenImageEditPipeline always conditions on a reference image; "
            "pass image=<PIL image>"
        )
        assert "guidance_scale" not in kwargs, (
            "Qwen-Image-Edit is not guidance-distilled; guidance_scale would "
            "be ignored — drop it"
        )
        true_cfg_scale = kwargs.pop("true_cfg_scale", 1.0)
        assert true_cfg_scale == 1.0, (
            f"true_cfg_scale={true_cfg_scale}: the capture/patch machinery "
            f"assumes exactly one conditional forward per step (no negative "
            f"branch). CFG-aware hooks are not implemented."
        )
        if num_inference_steps is None:
            num_inference_steps = self.default_num_inference_steps
        generator = torch.Generator(self.device).manual_seed(seed)
        output = self.pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            true_cfg_scale=true_cfg_scale,
            **kwargs,
        )
        return output.images[0]
