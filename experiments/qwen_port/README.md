# Qwen-Image-Edit: I2I-to-I2I style patching

Port of the I2I-to-I2I patching experiment to **Qwen-Image-Edit-2511**, the
paper's second unified-attention editing model (60 uniform dual-stream MMDiT
blocks; the reference image conditions the model through *two* routes: VAE
ref latents in the image stream, and Qwen2.5-VL vision tokens inside the
text stream).

The paper's appendix cell: rerun the same 450 style->real pairs as the klein
`i2i2i_style` experiment, with a content-only instruction (the leading
"A photograph of " dropped from both runs) and the source run's text-band
activations patched into the target at blocks 30-39 (of 60). This transfers
the source style in 96.7% of pairs, versus 87.6% on FLUX.2 with a
single-block patch.

Model/layout code lives in [`utils/qwen_image_edit.py`](../../utils/qwen_image_edit.py);
the shared `TokenLayout` (with a vision-token span) in
[`utils/token_layout.py`](../../utils/token_layout.py). The klein patch hooks
apply unchanged: Qwen blocks return the same `(txt, img)` tuple and are all
named `transformer_blocks.<i>`.

Default generation regime: **Lightning distilled LoRA, fused**, 8 steps,
`true_cfg_scale=1.0` (one conditional forward per step, the regime the
capture/patch machinery assumes). `--no-lightning` runs the base model
(50-step default, still cfg=1; expect weaker adherence).

## Running

```bash
uv run python -m experiments.qwen_port.e12_span_pairs \
    --start 0 --end 450 --block-lo 29 --block-hi 38 \
    --out-dir results/qwen_port/e12_span29_38
```

(`--block-lo`/`--block-hi` are 0-indexed; 29..38 is the paper's layers
30-39.) `--dry-run` checks text-band alignment on CPU without generating.
Per-pair outputs (`ref_source.png`, `ref_target.png`, `source.png`,
`baseline_target.png`, `patched.png`, `task_metadata.json`) land under
`--out-dir`, one directory per pair.

Grade with the same style question as the klein cell:

```bash
uv run python -m scripts.run_judge \
    --judge qwen2511_i2i2i_style_span10 --model claude-sonnet-5
```

GPU need: one H200/H100 (bf16 transformer ~40 GB + Qwen2.5-VL-7B encoder),
64 GB CPU RAM for the model load.
