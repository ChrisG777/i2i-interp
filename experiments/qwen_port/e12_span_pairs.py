"""Qwen-Image-Edit port of the paper's i2i-to-i2i style-transfer patching.

Reruns the paper's 450 style->real pairs
(``experiments/i2i_to_i2i_patching/pairs/mm7_4step_style_to_real.txt``) on
Qwen-Image-Edit-2511 (60 uniform dual-stream blocks): for each pair, the
SOURCE run (stylized ref) and TARGET run (real-photo ref) share the same
content-only instruction (the leading "A photograph of " is dropped from
both runs), and the source's text-band activations are patched into the
target over a narrow block span, 29..38 by default (10 of the 60 blocks;
reported as layers 30-39 in the paper appendix). Bring-up sweeps located the
style-flip window at blocks 29-42 with the effect peaking at 29-31; use
``--block-lo``/``--block-hi`` for other windows (the output dir should be
span-tagged so runs never collide).

Text-band alignment: the target ref is center-cropped to the source ref's
aspect ratio (Qwen's ~384^2 VL condition resize makes the vision grid
aspect-dependent); pairs whose text bands still mismatch are skipped and
recorded in the diffs CSV.

Generation regime: Lightning distilled LoRA fused, 8 steps,
``true_cfg_scale=1.0`` (one conditional forward per step, the regime the
capture/patch machinery assumes). ``--no-lightning`` runs the base model.

Per-pair artifacts:

    <idx>_<target_task_id>/
        ref_source.png        stylized reference
        ref_target.png        real-photo reference, as used (aspect-cropped)
        source.png            SOURCE i2i run (stylized)
        baseline_target.png   TARGET i2i run, unpatched
        patched.png           TARGET run with the span text patch
        task_metadata.json    task ids, reworded + original prompt, seeds,
                              block span, model

Grading: ``uv run python -m scripts.run_judge --judge
qwen2511_i2i2i_style_span10 --model claude-sonnet-5`` (byte-identical style
question to the klein ``i2i2i_style`` cell; expects the outputs under
``results/qwen_port/e12_span29_38``).

Run:
    uv run python -m experiments.qwen_port.e12_span_pairs \\
        --start 0 --end 450 --block-lo 29 --block-hi 38 \\
        --out-dir results/qwen_port/e12_span29_38
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.common.tasks import load_tasks
from experiments.i2i_to_i2i_patching.pair_io import read_pair_list
from experiments.patching.hooks import make_patch_hook_multi_step
from experiments.patching.utils import run_pipeline_with_hooks
from utils.model_registry import load_real_reference
from utils.qwen_image_edit import ALL_BLOCK_NAMES, QwenImageEditModel, layout_for

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = (REPO_ROOT / "experiments" / "i2i_to_i2i_patching" / "pairs"
                 / "mm7_4step_style_to_real.txt")
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "qwen_port"


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--num-inference-steps", type=int, default=None,
        help="Default: the model's default (8 with Lightning, 50 without).",
    )
    p.add_argument(
        "--no-lightning", action="store_true",
        help="Skip the Lightning LoRA (base model, undistilled).",
    )
    p.add_argument(
        "--variant", choices=("v1", "2511"), default="2511",
        help="Checkpoint variant: original Qwen-Image-Edit (v1) or "
        "Qwen-Image-Edit-2511 (default).",
    )
    return p


def load_model(args) -> QwenImageEditModel:
    model = QwenImageEditModel(
        device=args.device, lightning=not args.no_lightning,
        variant=args.variant,
    )
    if args.num_inference_steps is None:
        args.num_inference_steps = model.default_num_inference_steps
    return model


def report_peak_memory(tag: str) -> None:
    """Print peak GPU + CPU memory for this process, for the batch log.

    ``ru_maxrss`` is kilobytes on Linux, bytes on macOS.
    """
    import resource
    import sys

    gib = 1024**3
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gib = maxrss / gib if sys.platform == "darwin" else maxrss / (1024**2)
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / gib
        reserved = torch.cuda.max_memory_reserved() / gib
        print(
            f"[{tag}] peak memory: GPU allocated {alloc:.1f} GiB, "
            f"GPU reserved {reserved:.1f} GiB, CPU RSS {rss_gib:.1f} GiB"
        )
    else:
        print(f"[{tag}] peak memory: CPU RSS {rss_gib:.1f} GiB (no CUDA)")


def center_crop_to_aspect(img: Image.Image, target_ar: float) -> Image.Image:
    w, h = img.size
    if w / h > target_ar:
        new_w = round(h * target_ar)
        box = ((w - new_w) // 2, 0, (w - new_w) // 2 + new_w, h)
    else:
        new_h = round(w / target_ar)
        box = (0, (h - new_h) // 2, w, (h - new_h) // 2 + new_h)
    return img.crop(box)


def content_instruction(paper_instruction: str) -> str:
    """"A photograph of the girl in this image X" -> "The girl in this image X"."""
    prefix = "A photograph of "
    assert paper_instruction.startswith(prefix), paper_instruction
    rest = paper_instruction[len(prefix):]
    return rest[0].upper() + rest[1:]


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument("--pairs-file", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=450)
    parser.add_argument("--block-lo", type=int, default=29)
    parser.add_argument("--block-hi", type=int, default=38,
                        help="Inclusive. Default span 29..38 = 10 blocks.")
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU-only: print the reworded prompts and check "
                             "text-band alignment; no generation.")
    args = parser.parse_args()

    assert 0 <= args.block_lo <= args.block_hi <= 59, (args.block_lo,
                                                       args.block_hi)
    blocks = [f"transformer_blocks.{i}"
              for i in range(args.block_lo, args.block_hi + 1)]
    assert set(blocks) <= set(ALL_BLOCK_NAMES)
    span_tag = f"{args.block_lo:02d}_{args.block_hi:02d}"
    print(f"[e12] span {args.block_lo}..{args.block_hi} "
          f"({len(blocks)} of {len(ALL_BLOCK_NAMES)} blocks)", flush=True)

    style = {t.task_id: t for t in load_tasks("style")}
    manual = {t.task_id: t for t in load_tasks("manual")}
    pairs = read_pair_list(args.pairs_file)[args.start:args.end]

    if args.dry_run:
        from utils.qwen_image_edit import load_processor
        processor = load_processor(variant=args.variant)
        model = None
    else:
        model = load_model(args)
        processor = model.processor

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    csv_rows = ["idx,source,target,text_len,mean_diff,max_diff,status"]
    ref_cache: dict[str, Image.Image] = {}

    def get_ref(task):
        key = f"{task.real_ref_dir}/{task.real_ref_name}"
        if key not in ref_cache:
            ref_cache[key] = load_real_reference(task.real_ref_name,
                                                 task.real_ref_dir)
        return ref_cache[key]

    n_ok = n_skip = 0
    for i, (src_id, tgt_id) in enumerate(pairs, start=args.start):
        s, t = style[src_id], manual[tgt_id]
        assert s.instruction == t.instruction, (src_id, tgt_id)
        instr = content_instruction(s.instruction)

        src_ref = get_ref(s)
        tgt_ref = center_crop_to_aspect(get_ref(t),
                                        src_ref.width / src_ref.height)
        kw = dict(prompt=instr, processor=processor, variant=args.variant)
        lay_s = layout_for(t.height, t.width, ref_image=src_ref, **kw)
        lay_t = layout_for(t.height, t.width, ref_image=tgt_ref, **kw)
        if lay_s.text_seq_len != lay_t.text_seq_len:
            csv_rows.append(f"{i},{src_id},{tgt_id},"
                            f"{lay_s.text_seq_len}/{lay_t.text_seq_len},,,SKIP")
            n_skip += 1
            print(f"[e12] {i} SKIP bands {lay_s.text_seq_len} vs "
                  f"{lay_t.text_seq_len}", flush=True)
            continue
        n_ok += 1
        if args.dry_run:
            csv_rows.append(f"{i},{src_id},{tgt_id},{lay_s.text_seq_len},,,OK")
            if i < args.start + 3:
                print(f"[e12] {i} prompt: {instr}", flush=True)
            continue

        pair_dir = out / f"{i:03d}_{tgt_id}"
        pair_dir.mkdir(exist_ok=True)
        n_steps = args.num_inference_steps

        # SOURCE run (stylized ref, content-only instruction). Only the span's
        # blocks are captured.
        src_img, cap = model.capture_activations(
            instr, s.noise_seed, list(blocks), captures_to_cpu=True,
            num_inference_steps=n_steps, height=t.height, width=t.width,
            image=src_ref,
        )
        txt = {b: [e[0] for e in cap[b]] for b in blocks}
        del cap

        baseline = model.generate(
            instr, seed=t.noise_seed, num_inference_steps=n_steps,
            height=t.height, width=t.width, image=tgt_ref,
        )

        hooks = [(b, make_patch_hook_multi_step(b, txt[b], "text", lay_t))
                 for b in blocks]
        gen = torch.Generator(model.device).manual_seed(t.noise_seed)
        patched = run_pipeline_with_hooks(
            model, hooks, prompt=instr, generator=gen,
            num_inference_steps=n_steps, height=t.height, width=t.width,
            image=tgt_ref, true_cfg_scale=1.0,
        )
        del txt

        src_ref.save(pair_dir / "ref_source.png")
        tgt_ref.save(pair_dir / "ref_target.png")
        src_img.save(pair_dir / "source.png")
        baseline.save(pair_dir / "baseline_target.png")
        patched.save(pair_dir / "patched.png")
        diff = np.abs(np.asarray(baseline).astype(np.int16)
                      - np.asarray(patched).astype(np.int16))
        (pair_dir / "task_metadata.json").write_text(json.dumps({
            "source_task_id": src_id,
            "target_task_id": tgt_id,
            "prompt": instr,
            "original_paper_prompt": s.instruction,
            "prompt_rewording": "dropped the leading 'A photograph of ' and "
                                "recapitalized; applied to BOTH runs",
            "source_noise_seed": s.noise_seed,
            "target_noise_seed": t.noise_seed,
            "patched_blocks": f"{blocks[0]}..{blocks[-1]} "
                              f"({len(blocks)} of 60 blocks, full text band)",
            "block_span": [args.block_lo, args.block_hi],
            "text_seq_len": lay_s.text_seq_len,
            "model": "Qwen-Image-Edit-2511 + Lightning 8-step, cfg 1.0",
            "ref_target_note": "center-cropped to source aspect ratio",
        }, indent=1))
        csv_rows.append(f"{i},{src_id},{tgt_id},{lay_s.text_seq_len},"
                        f"{diff.mean():.2f},{diff.max()},OK")
        print(f"[e12] {i} done mean diff {diff.mean():.1f} ({tgt_id})",
              flush=True)

    (out / f"diffs_{span_tag}_{args.start:03d}_{args.end:03d}.csv").write_text(
        "\n".join(csv_rows) + "\n")
    print(f"[e12] {n_ok} ok, {n_skip} skipped -> {out}", flush=True)
    if not args.dry_run:
        report_peak_memory("e12")


if __name__ == "__main__":
    main()
