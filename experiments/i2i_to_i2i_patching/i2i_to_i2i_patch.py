"""Entry point for i2i-to-i2i text-token activation patching.

Thin adapter over :class:`I2IToI2IRunner`. Takes one or more
``--pair SOURCE_TASK_ID TARGET_TASK_ID`` arguments. Each pair must satisfy:
same instruction (relaxable via ``--allow-mismatched-instruction``) and
different ref (always enforced). Noise seeds may differ freely across the
two sides.

Per-pair output:
``results/i2i_to_i2i_patching/<source_task_id>__<target_task_id>/<run_timestamp>/``.

Usage:
    uv run python experiments/i2i_to_i2i_patching/i2i_to_i2i_patch.py \\
        --pair manual_solid_yellow_couch manual_solid_blue_couch \\
        --pair manual_dog_style manual_dog_real
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common.cli import add_output_overrides
from experiments.common.tasks import NUM_INFERENCE_STEPS, TaskDefinition, get_task
from experiments.i2i_to_i2i_patching.pair_io import read_pair_list
from experiments.i2i_to_i2i_patching.runner import I2IToI2IRunner
from utils.flux2_klein import ALL_BLOCK_NAMES, Flux2KleinModel
from utils.model_registry import load_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="i2i-to-i2i text-token activation patching: capture "
                    "activations from a source i2i run and patch text tokens "
                    "into a target i2i run, sweeping across blocks.",
    )
    pair_src = parser.add_mutually_exclusive_group(required=True)
    pair_src.add_argument(
        "--pair", nargs=2, action="append", metavar=("SOURCE", "TARGET"),
        default=None,
        help="A (source_task_id, target_task_id) pair. Repeatable for batch runs.",
    )
    pair_src.add_argument(
        "--pair-list", type=str, default=None, metavar="PATH",
        help="Path to a text file of pairs, one per line, "
             "tab- or whitespace-separated SOURCE TARGET. Blank lines "
             "and lines starting with '#' are ignored.",
    )
    add_output_overrides(parser)
    parser.add_argument(
        "--block-range", type=int, nargs=2, default=None,
        metavar=("FIRST", "LAST"),
        help="Limit the block sweep to block indices [FIRST, LAST] inclusive. "
             "Default: all 32 blocks. Useful for fixture / smoke runs.",
    )
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help="0-indexed shard to run from the pair list (for SLURM arrays).",
    )
    parser.add_argument(
        "--shard-total", type=int, default=1,
        help="Total number of shards. Default: 1 (no sharding).",
    )
    parser.add_argument(
        "--allow-mismatched-instruction", action="store_true",
        help="Skip the same-instruction assert. Source and target may have "
             "different instructions (each used on its own side).",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS,
        help="Steps for both source capture and target generation. With "
             "N>1, source captures one activation per layer per step; the "
             "multi-step patch hook patches the source's step-k activation "
             "into the target at step k (same target block every step). "
             f"Default {NUM_INFERENCE_STEPS} (single-step, original behavior).",
    )
    parser.add_argument(
        "--text-token-mode", nargs="+",
        choices=["all", "padding_only", "content_only"], default=["all"],
        help="Which text-token positions to patch. ``all`` patches all 512 "
             "text-token slots (default, original behavior). ``padding_only`` "
             "patches only the Qwen3 padding positions (complement of the "
             "real prompt content). ``content_only`` patches only the content "
             "positions themselves. Pass any combination to run multiple "
             "variants in the same task dir, sharing reference + baselines.",
    )
    args = parser.parse_args(argv)
    assert args.num_inference_steps >= 1, (
        f"--num-inference-steps must be >= 1, got {args.num_inference_steps}"
    )
    if args.block_range is not None:
        first, last = args.block_range
        assert 0 <= first <= last < len(ALL_BLOCK_NAMES), (
            f"--block-range requires 0 <= FIRST <= LAST < {len(ALL_BLOCK_NAMES)}, "
            f"got {first} {last}"
        )
    assert 0 <= args.shard_index < args.shard_total, (
        f"--shard-index must be in [0, --shard-total); "
        f"got {args.shard_index} / {args.shard_total}"
    )
    return args


def _resolve_pairs(args: argparse.Namespace) -> list[tuple[TaskDefinition, TaskDefinition]]:
    if args.pair_list is not None:
        raw = read_pair_list(Path(args.pair_list))
    else:
        raw = [(s, t) for s, t in args.pair]
    pairs = [(get_task(s), get_task(t)) for s, t in raw]
    # Take this shard's slice of pairs.
    return pairs[args.shard_index :: args.shard_total]


def main() -> None:
    args = parse_args()
    pairs = _resolve_pairs(args)
    total_src = (
        sum(1 for _ in open(args.pair_list))
        if args.pair_list is not None else len(args.pair)
    )
    assert len(pairs) > 0, (
        f"No pairs in this shard (shard_index={args.shard_index}, "
        f"shard_total={args.shard_total}, total_pairs={total_src})"
    )

    print(f"Loading model (flux2_klein)...")
    model = load_model("flux2_klein", device="cuda:0")
    assert isinstance(model, Flux2KleinModel)

    runner = I2IToI2IRunner(model, extra_args=args)
    runner.run_pairs(pairs)
    print(f"\nAll pairs complete.")


if __name__ == "__main__":
    main()
