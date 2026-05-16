"""Entry point for i2i-to-unconditional activation patching.

Thin adapter over :class:`I2IToUnconditionalRunner`. Resolves tasks via
:mod:`experiments.common.cli` and forwards every flag through. Per-task
output: ``results/i2i_to_unconditional/<edit_type>/<task_id>/<ts>/``.

Two sweep modes (``--sweep-mode`` accepts ONE OR MORE; pair element-wise with
``--patched-inference-steps``):
    input_to_block0  (default) all 32 sources patched as INPUT to
                     transformer_blocks.0 (via context_embedder), so block 0
                     processes each source's text stream as its own input.
                     Supports --patched-inference-steps > 1 for real
                     multi-step generation. Text-only.
    diagonal         src == dst block, patching the block's output
                     (1-step localization probe; supports both image and
                     text categories).

Five text-token modes (``--text-token-mode`` accepts ONE OR MORE; sweeps
concat). Only affect the "text" category:
    all                  (default) replace the full 512-token text slice
    per_content          one sweep per content-token in the instruction,
                         replacing only that single position
    per_position         one sweep per EVERY position 0..511 (expensive;
                         combine with --block-range to cap the block sweep)
    content_only         ONE sweep patching all content tokens together.
    padding_only         ONE sweep patching all padding tokens together.

Or pass explicit text-token positions with ``--text-token-indices N N N...``
to sweep each of those positions individually.

Usage:
    uv run python experiments/i2i_to_unconditional/i2i_to_unconditional_patch.py \\
        --task-id real_bedroom_tv

    uv run python experiments/i2i_to_unconditional/i2i_to_unconditional_patch.py \\
        --task-id real_bedroom_tv --text-token-mode per_content

    uv run python experiments/i2i_to_unconditional/i2i_to_unconditional_patch.py \\
        --edit-type add --limit 5

Run BOTH the diagonal-1step localization probe and the input_to_block0-4step
sweep in one invocation (shared model load + shared source-i2i capture):

    uv run python experiments/i2i_to_unconditional/i2i_to_unconditional_patch.py \\
        --task-id real_bedroom_tv \\
        --sweep-mode diagonal input_to_block0 \\
        --patched-inference-steps 1 4 \\
        --text-token-mode all content_only padding_only
"""

from __future__ import annotations

import argparse

from experiments.common.cli import add_output_overrides, add_task_selection, resolve_tasks
from experiments.i2i_to_unconditional.runner import (
    ALL_CATEGORIES,
    KNOCKOUT_SIDES,
    SOURCE_TARGET_SEED_OFFSET,
    SWEEP_MODES,
    TEXT_TOKEN_MODES,
    I2IToUnconditionalRunner,
)
from experiments.common.tasks import NUM_INFERENCE_STEPS
from utils.flux2_klein import ALL_BLOCK_NAMES, Flux2KleinModel, TEXT_SEQ_LEN
from utils.model_registry import load_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="i2i-to-unconditional activation patching: patch i2i token "
                    "categories into an empty-prompt t2i, sweeping all 32 blocks.",
    )
    add_task_selection(parser)
    add_output_overrides(parser)
    parser.add_argument(
        "--categories", nargs="+", choices=ALL_CATEGORIES, default=["text"],
        help="Token categories to sweep. Default: text (matches the default "
             "--sweep-mode input_to_block0, which is text-only). Pass "
             "'--categories image text' or 'image' only with "
             "--sweep-mode diagonal.",
    )
    parser.add_argument(
        "--sweep-mode", nargs="+", choices=SWEEP_MODES, default=["input_to_block0"],
        help="One or more sweep modes (paired element-wise with "
             "--patched-inference-steps). input_to_block0 patches all 32 "
             "sources as block 0 input via context_embedder, supporting "
             "multi-step generation. diagonal patches src==dst block output "
             "(1-step localization probe). Default: ['input_to_block0'].",
    )
    parser.add_argument(
        "--patched-inference-steps", type=int, nargs="+", default=[4],
        metavar="N",
        help="One or more inference-step counts (paired element-wise with "
             "--sweep-mode). For each pair, the t2i sweep AND the "
             f"unconditional/t2i-clean/KO bookends use N. Source i2i capture "
             f"always stays at NUM_INFERENCE_STEPS={NUM_INFERENCE_STEPS}. "
             "Values != NUM_INFERENCE_STEPS only valid for input_to_block0. "
             "Default: [4].",
    )
    parser.add_argument(
        "--text-token-mode", nargs="+", choices=TEXT_TOKEN_MODES,
        default=["all"],
        help="One or more text-token modes; sweeps concat (deduped by output "
             "subdir). 'all' replaces the full 512-token slice; "
             "'per_content' produces one grid per content token; "
             "'per_position' produces one grid per every 0..511 position; "
             "'content_only' / 'padding_only' produce ONE grid for content "
             "or padding alone. Default: ['all']. Ignored when "
             "--text-token-indices is set.",
    )
    parser.add_argument(
        "--text-token-indices", type=int, nargs="+", default=None,
        metavar="N",
        help="Explicit text-token positions (0..511) to sweep, one grid per "
             "position. When set, overrides --text-token-mode.",
    )
    parser.add_argument(
        "--block-range", type=int, nargs=2, default=None,
        metavar=("FIRST", "LAST"),
        help="Limit the block sweep to block indices [FIRST, LAST] inclusive. "
             "Default: all 32 blocks.",
    )
    parser.add_argument(
        "--position-range", type=int, nargs=2, default=None,
        metavar=("FIRST", "LAST"),
        help="For --text-token-mode per_position, restrict to text-token "
             "positions [FIRST, LAST] inclusive. Default: all 512 positions.",
    )
    parser.add_argument(
        "--target-seed-offset", type=int, default=SOURCE_TARGET_SEED_OFFSET,
        help=f"Offset from source i2i noise_seed to target t2i noise_seed. "
             f"Default {SOURCE_TARGET_SEED_OFFSET} decouples the two. "
             "Set to 0 ONLY to reproduce the pre-decoupling matched-noise regime.",
    )
    parser.add_argument(
        "--knockout-setting", type=str, nargs="+", default=None,
        metavar="SETTING",
        help="One or more attention-knockout setting names (e.g. 'image->text' "
             "'ref->text'); each must be shell-quoted due to the arrow. When "
             "multiple are given, the full task sweep runs once per setting. "
             "Requires --knockout-side.",
    )
    parser.add_argument(
        "--knockout-side", choices=KNOCKOUT_SIDES, default=None,
        help="Where to install the attention knockout: 'source' (during i2i "
             "capture), 'target' (during t2i generation in the sweep), or "
             "'both'. Required when --knockout-setting is set.",
    )
    args = parser.parse_args(argv)

    if "per_content" in args.text_token_mode:
        assert "text" in args.categories, (
            "--text-token-mode per_content requires 'text' in --categories"
        )
    if args.text_token_indices is not None:
        assert "text" in args.categories, (
            "--text-token-indices requires 'text' in --categories"
        )
        for i in args.text_token_indices:
            assert 0 <= i < TEXT_SEQ_LEN, (
                f"--text-token-indices value {i} out of range [0, {TEXT_SEQ_LEN})"
            )
    assert len(args.sweep_mode) == len(args.patched_inference_steps), (
        f"--sweep-mode and --patched-inference-steps must be paired "
        f"element-wise; got {len(args.sweep_mode)} mode(s) vs "
        f"{len(args.patched_inference_steps)} step count(s)"
    )
    assert len(args.sweep_mode) > 0, "--sweep-mode must have at least one entry"
    seen_pairs: set[tuple[str, int]] = set()
    for mode, n in zip(args.sweep_mode, args.patched_inference_steps):
        pair = (mode, n)
        assert pair not in seen_pairs, (
            f"duplicate (--sweep-mode, --patched-inference-steps) pair: {pair}"
        )
        seen_pairs.add(pair)
        if mode == "input_to_block0":
            assert args.categories == ["text"], (
                "--sweep-mode input_to_block0 only supports --categories text "
                "(context_embedder is text-only), got "
                f"--categories {' '.join(args.categories)}"
            )
        assert n >= 1, f"--patched-inference-steps must be >= 1, got {n}"
        if n != NUM_INFERENCE_STEPS:
            assert mode == "input_to_block0", (
                f"--patched-inference-steps={n} != NUM_INFERENCE_STEPS "
                f"({NUM_INFERENCE_STEPS}) is only supported with "
                f"--sweep-mode input_to_block0, got pair ({mode!r}, {n})"
            )
    if args.block_range is not None:
        first, last = args.block_range
        assert 0 <= first <= last < len(ALL_BLOCK_NAMES), (
            f"--block-range requires 0 <= FIRST <= LAST < {len(ALL_BLOCK_NAMES)}, "
            f"got FIRST={first}, LAST={last}"
        )
    if args.position_range is not None:
        assert "per_position" in args.text_token_mode, (
            "--position-range requires 'per_position' in --text-token-mode"
        )
        first, last = args.position_range
        assert 0 <= first <= last < TEXT_SEQ_LEN, (
            f"--position-range requires 0 <= FIRST <= LAST < {TEXT_SEQ_LEN}, "
            f"got FIRST={first}, LAST={last}"
        )
    if args.knockout_setting is not None:
        assert args.knockout_side is not None, (
            "--knockout-setting requires --knockout-side"
        )
    if args.knockout_side is not None:
        assert args.knockout_setting is not None, (
            "--knockout-side requires --knockout-setting"
        )
    return args


def main() -> None:
    args = parse_args()
    tasks = resolve_tasks(args)

    print(f"Loading model (flux2_klein)...")
    model = load_model("flux2_klein", device="cuda:0")
    assert isinstance(model, Flux2KleinModel)

    runner = I2IToUnconditionalRunner(model, extra_args=args)
    try:
        runner.run_many(tasks)
    finally:
        runner.teardown()
    print(f"\nAll experiments complete.")


if __name__ == "__main__":
    main()
