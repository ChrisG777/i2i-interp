"""Entry point for the i2i attention-knockout sweep.

Thin adapter over :class:`AttentionKnockoutRunner`. Parses CLI flags
(--task-id / --edit-type, --settings, --layer-mode, --window-size,
--all-layers-4step), resolves tasks via
:mod:`experiments.common.cli`, and dispatches one task at a time.

Per-task output: ``results/attention_knockout/<edit_type>/<task_id>/<ts>/``.

Usage:
    uv run python experiments/attention_knockout/knockout_run.py \\
        --task-id solid_red_couch

    uv run python experiments/attention_knockout/knockout_run.py \\
        --task-id solid_red_couch --settings 'image->text' --all-layers-4step

    uv run python experiments/attention_knockout/knockout_run.py \\
        --edit-type add --limit 3
"""

from __future__ import annotations

import argparse

from experiments.attention_knockout.runner import (
    ALL_KNOWN_SETTINGS,
    NUM_BLOCKS,
    TEXT_SUBSET_SETTINGS,
    AttentionKnockoutRunner,
)
from experiments.attention_knockout.masks import LAYER_MODES
from experiments.common.cli import add_output_overrides, add_task_selection, resolve_tasks
from utils.flux2_klein import ALL_BLOCK_NAMES, Flux2KleinModel
from utils.model_registry import load_model


DEFAULT_SETTINGS: list[str] = list(ALL_KNOWN_SETTINGS)


def _resolve_settings_arg(settings_arg: list[str]) -> list[str]:
    if settings_arg == ["all"]:
        return list(ALL_KNOWN_SETTINGS)
    for name in settings_arg:
        assert name in ALL_KNOWN_SETTINGS, (
            f"Unknown setting {name!r}. Known: {ALL_KNOWN_SETTINGS + ['all']}"
        )
    return list(dict.fromkeys(settings_arg))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attention knockout sweep: mask pre-softmax attention "
                    "between token categories (or image-token subsets) on "
                    "subsets of transformer blocks.",
    )
    add_task_selection(parser)
    add_output_overrides(parser)
    parser.add_argument(
        "--settings", type=str, nargs="+", default=list(DEFAULT_SETTINGS),
        help=f"Knockout setting(s) to sweep, or 'all' for every known setting. "
             f"Default: every known setting. "
             f"Options: {', '.join(ALL_KNOWN_SETTINGS + ['all'])}",
    )
    parser.add_argument(
        "--layer-mode", type=str, nargs="+", default=["prefix"],
        choices=list(LAYER_MODES),
        help="Which layer-selection modes to sweep. Multiple modes run "
             "sequentially so the model loads once. Default: ['prefix'].",
    )
    parser.add_argument(
        "--window-size", type=int, nargs="+", default=[3],
        help="Window size(s) k for --layer-mode window. Each k sweeps every "
             "sliding window of exactly k consecutive blocks. Ignored unless "
             "'window' is in --layer-mode. Default: [3].",
    )
    parser.add_argument(
        "--all-layers-4step", action="store_true",
        help="After each layer-mode sweep, also generate one extra image per "
             "(task, setting): num_inference_steps=4 with the mask installed "
             "on every block. Saved as full_ko_4step.png and appended as a "
             "single trailing 'i2i full KO 4step' cell in every grid for that "
             "(task, setting). Off by default.",
    )
    parser.add_argument(
        "--full-ko-only", action="store_true",
        help="Generate only the full-KO image per (task, setting) and skip "
             "the per-block layer sweep. Saved as full_ko.png (and "
             "full_ko_4step.png if --all-layers-4step). The bulk of GPU time "
             "per setting is the layer sweep, so this is the fast path for "
             "iterating on the full-asymptote behavior. --layer-mode and "
             "--window-size are ignored when this is set.",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=None, metavar="N",
        help="Override the number of inference steps for the i2i baseline, "
             "t2i clean, and full-KO generations. Defaults to "
             "experiments.common.tasks.NUM_INFERENCE_STEPS (1). The paper-scale "
             "knockout runs use --num-inference-steps 4 so the full-KO image "
             "matches the 4-step baseline.",
    )
    parser.add_argument(
        "--split-block", type=str, nargs="+", default=None, metavar="BLOCK_NAME",
        help="Enable split-schedule knockout. Each BLOCK_NAME (e.g. "
             "'single_transformer_blocks.2') is the first block of the suffix: "
             "the prefix mask covers blocks [0, split) and the suffix mask "
             "covers [split, NUM_BLOCKS). Passing several block names sweeps "
             "multiple cutoffs in one run. Requires --suffix-setting and "
             "--full-ko-only --num-inference-steps 4; --prefix-setting is "
             "optional. Mutually exclusive with the per-block layer sweep "
             "(--layer-mode / --window-size / --settings are ignored).",
    )
    parser.add_argument(
        "--prefix-setting", type=str, nargs="*", default=None,
        help="Knockout setting name(s) OR-unioned into the mask installed on "
             "the prefix blocks [0, split). Omit for a suffix-only schedule "
             "(prefix blocks left stock). Single-quote names containing '->'.",
    )
    parser.add_argument(
        "--suffix-setting", type=str, nargs="+", default=None,
        help="Knockout setting name(s) OR-unioned into the mask installed on "
             "the suffix blocks [split, NUM_BLOCKS). Single-quote names "
             "containing '->'.",
    )
    args = parser.parse_args(argv)

    args.settings = _resolve_settings_arg(args.settings)
    args.layer_mode = list(dict.fromkeys(args.layer_mode))
    args.window_size = list(dict.fromkeys(args.window_size))
    if "window" in args.layer_mode:
        for k in args.window_size:
            assert 1 <= k <= NUM_BLOCKS, (
                f"--window_size {k} out of range [1, {NUM_BLOCKS}]"
            )

    # Split-schedule validation. --split-block + --suffix-setting are required
    # together; --prefix-setting is optional (absent => suffix-only schedule).
    is_split = args.split_block is not None
    assert is_split == (args.suffix_setting is not None), (
        "--split-block and --suffix-setting must be given together "
        "(or neither, for the standard --settings sweep)"
    )
    if is_split:
        args.split_block = list(dict.fromkeys(args.split_block))
        split_index: list[int] = []
        for name in args.split_block:
            try:
                idx = ALL_BLOCK_NAMES.index(name)
            except ValueError:
                raise AssertionError(
                    f"Unknown --split-block {name!r}. Valid block names: "
                    f"{list(ALL_BLOCK_NAMES)}"
                )
            assert 0 < idx < NUM_BLOCKS, (
                f"--split-block {name!r} is block index {idx}; the split must "
                f"be in (0, {NUM_BLOCKS}) so both halves are non-empty"
            )
            split_index.append(idx)
        args.split_index = split_index
        args.suffix_setting = list(dict.fromkeys(args.suffix_setting))
        args.prefix_setting = (
            list(dict.fromkeys(args.prefix_setting))
            if args.prefix_setting is not None else []
        )
        for name in (*args.prefix_setting, *args.suffix_setting):
            assert name in ALL_KNOWN_SETTINGS, (
                f"Unknown split setting {name!r}. Known: {ALL_KNOWN_SETTINGS}"
            )
            assert name not in TEXT_SUBSET_SETTINGS, (
                f"Split-schedule settings must be full-category; {name!r} is a "
                f"text-subset setting"
            )
        assert args.full_ko_only, (
            "--split-block requires --full-ko-only (the schedule is the layer "
            "assignment; there is no per-block sweep)"
        )
        assert args.num_inference_steps == 4, (
            f"--split-block requires --num-inference-steps 4, got "
            f"{args.num_inference_steps}"
        )
    else:
        args.split_index = None
    return args


def main() -> None:
    args = parse_args()
    tasks = resolve_tasks(args)

    print(f"Loading model (flux2_klein)...")
    model = load_model("flux2_klein")
    assert isinstance(model, Flux2KleinModel)

    runner = AttentionKnockoutRunner(model, extra_args=args)
    try:
        runner.run_many(tasks)
    finally:
        runner.teardown()
    if args.split_index is not None:
        print(
            f"\nAll done. Ran {len(tasks)} split-schedule task(s): "
            f"cutoffs={args.split_block}, "
            f"prefix={args.prefix_setting or 'stock'}, "
            f"suffix={args.suffix_setting}"
        )
    else:
        print(
            f"\nAll done. Ran {len(tasks)} task(s) "
            f"with modes {args.layer_mode} and settings {args.settings}: "
            f"{', '.join(t.task_id for t in tasks)}"
        )


if __name__ == "__main__":
    main()
