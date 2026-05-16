"""Runner for the i2i-to-unconditional activation-patching experiment.

Subclass of :class:`experiments.common.runner.ExperimentRunner`. Per-task
output: ``results/i2i_to_unconditional/<edit_type>/<task_id>/<run_timestamp>/``.

Filenames, suptitle, and per-mode/per-knockout subdirectory layout match the
pre-Phase-4 layout exactly when the run uses a single (sweep_mode,
patch_steps) pair and a single text_token_mode, so grid PNGs stay
byte-identical against migrated fixtures. When the run uses multiple sweep
specs, baseline filenames carry an ``_{N}step`` suffix so the differing-step
versions don't collide (single-pair runs drop the suffix to preserve byte
identity).

When ``--knockout-setting`` is set, ``run_many`` loops settings around tasks
(one pass over all tasks per setting). When unset, a single pass runs each
task with ``setting=None``. Tasks with empty ``instruction`` combined with
content/padding-requiring text-token modes raise an assertion at startup
(no silent skipping).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from experiments.attention_knockout.knockout_processors import (
    KnockoutFlux2AttnProcessor,
    KnockoutFlux2ParallelSelfAttnProcessor,
    install_knockout_processors,
    restore_processors,
)
from experiments.attention_knockout.masks import (
    KnockoutSetting,
    apply_mask_to_layers,
    build_knockout_mask,
    clear_all_masks,
    resolve_settings,
)
from experiments.common.file_cache import load_or_run
from experiments.common.runner import ExperimentRunner
from experiments.common.tasks import NUM_INFERENCE_STEPS, TaskDefinition
from experiments.patching.sweep import (
    make_input_to_block0_producer,
    make_patch_pipeline_producer,
    sweep_and_grid,
)
from experiments.patching.utils import (
    get_token_strings,
    resolve_content_token_indices,
)
from utils.flux2_klein import (
    ALL_BLOCK_NAMES,
    Flux2KleinModel,
    TEXT_SEQ_LEN,
    TokenLayout,
    layout_for,
)

ALL_CATEGORIES = ["image", "text"]
SWEEP_MODES = ["diagonal", "input_to_block0"]
TEXT_TOKEN_MODES = [
    "all",
    "per_content",
    "per_position",
    "content_only",
    "padding_only",
]
KNOCKOUT_SIDES = ["source", "target", "both"]

_KNOCKOUT_PROCESSOR_TYPES = (
    KnockoutFlux2AttnProcessor,
    KnockoutFlux2ParallelSelfAttnProcessor,
)

# When patching from i2i to t2i, we deliberately use DIFFERENT noise seeds for
# the source (i2i capture) and target (t2i generation). Matched noise would
# introduce a spurious identity coupling between source and target noise
# latents unrelated to the patch; using distinct seeds isolates the patch
# contribution.
SOURCE_TARGET_SEED_OFFSET = 1
assert SOURCE_TARGET_SEED_OFFSET != 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slug(s: str) -> str:
    """Filesystem-safe slug for a token string."""
    import re
    s = s.strip() or "pad"
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_") or "tok"


def _token_strings_for_prompt(pipe, instruction_prompt: str) -> list[str]:
    if not instruction_prompt.strip():
        return ["<pad>"] * TEXT_SEQ_LEN
    return get_token_strings(pipe, instruction_prompt)


def _fixed_dst_for_sweep_mode(sweep_mode: str):
    if sweep_mode == "diagonal":
        return None, None
    if sweep_mode == "input_to_block0":
        return "context_embedder", "→block 0 input"
    raise ValueError(f"Unknown sweep_mode={sweep_mode!r}")


def _assert_knockout_state(transformer, expect_knockout: bool) -> int:
    """Walk every attention module and confirm the processor types match
    what the knockout flag implies. Raises on mismatch.
    """
    inspected = 0
    mismatches = []
    for name, module in transformer.named_modules():
        processor = getattr(module, "processor", None)
        if processor is None:
            continue
        inspected += 1
        is_knockout = isinstance(processor, _KNOCKOUT_PROCESSOR_TYPES)
        if expect_knockout and not is_knockout:
            mismatches.append(
                f"{name}: expected knockout processor, got "
                f"{type(processor).__name__}"
            )
        elif not expect_knockout and is_knockout:
            mismatches.append(
                f"{name}: knockout processor installed but no KO flags set"
            )
    assert not mismatches, (
        "Attention-processor state does not match knockout flags:\n  "
        + "\n  ".join(mismatches)
    )
    return inspected


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class I2IToUnconditionalRunner(ExperimentRunner):
    name = "i2i_to_unconditional"
    results_root = "results/i2i_to_unconditional"

    def __init__(
        self,
        model: Flux2KleinModel,
        *,
        extra_args: argparse.Namespace,
    ) -> None:
        super().__init__(model, extra_args=extra_args)
        assert isinstance(model, Flux2KleinModel)

        self.categories: list[str] = list(extra_args.categories)
        # Paired (sweep_mode, patched_inference_steps) list. CLI parser asserts
        # equal length and no duplicate pairs.
        self.sweep_specs: list[tuple[str, int]] = list(zip(
            extra_args.sweep_mode, extra_args.patched_inference_steps,
        ))
        assert len(self.sweep_specs) > 0, "sweep_specs must be non-empty"
        # Dedup-preserve-order so user can ask for overlapping modes safely.
        seen: set[str] = set()
        self.text_token_modes: list[str] = []
        for m in extra_args.text_token_mode:
            if m not in seen:
                seen.add(m)
                self.text_token_modes.append(m)
        self.text_token_indices: list[int] | None = extra_args.text_token_indices
        self.target_seed_offset: int = extra_args.target_seed_offset
        # Step counts that need bookend baselines: every patch step count plus
        # the source-capture step count (which is fixed at NUM_INFERENCE_STEPS).
        self.distinct_steps: list[int] = sorted(
            {n for _, n in self.sweep_specs} | {NUM_INFERENCE_STEPS}
        )
        # When there's only a single distinct step count, drop the _{N}step
        # suffix from baseline filenames to preserve byte-identity with the
        # canon fixture (which uses --patched-inference-steps 1, NUM=1).
        self.suffix_baseline_steps: bool = len(self.distinct_steps) > 1
        self.block_slice: slice | None = (
            slice(extra_args.block_range[0], extra_args.block_range[1] + 1)
            if extra_args.block_range is not None
            else None
        )
        self.position_range: tuple[int, int] | None = (
            tuple(extra_args.position_range)  # type: ignore[arg-type]
            if extra_args.position_range is not None
            else None
        )

        # Knockout state. Settings list is empty when no KO is requested.
        self.knockout_side: str | None = extra_args.knockout_side
        self.knockout_settings: list[KnockoutSetting] = (
            resolve_settings(extra_args.knockout_setting)
            if extra_args.knockout_setting is not None
            else []
        )
        self.knockout_procs: dict | None = None
        self.knockout_mask_dtype: torch.dtype | None = None
        self._original_procs = None
        self._current_setting: KnockoutSetting | None = None

        if self.knockout_settings:
            self.knockout_procs, self._original_procs = install_knockout_processors(
                model.transformer,
            )
            self.knockout_mask_dtype = next(model.transformer.parameters()).dtype
            print(
                f"Installed knockout processors; "
                f"settings={[s.name for s in self.knockout_settings]} "
                f"side={self.knockout_side}"
            )

        # Flat layout requires a single (sweep_mode, patched_steps) pair, a
        # single text_token_mode, a single block in --block-range, and no
        # knockout (the paper-scale T2I-unc judges read one patched cell per
        # task and the layout reflects that).
        if self.is_flat_layout:
            assert len(self.sweep_specs) == 1, (
                "--results-subdir requires a single (sweep_mode, "
                "patched_inference_steps) pair"
            )
            valid_flat_modes = {"all", "padding_only", "content_only"}
            bad_modes = [m for m in self.text_token_modes if m not in valid_flat_modes]
            assert not bad_modes, (
                f"--results-subdir supports only {sorted(valid_flat_modes)}; "
                f"got {self.text_token_modes!r}"
            )
            assert len(set(self.text_token_modes)) == len(self.text_token_modes), (
                f"duplicate entries in --text-token-mode: {self.text_token_modes!r}"
            )
            assert (
                self.extra_args.block_range is not None
                and self.extra_args.block_range[0] == self.extra_args.block_range[1]
            ), "--results-subdir requires a single-block --block-range"
            assert not self.knockout_settings, (
                "--results-subdir is incompatible with --knockout-setting"
            )

        # Sanity check at construction so misconfigurations fail before any
        # generation kicks off.
        n_attn = _assert_knockout_state(
            model.transformer, expect_knockout=bool(self.knockout_settings),
        )
        ko_label = (
            f"KO ACTIVE ({self.knockout_side}, "
            f"{[s.name for s in self.knockout_settings]})"
            if self.knockout_settings
            else "NO KO (stock attention processors)"
        )
        print(f"Knockout state verified at runtime: {ko_label} across {n_attn} attention modules.")

    def teardown(self) -> None:
        if self._original_procs is not None:
            restore_processors(self.model.transformer, self._original_procs)
            self._original_procs = None

    # ------------------------------------------------------------------
    # Driver: setting × task loop
    # ------------------------------------------------------------------

    def run_many(self, tasks: list[TaskDefinition]) -> list[Path]:
        assert len(tasks) > 0, "run_many: empty task list"
        # Fail fast if any task has an empty instruction but the requested
        # text-token modes need content tokens. Previously this was a silent
        # skip; we now assert at startup so misconfigurations don't waste
        # cluster time.
        content_modes = {"per_content", "content_only", "padding_only"}
        if any(m in content_modes for m in self.text_token_modes):
            bad = [t.task_id for t in tasks if not t.instruction.strip()]
            assert not bad, (
                f"text_token_modes={self.text_token_modes} requires non-empty "
                f"instruction; these tasks have empty instructions: {bad}"
            )
        setting_passes: list[KnockoutSetting | None] = (
            list(self.knockout_settings) if self.knockout_settings else [None]
        )
        out_dirs: list[Path] = []
        kept: list[TaskDefinition] = []
        for setting in setting_passes:
            if setting is not None:
                print(f"\n{'#' * 60}\n# Setting: {setting.name} ({self.knockout_side})\n{'#' * 60}")
            for task in tasks:
                expected = self.expected_artifacts(task)
                if self._skip_if_completed and self.is_completed(task.task_id, expected):
                    print(f"[skip] {task.task_id} (completed)")
                    continue
                self._current_setting = setting
                out_dir = self.run_one(task)
                out_dirs.append(out_dir)
                kept.append(task)
                torch.cuda.empty_cache()
        if kept:
            self._write_run_metadata(kept, out_dirs)
        return out_dirs

    def _flat_layout_block_suffix(self) -> tuple[str, str]:
        """Return (src_suffix, dst_suffix) used for flat-layout patched
        filenames. Mirrors the logic in :meth:`_finalize_flat_layout`."""
        sweep_mode, _ = self.sweep_specs[0]
        block_idx = self.extra_args.block_range[0]
        block_name = ALL_BLOCK_NAMES[block_idx]
        src_suffix = block_name.replace(
            "transformer_blocks.", "mm",
        ).replace("single_transformer_blocks.", "single")
        if sweep_mode == "input_to_block0":
            dst_suffix = "context_embedder"
        else:
            assert sweep_mode == "diagonal", f"unexpected sweep_mode={sweep_mode}"
            dst_suffix = src_suffix
        return src_suffix, dst_suffix

    def _flat_filename_for_mode(self, mode: str) -> str:
        """Flat-layout patched filename for a single text_token_mode. Modes
        coexist by filename suffix so they share the task dir."""
        src_suffix, _ = self._flat_layout_block_suffix()
        if mode == "all":
            return f"patched_{src_suffix}.png"
        if mode == "padding_only":
            return f"patched_{src_suffix}_text_padding.png"
        assert mode == "content_only", f"flat layout doesn't support mode={mode!r}"
        return f"patched_{src_suffix}_text_content.png"

    def _mode_for_cat_subdir_leaf(self, cat_subdir_leaf: str) -> str | None:
        """Reverse map a sweep-loop ``cat_subdir_leaf`` back to its
        text_token_mode in flat layout. Returns ``None`` when the leaf doesn't
        correspond to any flat-layout mode (e.g. nested-only modes)."""
        if cat_subdir_leaf == "text_tokens":
            return "all"
        if cat_subdir_leaf == os.path.join("text", "all_padding"):
            return "padding_only"
        if cat_subdir_leaf == os.path.join("text", "all_content"):
            return "content_only"
        return None

    def _flat_filename_for_group_exists(
        self, cat_subdir_leaf: str, save_dir: Path,
    ) -> bool:
        mode = self._mode_for_cat_subdir_leaf(cat_subdir_leaf)
        if mode is None:
            return False
        return (save_dir / self._flat_filename_for_mode(mode)).exists()

    def expected_artifacts(self, task: TaskDefinition) -> list[str]:
        """Files the current invocation will produce under ``task_dir(task)``.
        Only meaningful in flat layout."""
        if not self.is_flat_layout:
            return []
        _, patch_steps = self.sweep_specs[0]
        files: list[str] = ["reference.png"]
        for stem in ("source_i2i", "t2i_clean", "unconditional_baseline"):
            files.append(f"{stem}_{patch_steps}step.png")
        for mode in self.text_token_modes:
            files.append(self._flat_filename_for_mode(mode))
        return files

    # ------------------------------------------------------------------
    # Per-task
    # ------------------------------------------------------------------

    def run_one(self, task: TaskDefinition) -> Path:
        save_dir = self.task_dir(task)
        save_dir.mkdir(parents=True, exist_ok=True)

        instruction_prompt = task.instruction
        source_seed = task.noise_seed
        assert source_seed is not None, (
            f"task {task.task_id}: I2IToUnconditionalRunner requires noise_seed"
        )
        target_seed = source_seed + self.target_seed_offset
        if self.target_seed_offset == 0:
            print(
                "  ****  WARNING: --target-seed-offset 0 -> source and target "
                f"share seed {source_seed}. Generations are matched-noise, "
                "reproducing the pre-decoupling regime. Do not use for "
                "primary results.  ****"
            )
        else:
            assert source_seed != target_seed, (
                f"Source and target noise seeds must differ with offset "
                f"{self.target_seed_offset} but coincide at {source_seed}"
            )

        # Per-task metadata file (replaces the old args.json).
        self.write_task_metadata(
            task,
            extra={
                "categories": self.categories,
                "sweep_specs": [list(p) for p in self.sweep_specs],
                "text_token_modes": list(self.text_token_modes),
                "text_token_indices": self.text_token_indices,
                "block_range": (
                    list(self.extra_args.block_range)
                    if self.extra_args.block_range is not None
                    else None
                ),
                "position_range": (
                    list(self.extra_args.position_range)
                    if self.extra_args.position_range is not None
                    else None
                ),
                "target_seed_offset": self.target_seed_offset,
                "distinct_steps": list(self.distinct_steps),
                "source_seed": source_seed,
                "target_seed": target_seed,
                "knockout_setting": (
                    self._current_setting.name if self._current_setting else None
                ),
                "knockout_side": self.knockout_side,
            },
        )

        ref_label = task.ref_label

        # Suptitle/print labels — single-spec / single-mode runs render
        # exactly like the pre-refactor format so fixture grids stay
        # byte-identical.
        if len(self.sweep_specs) == 1:
            sweep_label = self.sweep_specs[0][0]
        else:
            sweep_label = "+".join(f"{m}-{n}step" for m, n in self.sweep_specs)
        text_mode_label = "+".join(self.text_token_modes)

        setting = self._current_setting
        if setting is not None:
            assert self.knockout_side in KNOCKOUT_SIDES
            assert self.knockout_procs is not None and self.knockout_mask_dtype is not None

        ko_on_source = setting is not None and self.knockout_side in {"source", "both"}
        ko_on_target = setting is not None and self.knockout_side in {"target", "both"}

        print(f"\n{'='*60}")
        print(
            f"Task: {task.task_id}  |  sweep_mode={sweep_label}  |  "
            f"text_token_mode={text_mode_label}"
        )
        if setting is not None:
            print(f"Knockout: {setting.name} on side={self.knockout_side}")
        print(f"Save dir: {save_dir}")
        print(f"{'='*60}")

        # Phase 0: reference.
        print(f"\n[Phase 0] Reference image: {ref_label}")
        ref_image = self.reference_image(task)
        if not (save_dir / "reference.png").exists():
            ref_image.save(save_dir / "reference.png")

        ref_w, ref_h = ref_image.size
        source_layout = layout_for(task.height, task.width, ref_h=ref_h, ref_w=ref_w)
        target_layout = layout_for(task.height, task.width)

        # Text-subset knockout settings (e.g. ref->text[padding]) need a
        # 1D bool tensor marking which text positions are content tokens.
        # Cheap to build any time KO is on; build_knockout_mask ignores it
        # for non-subset settings.
        text_content_mask = None
        if setting is not None:
            content, _ = self._content_padding_indices(instruction_prompt)
            text_content_mask = torch.zeros(source_layout.text_seq_len, dtype=torch.bool)
            text_content_mask[content] = True

        def _baseline_path(base: str, n: int) -> Path:
            if self.suffix_baseline_steps:
                return save_dir / f"{base}_{n}step.png"
            return save_dir / f"{base}.png"

        # Phase 1: source i2i + capture (always at NUM_INFERENCE_STEPS so
        # captured activations match the source the patch experiments expect).
        print(
            f"[Phase 1] Source i2i: '{instruction_prompt}' "
            f"(source_seed={source_seed}, target_seed={target_seed})"
        )
        if ko_on_source:
            assert setting is not None
            src_mask = build_knockout_mask(
                source_layout, setting,
                device=self.model.device, dtype=self.knockout_mask_dtype,
                text_content_mask=text_content_mask,
            )
            apply_mask_to_layers(
                self.knockout_procs, "suffix", 0, ALL_BLOCK_NAMES, src_mask,
            )
            print(f"  Source KO active: {setting.name}")
        source_img, source_captured = self.model.capture_activations(
            instruction_prompt,
            source_seed,
            list(ALL_BLOCK_NAMES),
            num_inference_steps=NUM_INFERENCE_STEPS,
            height=task.height,
            width=task.width,
            image=ref_image,
            captures_to_cpu=True,
        )
        if ko_on_source:
            clear_all_masks(self.knockout_procs)
        # source_imgs_by_step maps step count -> visual source-i2i image at
        # that step count. Only the NUM_INFERENCE_STEPS entry has matching
        # captures (others are visuals only).
        source_imgs_by_step: dict[int, Image.Image] = {NUM_INFERENCE_STEPS: source_img}
        src_baseline_path = _baseline_path("source_i2i", NUM_INFERENCE_STEPS)
        if not src_baseline_path.exists():
            source_img.save(src_baseline_path)

        # Phase 1.5: extra source-i2i visuals at non-NUM step counts (visual
        # only, no capture). Skipped when distinct_steps == [NUM_INFERENCE_STEPS].
        for n in self.distinct_steps:
            if n == NUM_INFERENCE_STEPS:
                continue
            print(f"[Phase 1.5] Source i2i visual at {n} steps (no capture)")
            extra_src = load_or_run(
                _baseline_path("source_i2i", n),
                generate=lambda n=n: self.model.generate(
                    instruction_prompt, seed=source_seed,
                    num_inference_steps=n,
                    height=task.height, width=task.width,
                    image=ref_image,
                ),
            )
            source_imgs_by_step[n] = extra_src

        # Phase 1b (source-side KO only): no-KO i2i for visual comparison
        # (single image at NUM_INFERENCE_STEPS — preserves prior behavior).
        source_img_no_ko = None
        if ko_on_source:
            print(f"[Phase 1b] Reference i2i WITHOUT KO (seed={source_seed})")
            source_img_no_ko = self.model.generate(
                instruction_prompt, seed=source_seed,
                num_inference_steps=NUM_INFERENCE_STEPS,
                height=task.height, width=task.width,
                image=ref_image,
            )

        # Phase 2: unconditional baseline at every distinct step count.
        unconditional_by_step: dict[int, Image.Image] = {}
        for n in self.distinct_steps:
            print(
                f"[Phase 2] Unconditional t2i baseline "
                f"(target_seed={target_seed}, steps={n})"
            )
            img = load_or_run(
                _baseline_path("unconditional_baseline", n),
                generate=lambda n=n: self.model.generate(
                    prompt="", seed=target_seed, num_inference_steps=n,
                    height=task.height, width=task.width,
                ),
            )
            unconditional_by_step[n] = img

        # Phase 2b: t2i clean at every distinct step count.
        t2i_clean_by_step: dict[int, Image.Image] = {}
        for n in self.distinct_steps:
            print(
                f"[Phase 2b] t2i clean: '{instruction_prompt}' "
                f"(target_seed={target_seed}, no ref, steps={n})"
            )
            img = load_or_run(
                _baseline_path("t2i_clean", n),
                generate=lambda n=n: self.model.generate(
                    prompt=instruction_prompt, seed=target_seed,
                    num_inference_steps=n,
                    height=task.height, width=task.width,
                ),
            )
            t2i_clean_by_step[n] = img

        # KO target-side mask installation: install once before sweeps so
        # both the KO baselines (Phase 2c) and the patch sweeps see it.
        ko_baselines_by_step: dict[int, Image.Image] = {}
        if ko_on_target:
            assert setting is not None
            target_mask = build_knockout_mask(
                target_layout, setting,
                device=self.model.device, dtype=self.knockout_mask_dtype,
                text_content_mask=text_content_mask,
            )
            apply_mask_to_layers(
                self.knockout_procs, "suffix", 0, ALL_BLOCK_NAMES, target_mask,
            )
            for n in self.distinct_steps:
                print(
                    f"[Phase 2c] Unconditional t2i baseline WITH "
                    f"KO={setting.name} (steps={n})"
                )
                ko_baseline_img = self.model.generate(
                    prompt="", seed=target_seed, num_inference_steps=n,
                    height=task.height, width=task.width,
                )
                ko_baselines_by_step[n] = ko_baseline_img

        # Suptitle base — keeps the pre-refactor format byte-identical for
        # single-spec single-mode runs.
        suptitle = (
            f"i2i → unconditional | instruction: '{instruction_prompt}' | "
            f"sweep={sweep_label} | text_token_mode={text_mode_label}\n"
            f"ref: {ref_label} | source_seed={source_seed} target_seed={target_seed}"
        )
        if setting is not None:
            suptitle += f" | KO={setting.name} ({self.knockout_side})"
        source_label = (
            f"Source (i2i, KO={setting.name})" if ko_on_source else "Source (i2i)"
        )

        # Phase 3: per-spec sweep loop.
        for sweep_mode, patch_steps in self.sweep_specs:
            fixed_dst_name, fixed_dst_label = _fixed_dst_for_sweep_mode(sweep_mode)

            if setting is not None:
                safe_setting = setting.name.replace("->", "_to_")
                mode_subdir = os.path.join(
                    sweep_mode, f"knockout_{safe_setting}_{self.knockout_side}",
                )
            else:
                mode_subdir = os.path.join(sweep_mode, "no_knockout")

            # Bookends matched to this spec's step count.
            bookend_images = [
                source_imgs_by_step[NUM_INFERENCE_STEPS],
                unconditional_by_step[patch_steps],
                t2i_clean_by_step[patch_steps],
            ]
            bookend_labels = [
                source_label,
                f"Unconditional baseline ({patch_steps}-step)"
                if self.suffix_baseline_steps else "Unconditional baseline",
                f"t2i clean ({patch_steps}-step)"
                if self.suffix_baseline_steps else "t2i clean",
            ]
            if source_img_no_ko is not None:
                bookend_images.append(source_img_no_ko)
                bookend_labels.append("Source (no KO)")
            if ko_on_target:
                ko_img = ko_baselines_by_step[patch_steps]
                ko_path = save_dir / mode_subdir / (
                    f"unconditional_baseline_with_ko_{patch_steps}step.png"
                    if self.suffix_baseline_steps
                    else "unconditional_baseline_with_ko.png"
                )
                ko_path.parent.mkdir(parents=True, exist_ok=True)
                ko_img.save(ko_path)
                bookend_images.append(ko_img)
                bookend_labels.append(f"KO baseline ({setting.name})")

            # Optionally save the source-i2i-without-KO copy under THIS
            # spec's mode_subdir (preserves prior behavior).
            if source_img_no_ko is not None:
                src_no_ko_path = save_dir / mode_subdir / "source_i2i_without_ko.png"
                src_no_ko_path.parent.mkdir(parents=True, exist_ok=True)
                source_img_no_ko.save(src_no_ko_path)

            for category in self.categories:
                if sweep_mode == "input_to_block0":
                    assert category == "text"

                if category == "text":
                    groups = self._text_sweep_groups(instruction_prompt)
                else:
                    groups = [(f"{category}_tokens", None, f"full {category} slice")]

                print(
                    f"\n[Phase 3] sweep_mode={sweep_mode} steps={patch_steps} "
                    f"category={category} ({len(groups)} group(s))"
                )
                for cat_subdir_leaf, indices, log_label in groups:
                    # Flat layout per-mode skip: each text-token mode lands at
                    # a unique top-level filename (patched_<sweep>.png for the
                    # full slice, patched_<sweep>_text_padding.png for padding-
                    # only). If the file already exists, skip the sweep so a
                    # re-run that adds a new mode generates only the new cell.
                    if (
                        self.is_flat_layout
                        and category == "text"
                        and self._flat_filename_for_group_exists(
                            cat_subdir_leaf, save_dir,
                        )
                    ):
                        print(f"  [skip] {cat_subdir_leaf} (flat output exists)")
                        continue
                    cat_subdir = os.path.join(mode_subdir, cat_subdir_leaf)
                    producer = self._build_producer(
                        sweep_mode=sweep_mode,
                        patch_steps=patch_steps,
                        category=category,
                        target_seed=target_seed,
                        target_layout=target_layout,
                        target_h=task.height,
                        target_w=task.width,
                        text_token_indices=indices,
                    )
                    print(f"  [{category}] {log_label}")
                    sweep_and_grid(
                        self.model, source_captured, category,
                        str(save_dir), suptitle,
                        source_layout=source_layout,
                        bookend_images=bookend_images,
                        bookend_labels=bookend_labels,
                        image_producer=producer,
                        fixed_dst_name=fixed_dst_name,
                        fixed_dst_label=fixed_dst_label,
                        cat_subdir=cat_subdir,
                        block_slice=self.block_slice,
                    )

        # Clear target-side KO so subsequent tasks start clean.
        if ko_on_target:
            clear_all_masks(self.knockout_procs)

        if self.is_flat_layout:
            self._finalize_flat_layout(task, save_dir)

        print(f"\nResults: {save_dir}")
        return save_dir

    # ------------------------------------------------------------------
    # Flat-layout finalization
    # ------------------------------------------------------------------

    def _finalize_flat_layout(self, task: TaskDefinition, save_dir: Path) -> None:
        """Move each text_token_mode's patched cell out of the deep nested
        sweep dir into the flat per-task dir, then prune the now-empty nested
        tree.

        Each mode lands at a unique flat filename (full → ``patched_<sweep>.png``,
        padding-only → ``patched_<sweep>_text_padding.png``, content-only →
        ``patched_<sweep>_text_content.png``), so multiple modes coexist in the
        same task dir without colliding.
        """
        import shutil

        sweep_mode, patch_steps = self.sweep_specs[0]
        src_suffix, dst_suffix = self._flat_layout_block_suffix()

        # Map each requested mode to its (nested cat subdir, flat filename).
        for mode in self.text_token_modes:
            flat_name = self._flat_filename_for_mode(mode)
            flat_patched = save_dir / flat_name
            if flat_patched.exists():
                # Mode skipped this run because its flat output already
                # existed; nothing to move.
                continue
            if mode == "all":
                cat_subdir_leaf = "text_tokens"
            elif mode == "padding_only":
                cat_subdir_leaf = os.path.join("text", "all_padding")
            else:
                assert mode == "content_only", f"unexpected mode={mode!r}"
                cat_subdir_leaf = os.path.join("text", "all_content")
            nested_dir = save_dir / sweep_mode / "no_knockout" / cat_subdir_leaf
            nested_patched = nested_dir / f"patched_{src_suffix}_to_{dst_suffix}.png"
            assert nested_patched.exists(), (
                f"expected patched file not found: {nested_patched}"
            )
            shutil.move(str(nested_patched), str(flat_patched))

        # Drop everything under the sweep_mode subdir (grids, single-cell
        # leftovers, etc.). Pre-flat baselines (reference.png,
        # source_i2i_*step.png, t2i_clean_*step.png,
        # unconditional_baseline_*step.png) live at save_dir top-level and
        # stay put.
        sweep_root = save_dir / sweep_mode
        if sweep_root.exists():
            shutil.rmtree(sweep_root)

        # Drop the non-matching baselines for non-color settings (we only judge
        # against the matching N-step baseline). Color (single9_1step) keeps
        # its 1-step baselines because patch_steps == 1 there.
        if patch_steps != NUM_INFERENCE_STEPS:
            for noise_step in self.distinct_steps:
                if noise_step == patch_steps:
                    continue
                for stem in ("source_i2i", "t2i_clean", "unconditional_baseline"):
                    p = save_dir / f"{stem}_{noise_step}step.png"
                    if p.exists():
                        p.unlink()

        produced: list[str] = ["reference.png"]
        for mode in self.text_token_modes:
            produced.append(self._flat_filename_for_mode(mode))
        for stem in ("source_i2i", "t2i_clean", "unconditional_baseline"):
            p = save_dir / f"{stem}_{patch_steps}step.png"
            if p.exists():
                produced.append(p.name)
        self.mark_completed(task.task_id, produced)

    # ------------------------------------------------------------------
    # Sweep helpers
    # ------------------------------------------------------------------

    def _text_sweep_groups(
        self,
        instruction_prompt: str,
    ) -> list[tuple[str, list[int] | None, str]]:
        # Explicit indices override mode list entirely.
        if self.text_token_indices is not None:
            token_strings = _token_strings_for_prompt(self.model.pipe, instruction_prompt)
            return [
                (
                    os.path.join("text", f"token_{i:03d}_{_slug(token_strings[i])}"),
                    [i],
                    f"explicit index {i}={token_strings[i]!r}",
                )
                for i in self.text_token_indices
            ]
        # Concatenate per-mode group lists, deduping by output subdir leaf
        # so overlapping mode selections don't run the same group twice.
        seen: set[str] = set()
        out: list[tuple[str, list[int] | None, str]] = []
        for mode in self.text_token_modes:
            for entry in self._groups_for_mode(mode, instruction_prompt):
                if entry[0] in seen:
                    continue
                seen.add(entry[0])
                out.append(entry)
        return out

    def _groups_for_mode(
        self, mode: str, instruction_prompt: str,
    ) -> list[tuple[str, list[int] | None, str]]:
        if mode == "per_content":
            positions = resolve_content_token_indices(self.model.pipe, instruction_prompt)
            return [
                (
                    os.path.join("text", f"token_{i:03d}_{_slug(t)}"),
                    [i],
                    f"content token {i}={t!r}",
                )
                for i, t in positions
            ]
        if mode == "per_position":
            token_strings = _token_strings_for_prompt(self.model.pipe, instruction_prompt)
            lo, hi = (
                self.position_range
                if self.position_range is not None
                else (0, TEXT_SEQ_LEN - 1)
            )
            return [
                (
                    os.path.join("text", f"token_{p:03d}_{_slug(token_strings[p])}"),
                    [p],
                    f"position {p}={token_strings[p]!r}",
                )
                for p in range(lo, hi + 1)
            ]
        if mode in ("content_only", "padding_only"):
            content, padding = self._content_padding_indices(instruction_prompt)
            if mode == "content_only":
                return [(
                    os.path.join("text", "all_content"),
                    content,
                    f"all_content ({len(content)} positions)",
                )]
            return [(
                os.path.join("text", "all_padding"),
                padding,
                f"all_padding ({len(padding)} positions)",
            )]
        assert mode == "all", f"Unknown text_token_mode={mode!r}"
        return [("text_tokens", None, "full 512-token slice")]

    def _content_padding_indices(
        self, instruction_prompt: str,
    ) -> tuple[list[int], list[int]]:
        positions = resolve_content_token_indices(self.model.pipe, instruction_prompt)
        content = [i for i, _ in positions]
        cset = set(content)
        padding = [i for i in range(TEXT_SEQ_LEN) if i not in cset]
        assert len(cset) == len(content), (
            f"resolve_content_token_indices returned duplicates: {content}"
        )
        assert len(content) + len(padding) == TEXT_SEQ_LEN, (
            f"content ({len(content)}) + padding ({len(padding)}) "
            f"!= TEXT_SEQ_LEN ({TEXT_SEQ_LEN})"
        )
        return content, padding

    def _build_producer(
        self,
        *,
        sweep_mode: str,
        patch_steps: int,
        category: str,
        target_seed: int,
        target_layout: TokenLayout,
        target_h: int,
        target_w: int,
        text_token_indices: list[int] | None,
    ):
        if sweep_mode == "input_to_block0":
            assert category == "text"
            return make_input_to_block0_producer(
                self.model, target_seed=target_seed,
                target_h=target_h, target_w=target_w,
                text_token_indices=text_token_indices,
                num_inference_steps=patch_steps,
            )
        assert sweep_mode == "diagonal"
        return make_patch_pipeline_producer(
            self.model, category,
            layout=target_layout,
            target_prompt="",
            target_seed=target_seed,
            target_h=target_h, target_w=target_w,
            target_ref_image=None,
            text_token_indices=text_token_indices if category == "text" else None,
        )
