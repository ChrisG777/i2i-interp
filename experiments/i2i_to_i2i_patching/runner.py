"""Runner for i2i-to-i2i text-token activation patching.

Takes a (source_task, target_task) pair and asserts compatibility:

* same ``instruction`` (override with ``--allow-mismatched-instruction``),
* both have a ``noise_seed`` set (the values may differ — paper-scale pair
  builders deliberately use *different* source/target seeds via cyclic shift),
* different ``ref`` (real_ref_name / source_image_path / ref_seed) — always enforced.

Captures activations from the source i2i run and patches the **text-token**
slice into the target i2i run across the chosen blocks. Image and ref token
patching are intentionally not exposed — text-only is the only mode.

Per-pair output:
``results/i2i_to_i2i_patching/<source_task_id>__<target_task_id>/<run_timestamp>/``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from experiments.common.file_cache import load_or_run
from experiments.common.runner import ExperimentRunner
from experiments.common.tasks import NUM_INFERENCE_STEPS, TaskDefinition
from experiments.patching.sweep import (
    make_patch_pipeline_producer,
    make_patch_pipeline_producer_multi_step,
    sweep_and_grid,
)
from experiments.patching.utils import resolve_content_token_indices
from utils.flux2_klein import ALL_BLOCK_NAMES, Flux2KleinModel, TEXT_SEQ_LEN, layout_for
from utils.model_registry import generate_i2i, load_or_generate_reference, load_real_reference


# Maps each --text-token-mode value to (cat_subdir for sweep_and_grid,
# flat-layout patched filename). A mode runs only when its flat filename is
# missing on disk, so adding a new mode to a previous run only generates the
# new variant.
_MODE_SUBDIR: dict[str, str] = {
    "all": "text_tokens",
    "padding_only": "text_padding",
    "content_only": "text_content",
}
_MODE_FLAT_FILENAME: dict[str, str] = {
    "all": "patched.png",
    "padding_only": "patched_text_padding.png",
    "content_only": "patched_text_content.png",
}


def _ref_key(task: TaskDefinition) -> tuple:
    """Identifier tuple for this task's ref source. Different tuples = different refs."""
    return (task.real_ref_name, task.source_image_path, task.ref_seed, task.source_caption)


def _resolve_ref(model, task: TaskDefinition) -> Image.Image:
    """Load this task's reference image into a PIL.Image."""
    if task.real_ref_name is not None:
        assert task.real_ref_dir is not None  # invariant
        return load_real_reference(task.real_ref_name, task.real_ref_dir)
    if task.source_image_path is not None:
        return Image.open(task.source_image_path).convert("RGB")
    assert task.ref_seed is not None and task.source_caption is not None, (
        f"task {task.task_id}: no ref source"
    )
    return load_or_generate_reference(model, task.source_caption, task.ref_seed)


class I2IToI2IRunner(ExperimentRunner):
    name = "i2i_to_i2i_patching"
    results_root = "results/i2i_to_i2i_patching"

    def __init__(self, model: Flux2KleinModel, *, extra_args: argparse.Namespace) -> None:
        super().__init__(model, extra_args=extra_args)
        assert isinstance(model, Flux2KleinModel)
        self.block_slice: slice | None = (
            slice(extra_args.block_range[0], extra_args.block_range[1] + 1)
            if extra_args.block_range is not None
            else None
        )
        self.num_inference_steps: int = int(
            getattr(extra_args, "num_inference_steps", NUM_INFERENCE_STEPS)
        )
        assert self.num_inference_steps >= 1, (
            f"--num-inference-steps must be >= 1, got {self.num_inference_steps}"
        )
        if self.is_flat_layout:
            assert self.num_inference_steps == 4, (
                f"--results-subdir requires --num-inference-steps 4, "
                f"got {self.num_inference_steps}"
            )
        modes = list(getattr(extra_args, "text_token_mode", ["all"]))
        # Dedup while preserving order; assert known names.
        seen: set[str] = set()
        self.text_token_modes: list[str] = []
        for m in modes:
            assert m in _MODE_SUBDIR, (
                f"Unknown text_token_mode {m!r}. Valid: {list(_MODE_SUBDIR)}"
            )
            if m not in seen:
                seen.add(m)
                self.text_token_modes.append(m)

    def run_one(self, task: TaskDefinition) -> Path:
        raise NotImplementedError(
            "i2i-to-i2i is a pair-based experiment; use run_pair(source, target)"
        )

    def pair_dir(self, source: TaskDefinition, target: TaskDefinition) -> Path:
        # Use the inherited setting_root() (which respects --results-subdir)
        # so flat-layout pairs land at <root>/<subdir>/<src>__<tgt>/.
        out = self.setting_root() / f"{source.task_id}__{target.task_id}"
        if not self._no_timestamp and not self.is_flat_layout:
            out = out / self.run_timestamp
        return out

    def _pair_id(self, source: TaskDefinition, target: TaskDefinition) -> str:
        return f"{source.task_id}__{target.task_id}"

    def _expected_pair_artifacts(
        self, source: TaskDefinition, target: TaskDefinition,
    ) -> list[str]:
        """Files (relative to ``pair_dir``) that the current invocation
        will produce. Used by ``--skip-if-completed`` to skip a pair iff
        every file is already on disk."""
        n = self.num_inference_steps
        files = [
            "ref_source.png",
            "ref_target.png",
            f"source_i2i_{n}step.png" if self.is_flat_layout else "source.png",
            f"target_baseline_{n}step.png" if self.is_flat_layout else "target_baseline.png",
            "target_t2i_clean.png",
        ]
        files.extend(_MODE_FLAT_FILENAME[m] for m in self.text_token_modes)
        return files

    def run_pairs(self, pairs: list[tuple[TaskDefinition, TaskDefinition]]) -> list[Path]:
        assert len(pairs) > 0, "run_pairs: empty pair list"
        out_dirs: list[Path] = []
        for i, (source, target) in enumerate(pairs):
            pair_id = self._pair_id(source, target)
            expected = self._expected_pair_artifacts(source, target)
            if self._skip_if_completed and self.is_completed(pair_id, expected):
                print(f"\n[{i + 1}/{len(pairs)}] skip (completed): {pair_id}")
                continue
            print(
                f"\n{'=' * 60}\n[{i + 1}/{len(pairs)}] "
                f"pair=({source.task_id}, {target.task_id})\n{'=' * 60}"
            )
            out_dirs.append(self.run_pair(source, target))
            torch.cuda.empty_cache()
        return out_dirs

    def run_pair(self, source: TaskDefinition, target: TaskDefinition) -> Path:
        allow_instr = bool(getattr(self.extra_args, "allow_mismatched_instruction", False))
        if not allow_instr:
            assert source.instruction == target.instruction, (
                f"i2i-to-i2i pair requires same instruction; got "
                f"source={source.instruction!r} target={target.instruction!r}. "
                f"Pass --allow-mismatched-instruction to override."
            )
        assert _ref_key(source) != _ref_key(target), (
            f"i2i-to-i2i pair requires different refs; both are {_ref_key(source)}"
        )
        assert source.noise_seed is not None and target.noise_seed is not None, (
            f"pair ({source.task_id}, {target.task_id}): "
            f"noise_seed required on both source and target"
        )
        assert (source.height, source.width) == (target.height, target.width), (
            f"pair ({source.task_id}, {target.task_id}): source and target must "
            f"share dims for activation patching to be shape-compatible; got "
            f"source=({source.height}, {source.width}) "
            f"target=({target.height}, {target.width})"
        )

        save_dir = self.pair_dir(source, target)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Resolve refs (cheap — file open) and save once if not already there.
        ref_source = _resolve_ref(self.model, source)
        ref_target = _resolve_ref(self.model, target)
        if not (save_dir / "ref_source.png").exists():
            ref_source.save(save_dir / "ref_source.png")
        if not (save_dir / "ref_target.png").exists():
            ref_target.save(save_dir / "ref_target.png")

        # Per-mode skip: only run modes whose flat output is missing.
        modes_to_run: list[str] = []
        if self.is_flat_layout:
            for m in self.text_token_modes:
                fname = _MODE_FLAT_FILENAME[m]
                if (save_dir / fname).exists():
                    print(f"[skip mode] {m} ({fname} already exists)")
                else:
                    modes_to_run.append(m)
        else:
            modes_to_run = list(self.text_token_modes)
        if not modes_to_run:
            print(f"\nAll requested modes already complete: {save_dir}")
            return save_dir

        # Flat layout requires a single block in --block-range so there's
        # exactly one patched cell to extract at the end.
        if self.is_flat_layout:
            assert (
                self.extra_args.block_range is not None
                and self.extra_args.block_range[0] == self.extra_args.block_range[1]
            ), "--results-subdir requires a single-block --block-range"

        # Per-pair metadata (sibling of cli_args.json).
        meta_path = save_dir / "task_metadata.json"
        import json
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "experiment": self.name,
                    "model": self.model.name,
                    "run_timestamp": self.run_timestamp,
                    "git_sha": self.git_sha,
                    "source_task_id": source.task_id,
                    "target_task_id": target.task_id,
                    "source_instruction": source.instruction,
                    "target_instruction": target.instruction,
                    "source_noise_seed": source.noise_seed,
                    "target_noise_seed": target.noise_seed,
                    "source_ref_label": source.ref_label,
                    "target_ref_label": target.ref_label,
                    "block_range": (
                        list(self.extra_args.block_range)
                        if self.extra_args.block_range is not None
                        else None
                    ),
                    "num_inference_steps": self.num_inference_steps,
                },
                f, indent=2, default=str,
            )
        cli_path = save_dir / "cli_args.json"
        with open(cli_path, "w") as f:
            json.dump(vars(self.extra_args), f, indent=2, default=str)

        # Source i2i with activation capture across all 32 blocks. When
        # num_inference_steps > 1, capture stores one tensor per layer per
        # step (used by the multi-step patch hook to swap source acts per
        # target step). Capture is unconditional because the activations are
        # required by every mode's sweep, and we have no on-disk cache for
        # them; the image save below is idempotent.
        n_steps = self.num_inference_steps
        print(f"\n[Phase 1] Source i2i: '{source.instruction}' "
              f"(noise_seed={source.noise_seed}, steps={n_steps})")
        source_img, source_captured = self.model.capture_activations(
            source.instruction,
            source.noise_seed,
            list(ALL_BLOCK_NAMES),
            num_inference_steps=n_steps,
            height=source.height,
            width=source.width,
            image=ref_source,
            captures_to_cpu=True,
        )
        source_img_name = (
            f"source_i2i_{n_steps}step.png" if self.is_flat_layout else "source.png"
        )
        if not (save_dir / source_img_name).exists():
            source_img.save(save_dir / source_img_name)

        # Target i2i baseline (load if cached on disk; otherwise generate + save).
        target_img_name = (
            f"target_baseline_{n_steps}step.png"
            if self.is_flat_layout else "target_baseline.png"
        )
        print(f"[Phase 2] Target i2i baseline: '{target.instruction}' "
              f"(noise_seed={target.noise_seed}, steps={n_steps})")
        target_img = load_or_run(
            save_dir / target_img_name,
            generate=lambda: generate_i2i(
                self.model, target.instruction, target.noise_seed, ref_target,
                num_inference_steps=n_steps,
                height=target.height, width=target.width,
            ),
        )

        # Target t2i clean (no ref).
        print(f"[Phase 2b] Target t2i clean "
              f"(noise_seed={target.noise_seed}, steps={n_steps}, no ref)")
        target_t2i_clean = load_or_run(
            save_dir / "target_t2i_clean.png",
            generate=lambda: self.model.generate(
                target.instruction, seed=target.noise_seed,
                num_inference_steps=n_steps,
                height=target.height, width=target.width,
            ),
        )

        src_w, src_h = ref_source.size
        tgt_w, tgt_h = ref_target.size
        source_layout = layout_for(source.height, source.width, ref_h=src_h, ref_w=src_w)
        target_layout = layout_for(target.height, target.width, ref_h=tgt_h, ref_w=tgt_w)

        if source.instruction == target.instruction:
            instr_line = f"'{source.instruction}'"
        else:
            instr_line = f"src: '{source.instruction}'\ntgt: '{target.instruction}'"
        if source.noise_seed == target.noise_seed:
            seed_str = f"noise_seed={source.noise_seed}"
        else:
            seed_str = f"noise_seed src={source.noise_seed}, tgt={target.noise_seed}"
        suptitle = (
            f"i2i->i2i text-token patching | {instr_line}\n"
            f"source ref: {source.ref_label} | target ref: {target.ref_label} "
            f"| {seed_str}"
        )

        # Run each requested text-token mode in turn. Each mode writes into
        # its own ``cat_subdir`` so they coexist; with --skip-if-completed,
        # only the modes whose flat output is missing reach this loop.
        for mode in modes_to_run:
            text_token_indices = self._token_indices_for_mode(mode, target.instruction)
            cat_subdir = _MODE_SUBDIR[mode]
            print(
                f"\n[Phase 3:{mode}] Sweeping text tokens "
                f"(steps={n_steps}, subdir={cat_subdir})"
            )
            if n_steps == 1:
                producer = make_patch_pipeline_producer(
                    self.model, "text",
                    layout=target_layout,
                    target_prompt=target.instruction,
                    target_seed=target.noise_seed,
                    target_h=target.height, target_w=target.width,
                    target_ref_image=ref_target,
                    text_token_indices=text_token_indices,
                )
            else:
                producer = make_patch_pipeline_producer_multi_step(
                    self.model, "text",
                    layout=target_layout,
                    target_prompt=target.instruction,
                    target_seed=target.noise_seed,
                    target_h=target.height, target_w=target.width,
                    target_ref_image=ref_target,
                    text_token_indices=text_token_indices,
                    num_inference_steps=n_steps,
                )
            sweep_and_grid(
                self.model, source_captured, "text", str(save_dir), suptitle,
                source_layout=source_layout,
                bookend_images=[source_img, target_img, target_t2i_clean],
                bookend_labels=["Source", "Target (baseline)", "Target t2i clean"],
                image_producer=producer,
                cat_subdir=cat_subdir,
                block_slice=self.block_slice,
                num_inference_steps=n_steps,
            )

        if self.is_flat_layout:
            self._finalize_flat_layout(
                source, target, save_dir,
                source_img_name=source_img_name,
                target_img_name=target_img_name,
                modes_run=modes_to_run,
            )

        print(f"\nResults: {save_dir}")
        return save_dir

    def _token_indices_for_mode(
        self, mode: str, instruction_prompt: str,
    ) -> list[int] | None:
        """Resolve the text-token indices to patch for ``mode``. ``None`` means
        the full 512-token slice (mode == 'all'). For 'padding_only' we return
        the complement of the Qwen3 content positions; for 'content_only' we
        return the content positions themselves."""
        if mode == "all":
            return None
        assert mode in ("padding_only", "content_only"), (
            f"Unknown text_token_mode={mode!r}"
        )
        positions = resolve_content_token_indices(self.model.pipe, instruction_prompt)
        content = [i for i, _ in positions]
        if mode == "content_only":
            return content
        cset = set(content)
        return [i for i in range(TEXT_SEQ_LEN) if i not in cset]

    # ------------------------------------------------------------------
    # Flat-layout finalization
    # ------------------------------------------------------------------

    def _finalize_flat_layout(
        self,
        source: TaskDefinition,
        target: TaskDefinition,
        save_dir: Path,
        *,
        source_img_name: str,
        target_img_name: str,
        modes_run: list[str],
    ) -> None:
        """Move each mode's single patched PNG out of its nested ``cat_subdir``
        into the flat per-pair dir under the mode's flat filename (e.g.
        ``patched.png`` for ``all``, ``patched_text_padding.png`` for
        ``padding_only``, ``patched_text_content.png`` for ``content_only``),
        then prune the empty nested tree. Init-time asserts have already
        enforced single-block --block-range, so each mode has exactly one
        patched cell to extract."""
        import shutil

        block_idx = self.extra_args.block_range[0]
        block_name = ALL_BLOCK_NAMES[block_idx]
        suffix = block_name.replace(
            "transformer_blocks.", "mm",
        ).replace("single_transformer_blocks.", "single")

        produced = [
            "ref_source.png", "ref_target.png",
            source_img_name, target_img_name,
            "target_t2i_clean.png",
        ]
        for mode in modes_run:
            nested_dir = save_dir / _MODE_SUBDIR[mode]
            nested_patched = nested_dir / f"patched_{suffix}_to_{suffix}.png"
            assert nested_patched.exists(), (
                f"expected patched file not found: {nested_patched}"
            )
            flat_name = _MODE_FLAT_FILENAME[mode]
            shutil.move(str(nested_patched), str(save_dir / flat_name))
            if nested_dir.exists():
                shutil.rmtree(nested_dir)
            produced.append(flat_name)

        self.mark_completed(self._pair_id(source, target), produced)
