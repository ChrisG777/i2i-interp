"""Per-task runner for the attention-knockout i2i sweep.

:class:`AttentionKnockoutRunner` generates a reference + i2i baseline +
t2i clean, builds the configured masks, and produces one grid per
(setting, layer_mode). With ``--all-layers-4step`` set, each grid gains
one extra trailing "4-step full KO" cell (single 4-step generation with
the mask installed on every block).

Writes per-task outputs under ``results/<exp>/<edit_type>/<task_id>/<ts>/``
via the base class's :meth:`task_dir`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from experiments.attention_knockout.knockout_processors import (
    install_knockout_processors,
    restore_processors,
)
from experiments.attention_knockout.masks import (
    COMPOSITE_KNOCKOUT_SETTINGS,
    KNOCKOUT_SETTINGS,
    LAYER_MODES,
    KnockoutSetting,
    LayerMode,
    apply_mask_to_layers,
    apply_split_mask_to_layers,
    build_knockout_mask,
    clear_all_masks,
    combine_masks,
    resolve_settings,
)
from experiments.attention_knockout.sweep import run_layer_sweep
from experiments.common.file_cache import load_or_run
from experiments.common.runner import ExperimentRunner
from experiments.common.tasks import NUM_INFERENCE_STEPS, TaskDefinition
from experiments.patching.utils import resolve_content_token_indices
from utils.flux2_klein import (
    ALL_BLOCK_LABELS,
    ALL_BLOCK_NAMES,
    Flux2KleinModel,
    layout_for,
)

NUM_BLOCKS = len(ALL_BLOCK_NAMES)


def _safe_setting_name(name: str) -> str:
    """Filesystem-safe version of a knockout setting name. ``ref->text`` becomes
    ``ref_to_text``; ``ref<->image`` becomes ``ref_bidir_image``."""
    return (
        name.replace("<->", "_bidir_")
            .replace("->", "_to_")
            .replace("[", "_")
            .replace("]", "")
            .replace("+", "_plus_")
    )

# The text-subset settings resolve content/padding indices from the task's
# instruction at runtime via the Qwen3 tokenizer (no SAM3 needed).
TEXT_PADDING_REF_TO_IMAGE = "text[padding]+ref->image"
REF_TO_TEXT_PADDING = "ref->text[padding]"
REF_TO_TEXT_CONTENT = "ref->text[content]"

TEXT_SUBSET_SETTINGS = frozenset(
    {TEXT_PADDING_REF_TO_IMAGE, REF_TO_TEXT_PADDING, REF_TO_TEXT_CONTENT}
)

ALL_KNOWN_SETTINGS: list[str] = [
    TEXT_PADDING_REF_TO_IMAGE,
    REF_TO_TEXT_PADDING,
    REF_TO_TEXT_CONTENT,
    *COMPOSITE_KNOCKOUT_SETTINGS.keys(),
    *(s.name for s in KNOCKOUT_SETTINGS),
]


# ---------------------------------------------------------------------------
# i2i runner
# ---------------------------------------------------------------------------


class AttentionKnockoutRunner(ExperimentRunner):
    name = "attention_knockout"
    results_root = "results/attention_knockout"

    def __init__(
        self,
        model: Flux2KleinModel,
        *,
        extra_args: argparse.Namespace,
    ) -> None:
        super().__init__(model, extra_args=extra_args)
        assert isinstance(model, Flux2KleinModel), (
            f"AttentionKnockoutRunner requires Flux2KleinModel, got "
            f"{type(model).__name__}"
        )
        self.selected_settings: list[str] = list(extra_args.settings)
        self.layer_modes: list[LayerMode] = list(extra_args.layer_mode)
        self.window_sizes: list[int] = list(extra_args.window_size)
        self.all_layers_4step: bool = bool(extra_args.all_layers_4step)
        self.full_ko_only: bool = bool(extra_args.full_ko_only)
        nis_override = getattr(extra_args, "num_inference_steps", None)
        self.num_inference_steps: int = (
            int(nis_override) if nis_override is not None else NUM_INFERENCE_STEPS
        )
        assert self.num_inference_steps >= 1
        # Flat layout requires --full-ko-only (per-block sweep makes no sense
        # when each judge CSV addresses one ref direction at a time) and the
        # paper-scale 4-step setting (matches the t2i/i2i baselines the judges
        # compare against).
        if self.is_flat_layout:
            assert self.full_ko_only, (
                "--results-subdir requires --full-ko-only for attention_knockout"
            )
            assert self.num_inference_steps == 4, (
                f"--results-subdir requires --num-inference-steps 4, "
                f"got {self.num_inference_steps}"
            )

        # Split-schedule mode: install a prefix mask on blocks [0, split) and a
        # suffix mask on blocks [split, NUM_BLOCKS) within a single generation,
        # for one or more cutoff blocks. ``--split-block`` resolves to
        # ``split_index`` in knockout_run.py; an empty prefix list leaves the
        # prefix blocks stock (suffix-only schedule).
        self.split_blocks: list[str] = list(getattr(extra_args, "split_block", None) or [])
        self.split_indices: list[int] = list(getattr(extra_args, "split_index", None) or [])
        self.prefix_settings: list[str] = list(getattr(extra_args, "prefix_setting", None) or [])
        self.suffix_settings: list[str] = list(getattr(extra_args, "suffix_setting", None) or [])
        self.is_split: bool = len(self.split_indices) > 0
        if self.is_split:
            assert len(self.split_indices) == len(self.split_blocks), (
                "split_index / split_block length mismatch"
            )
            assert self.suffix_settings, "split mode requires --suffix-setting"
            assert all(0 < i < NUM_BLOCKS for i in self.split_indices), (
                f"split indices {self.split_indices} must be in (0, {NUM_BLOCKS})"
            )
            assert self.full_ko_only and self.is_flat_layout, (
                "split mode requires --full-ko-only and --results-subdir"
            )
            assert self.num_inference_steps == 4, (
                "split mode requires --num-inference-steps 4"
            )

        # Install knockout processors once for the runner's lifetime. They
        # start with ``_mask=None`` on every block, so all generations behave
        # identically to the stock processors until a mask is applied.
        self.procs, self._original_procs = install_knockout_processors(model.transformer)
        self.mask_dtype: torch.dtype = next(model.transformer.parameters()).dtype

    def teardown(self) -> None:
        if self._original_procs is not None:
            restore_processors(self.model.transformer, self._original_procs)
            self._original_procs = None

    # ------------------------------------------------------------------
    # ExperimentRunner override
    # ------------------------------------------------------------------

    def run_one(self, task: TaskDefinition) -> Path:
        out_dir = self.task_dir(task)
        out_dir.mkdir(parents=True, exist_ok=True)
        instruction = task.instruction
        noise_seed = task.noise_seed
        assert noise_seed is not None, (
            f"task {task.task_id}: AttentionKnockoutRunner requires a noise_seed"
        )

        print(f"[Phase 0] Reference image for task={task.task_id}")
        ref_img = self.reference_image(task)
        if not (out_dir / "reference.png").exists():
            ref_img.save(out_dir / "reference.png")

        ref_w, ref_h = ref_img.size
        layout = layout_for(task.height, task.width, ref_h=ref_h, ref_w=ref_w)

        if self.is_split:
            metadata_extra = {
                "split_blocks": list(self.split_blocks),
                "split_indices": list(self.split_indices),
                "prefix_settings": list(self.prefix_settings),
                "suffix_settings": list(self.suffix_settings),
                "num_blocks": NUM_BLOCKS,
                "total_seq_len": layout.total,
                "num_inference_steps": self.num_inference_steps,
            }
        else:
            metadata_extra = {
                "settings": list(self.selected_settings),
                "layer_modes": list(self.layer_modes),
                "window_sizes": (
                    list(self.window_sizes) if "window" in self.layer_modes else []
                ),
                "num_blocks": NUM_BLOCKS,
                "total_seq_len": layout.total,
                "all_layers_4step": self.all_layers_4step,
                "full_ko_only": self.full_ko_only,
            }
        self.write_task_metadata(task, extra=metadata_extra)

        i2i_baseline_name = (
            f"i2i_baseline_{self.num_inference_steps}step.png"
            if self.is_flat_layout else "i2i_baseline.png"
        )
        print(f"[Phase 1] i2i baseline: '{instruction}' (noise_seed={noise_seed})")
        i2i_baseline = load_or_run(
            out_dir / i2i_baseline_name,
            generate=lambda: self.model.generate(
                instruction, seed=noise_seed,
                num_inference_steps=self.num_inference_steps, image=ref_img,
                height=task.height, width=task.width,
            ),
        )

        i2i_baseline_4step: Image.Image | None = None
        if self.all_layers_4step:
            print(f"[Phase 1] i2i baseline 4-step: '{instruction}' (noise_seed={noise_seed})")
            i2i_baseline_4step = load_or_run(
                out_dir / "i2i_baseline_4step.png",
                generate=lambda: self.model.generate(
                    instruction, seed=noise_seed,
                    num_inference_steps=4, image=ref_img,
                    height=task.height, width=task.width,
                ),
            )

        t2i_clean_name = (
            f"t2i_clean_{self.num_inference_steps}step.png"
            if self.is_flat_layout else "t2i_clean.png"
        )
        print(f"[Phase 1] t2i clean: '{instruction}' (noise_seed={noise_seed}, no ref)")
        t2i_clean = load_or_run(
            out_dir / t2i_clean_name,
            generate=lambda: self.model.generate(
                instruction, seed=noise_seed,
                num_inference_steps=self.num_inference_steps,
                height=task.height, width=task.width,
            ),
        )

        if self.is_split:
            self._run_split_cutoffs(
                task=task, layout=layout, ref_img=ref_img,
                instruction=instruction, noise_seed=noise_seed, out_dir=out_dir,
                produced_baselines=[
                    "reference.png", t2i_clean_name, i2i_baseline_name,
                ],
            )
            print(f"\nResults: {out_dir}")
            return out_dir

        text_content_mask = self._maybe_text_content_mask(
            task, instruction, layout, out_dir,
        )
        built_masks = self._build_selected_masks(
            text_content_mask=text_content_mask, layout=layout,
        )

        produced_files: list[str] = [
            "reference.png", t2i_clean_name, i2i_baseline_name,
        ]
        if self.all_layers_4step:
            produced_files.append("i2i_baseline_4step.png")
        for setting_name, mask in built_masks:
            full_ko_filename = f"{_safe_setting_name(setting_name)}_full_ko.png"
            if self.is_flat_layout and (out_dir / full_ko_filename).exists():
                # Skip: this setting's expected flat-layout output is already
                # on disk. Re-running with a new setting added produces only
                # the new setting's file; previously-completed settings are
                # left untouched (no GPU spent regenerating them).
                print(f"\n[skip setting] {setting_name} ({full_ko_filename} exists)")
                produced_files.append(full_ko_filename)
                continue
            self._run_setting_sweep(
                setting_name=setting_name, mask=mask, task=task, layout=layout,
                ref_img=ref_img, i2i_baseline=i2i_baseline,
                i2i_baseline_4step=i2i_baseline_4step, t2i_clean=t2i_clean,
                instruction=instruction, noise_seed=noise_seed, out_dir=out_dir,
            )
            if self.is_flat_layout:
                produced_files.append(full_ko_filename)

        if self.is_flat_layout:
            self.mark_completed(task.task_id, produced_files)

        print(f"\nResults: {out_dir}")
        return out_dir

    def expected_artifacts(self, task: TaskDefinition) -> list[str]:
        """Files the current invocation will produce under ``task_dir(task)``.
        Used by the base class's artifact-aware ``--skip-if-completed`` to
        skip a task iff every file already exists. Only meaningful in flat
        layout; nested layout returns ``[]`` (no skip)."""
        if not self.is_flat_layout:
            return []
        files: list[str] = ["reference.png"]
        files.append(f"i2i_baseline_{self.num_inference_steps}step.png")
        files.append(f"t2i_clean_{self.num_inference_steps}step.png")
        if self.is_split:
            files += [
                self._split_full_ko_filename(i) for i in self.split_indices
            ]
            return files
        if self.all_layers_4step:
            files.append("i2i_baseline_4step.png")
        for setting_name in self.selected_settings:
            files.append(f"{_safe_setting_name(setting_name)}_full_ko.png")
        return files

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------

    def _maybe_text_content_mask(
        self,
        task: TaskDefinition,
        instruction: str,
        layout,
        out_dir: Path,
    ) -> torch.Tensor | None:
        if not (TEXT_SUBSET_SETTINGS & set(self.selected_settings)):
            return None
        positions = resolve_content_token_indices(self.model.pipe, instruction)
        content_indices = [i for i, _ in positions]
        text_content_mask = torch.zeros(layout.text_seq_len, dtype=torch.bool)
        text_content_mask[content_indices] = True
        padding_count = int((~text_content_mask).sum())
        with open(out_dir / "text_padding_indices.json", "w") as f:
            json.dump(
                {
                    "instruction": instruction,
                    "text_seq_len": layout.text_seq_len,
                    "content_indices": content_indices,
                    "content_tokens": [s for _, s in positions],
                    "padding_count": padding_count,
                },
                f, indent=2,
            )
        print(
            f"[text content mask] {task.task_id}: "
            f"{len(content_indices)} content / {padding_count} padding "
            f"text positions"
        )
        return text_content_mask

    @staticmethod
    def _directives_for(name: str) -> list[KnockoutSetting]:
        """Map a setting name to the list of atomic directives that build it.

        Composites (e.g. ``image<->ref``) expand to their stored tuple. The
        text-subset settings are single-directive — sources and destination
        carry the ``[content]``/``[padding]`` qualifier directly.
        """
        if name in COMPOSITE_KNOCKOUT_SETTINGS:
            return list(COMPOSITE_KNOCKOUT_SETTINGS[name])
        if name == TEXT_PADDING_REF_TO_IMAGE:
            return [KnockoutSetting(("text[padding]", "ref"), "image")]
        if name == REF_TO_TEXT_PADDING:
            return [KnockoutSetting(("ref",), "text[padding]")]
        if name == REF_TO_TEXT_CONTENT:
            return [KnockoutSetting(("ref",), "text[content]")]
        return [resolve_settings([name])[0]]

    @classmethod
    def _build_mask_for_settings(
        cls,
        names: list[str],
        *,
        layout,
        device: torch.device | str,
        dtype: torch.dtype,
        text_content_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """OR-union the masks of every directive of every setting in ``names``.

        Each name expands to one or more atomic ``KnockoutSetting`` directives
        via :meth:`_directives_for`; all directives across all names are built
        with :func:`build_knockout_mask` and combined via :func:`combine_masks`.
        ``device``/``dtype`` are explicit (not read from ``self``) so the mask
        construction is unit-testable without loading the model.
        """
        directives = [d for name in names for d in cls._directives_for(name)]
        assert directives, "Need at least one setting name"
        masks = [
            build_knockout_mask(
                layout, d, device=device, dtype=dtype,
                text_content_mask=text_content_mask,
            )
            for d in directives
        ]
        return masks[0] if len(masks) == 1 else combine_masks(*masks)

    def _build_selected_masks(
        self,
        *,
        text_content_mask: torch.Tensor | None,
        layout,
    ) -> list[tuple[str, torch.Tensor]]:
        return [
            (
                name,
                self._build_mask_for_settings(
                    [name], layout=layout,
                    device=self.model.device, dtype=self.mask_dtype,
                    text_content_mask=text_content_mask,
                ),
            )
            for name in self.selected_settings
        ]

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------

    def _run_setting_sweep(
        self,
        *,
        setting_name: str,
        mask: torch.Tensor,
        task: TaskDefinition,
        layout,
        ref_img: Image.Image,
        i2i_baseline: Image.Image,
        i2i_baseline_4step: Image.Image | None,
        t2i_clean: Image.Image,
        instruction: str,
        noise_seed: int,
        out_dir: Path,
    ) -> None:
        print(f"\n[Setting] {setting_name}")
        if self.is_flat_layout:
            # Flat layout: skip the per-setting subdir; the file name carries
            # the setting (e.g. ref_to_text_full_ko.png) and lives directly
            # under <task_dir>/.
            setting_dir = out_dir
            full_ko_filename = f"{_safe_setting_name(setting_name)}_full_ko.png"
        else:
            setting_dir = out_dir / setting_name
            setting_dir.mkdir(parents=True, exist_ok=True)
            full_ko_filename = "full_ko.png"

        # Full-KO i2i once per (task, setting): mask on every block, default
        # 1-step generation. Prepended to every mode's grid so the asymptote
        # is always visible.
        print(f"  [full KO] i2i with mask on all {NUM_BLOCKS} blocks")
        apply_mask_to_layers(self.procs, "suffix", 0, ALL_BLOCK_NAMES, mask)
        full_ko_img = self._emit_ko_image(
            task=task, ref_img=ref_img, instruction=instruction,
            noise_seed=noise_seed, num_inference_steps=self.num_inference_steps,
            out_path=setting_dir / full_ko_filename,
        )

        # Optional 4-step full-KO cell. Same mask on every block, but with
        # num_inference_steps=4 so the diffusion converges further. The
        # resulting image becomes a single appended cell on every grid for
        # this (task, setting) — not a row, just one extra cell.
        full_ko_4step_img: Image.Image | None = None
        if self.all_layers_4step:
            print(f"  [full KO 4-step] i2i with mask on all blocks, 4 inference steps")
            apply_mask_to_layers(self.procs, "suffix", 0, ALL_BLOCK_NAMES, mask)
            full_ko_4step_img = self._emit_ko_image(
                task=task, ref_img=ref_img, instruction=instruction,
                noise_seed=noise_seed, num_inference_steps=4,
                out_path=setting_dir / "full_ko_4step.png",
            )

        # ``--full-ko-only`` short-circuits the per-block grid sweep: the
        # full-KO image (and optional 4-step variant) is what we want, and
        # the per-layer sweep is the bulk of GPU time. Skip it.
        if self.full_ko_only:
            return

        for mode in self.layer_modes:
            mode_runs: list[tuple[str, int | None]] = (
                [("window/k=" + str(k), k) for k in self.window_sizes]
                if mode == "window"
                else [(mode, None)]
            )
            for rel_subdir, k in mode_runs:
                mode_dir = setting_dir / rel_subdir
                tag = f"{mode} k={k}" if mode == "window" else mode
                print(f"  [mode] {tag}")

                if mode == "window":
                    assert k is not None and 1 <= k <= NUM_BLOCKS, (
                        f"window_size={k} out of range for num_blocks={NUM_BLOCKS}"
                    )
                    L_range = range(NUM_BLOCKS - k + 1)
                else:
                    L_range = range(NUM_BLOCKS)

                append_images: list[Image.Image] = []
                append_labels: list[str] = []
                if full_ko_4step_img is not None:
                    append_images.append(full_ko_4step_img)
                    append_labels.append("4-step full KO")

                run_layer_sweep(
                    procs=self.procs,
                    mask=mask,
                    mode=mode,
                    L_range=L_range,
                    ordered_block_names=ALL_BLOCK_NAMES,
                    block_labels=ALL_BLOCK_LABELS,
                    window_size=k,
                    generate_fn=lambda: self.model.generate(
                        instruction, seed=noise_seed,
                        num_inference_steps=self.num_inference_steps, image=ref_img,
                        height=task.height, width=task.width,
                    ),
                    prepend_images=[
                        ref_img, i2i_baseline,
                        *([i2i_baseline_4step] if i2i_baseline_4step is not None else []),
                        t2i_clean, full_ko_img,
                    ],
                    prepend_labels=[
                        "Reference", "i2i baseline",
                        *(["i2i baseline 4-step"] if i2i_baseline_4step is not None else []),
                        "t2i clean",
                        f"i2i full KO [{setting_name}]",
                    ],
                    append_images=append_images,
                    append_labels=append_labels,
                    out_dir=str(mode_dir),
                    suptitle=(
                        f"Attention knockout {setting_name} [{tag}]: "
                        f"task={task.task_id} | '{instruction}'\n"
                        f"noise_seed={noise_seed}"
                    ),
                )

    # ------------------------------------------------------------------
    # Split schedule
    # ------------------------------------------------------------------

    def _emit_ko_image(
        self,
        *,
        task: TaskDefinition,
        ref_img: Image.Image,
        instruction: str,
        noise_seed: int,
        num_inference_steps: int,
        out_path: Path,
    ) -> Image.Image:
        """Generate one i2i image with whatever mask(s) are currently installed
        on ``self.procs``, then clear the masks, free the cache, and save.

        The caller installs the mask(s) first — a single mask via
        :func:`apply_mask_to_layers` (full-KO sweep) or a prefix/suffix pair via
        :func:`apply_split_mask_to_layers` (split schedule).
        """
        img = self.model.generate(
            instruction, seed=noise_seed,
            num_inference_steps=num_inference_steps, image=ref_img,
            height=task.height, width=task.width,
        )
        clear_all_masks(self.procs)
        torch.cuda.empty_cache()
        img.save(out_path)
        return img

    @staticmethod
    def _split_full_ko_filename(split_index: int) -> str:
        """Flat-layout output name for a split-schedule cutoff at ``split_index``
        — the first suffix block, e.g.
        ``split_at_single_transformer_blocks.2_full_ko.png``."""
        return f"split_at_{ALL_BLOCK_NAMES[split_index]}_full_ko.png"

    def _run_split_cutoffs(
        self,
        *,
        task: TaskDefinition,
        layout,
        ref_img: Image.Image,
        instruction: str,
        noise_seed: int,
        out_dir: Path,
        produced_baselines: list[str],
    ) -> None:
        """Split-schedule Phase 2: install the prefix mask on blocks
        ``[0, cutoff)`` and the suffix mask on ``[cutoff, NUM_BLOCKS)``, one
        full-KO generation per cutoff in ``self.split_indices``.

        Mirrors :meth:`_run_setting_sweep` as the per-task Phase-2 helper, but
        for a two-mask schedule instead of a single-mask layer sweep — the
        reference, baselines, and metadata are produced once by :meth:`run_one`.
        """
        suffix_mask = self._build_mask_for_settings(
            self.suffix_settings, layout=layout,
            device=self.model.device, dtype=self.mask_dtype,
        )
        prefix_mask = (
            self._build_mask_for_settings(
                self.prefix_settings, layout=layout,
                device=self.model.device, dtype=self.mask_dtype,
            )
            if self.prefix_settings
            else None
        )
        produced_files = list(produced_baselines)
        for split_index in self.split_indices:
            fname = self._split_full_ko_filename(split_index)
            cutoff_block = ALL_BLOCK_NAMES[split_index]
            if self.is_flat_layout and (out_dir / fname).exists():
                # Skip: this cutoff's output is already on disk. Re-running
                # with a new cutoff added produces only the new file.
                print(f"\n[skip cutoff] {cutoff_block} ({fname} exists)")
                produced_files.append(fname)
                continue
            print(
                f"\n[split] cutoff={cutoff_block} "
                f"prefix[0,{split_index})={self.prefix_settings or 'stock'} | "
                f"suffix[{split_index},{NUM_BLOCKS})={self.suffix_settings}"
            )
            apply_split_mask_to_layers(
                self.procs, split_index, ALL_BLOCK_NAMES, prefix_mask, suffix_mask,
            )
            self._emit_ko_image(
                task=task, ref_img=ref_img, instruction=instruction,
                noise_seed=noise_seed, num_inference_steps=self.num_inference_steps,
                out_path=out_dir / fname,
            )
            produced_files.append(fname)
        self.mark_completed(task.task_id, produced_files)
