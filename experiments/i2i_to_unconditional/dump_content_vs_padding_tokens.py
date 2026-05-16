"""Dump the content-vs-padding token split for i2i_to_unconditional scenes.

CPU-only, seconds to run — loads just the Qwen3 tokenizer (not the 9B
diffusion model). For each scene it resolves the content/padding indices
exactly the way ``i2i_to_unconditional_patch.py --text-token-mode
content_vs_padding`` does, then writes a per-scene txt file listing which
token positions fall into the ``all_content`` group and which into the
``all_padding`` group.

FLUX.2-klein-9B is a gated HF repo. If this is your first time pulling its
tokenizer on this machine, authenticate once via ``huggingface-cli login``
or export ``HF_TOKEN=...``. The download is only a few MB (just the tokenizer
subfolder).

Usage:
    # All five real-ref + add-object scenes (default)
    uv run python experiments/i2i_to_unconditional/dump_content_vs_padding_tokens.py

    # One or more specific scenes and a custom output dir
    uv run python experiments/i2i_to_unconditional/dump_content_vs_padding_tokens.py \\
        --name real_bedroom_tv real_desert_cactus \\
        --output-dir results/content_vs_padding_tokens
"""

import argparse
import os

from transformers import AutoTokenizer

from experiments.common.tasks import get_task
from utils.flux2_klein import MODEL_ID, TEXT_SEQ_LEN


FIVE_REAL_REF_SCENES = [
    "ocean_armchair_real",
    "real_bedroom_tv",
    "real_desert_cactus",
    "real_forest_deer",
    "real_stage_microphone",
]

DEFAULT_OUTPUT_DIR = "results/content_vs_padding_tokens"


class _StubPipe:
    """Tiny shim that exposes only ``.tokenizer`` so ``get_token_strings`` and
    ``resolve_content_token_indices`` can run without loading the transformer.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def _write_scene_report(
    *, path: str, name: str, instruction_prompt: str,
    token_strings: list[str], content: list[int], padding: list[int],
) -> None:
    with open(path, "w") as f:
        f.write(f"# Scene: {name}\n")
        f.write(f"# Instruction prompt: {instruction_prompt!r}\n")
        f.write(f"# TEXT_SEQ_LEN: {TEXT_SEQ_LEN}\n")
        f.write(f"# all_content positions: {len(content)}\n")
        f.write(f"# all_padding positions: {len(padding)}\n")
        f.write(f"# content indices (compact): {content}\n")
        f.write("\n")
        f.write("=== all_content (patched together as one group) ===\n")
        for i in content:
            f.write(f"  {i:3d}  {token_strings[i]!r}\n")
        f.write("\n")
        f.write("=== all_padding (patched together as one group) ===\n")
        for i in padding:
            f.write(f"  {i:3d}  {token_strings[i]!r}\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Dump the content-vs-padding token split used by "
            "--text-token-mode content_vs_padding, as text files."
        ),
    )
    parser.add_argument(
        "--name", nargs="+", default=FIVE_REAL_REF_SCENES,
        help="Experiment name(s) from I2I_TO_UNCONDITIONAL_EXPERIMENTS. "
             f"Default: {FIVE_REAL_REF_SCENES}",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write per-scene reports. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--model-id", default=MODEL_ID,
        help=f"HF model id to pull the tokenizer from. Default: {MODEL_ID}",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer: {args.model_id} (subfolder='tokenizer') ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    except OSError as e:
        if "gated repo" in str(e) or "401" in str(e):
            raise SystemExit(
                f"{e}\n\n"
                f"Hint: FLUX.2-klein-9B is gated. Authenticate once with "
                f"`huggingface-cli login` (or export HF_TOKEN=...) and "
                f"re-run. The tokenizer download is only a few MB."
            ) from e
        raise
    pipe = _StubPipe(tokenizer)

    # Defer imports that pull in heavy deps (torch, etc.) until after argparse.
    from experiments.patching.utils import get_token_strings
    from experiments.patching.utils import resolve_content_token_indices

    for name in args.name:
        task = get_task(name)
        instruction_prompt = task.instruction
        if not instruction_prompt.strip():
            print(f"Skipping {name!r}: empty instruction")
            continue

        token_strings = get_token_strings(pipe, instruction_prompt)
        positions = resolve_content_token_indices(pipe, instruction_prompt)
        content = [i for i, _ in positions]
        cset = set(content)
        padding = [i for i in range(TEXT_SEQ_LEN) if i not in cset]
        assert len(cset) == len(content), (
            f"resolve_content_token_indices returned duplicates for {name}: {content}"
        )
        assert len(content) + len(padding) == TEXT_SEQ_LEN, (
            f"{name}: content ({len(content)}) + padding ({len(padding)}) "
            f"!= TEXT_SEQ_LEN ({TEXT_SEQ_LEN})"
        )

        out_path = os.path.join(args.output_dir, f"{name}.txt")
        _write_scene_report(
            path=out_path, name=name, instruction_prompt=instruction_prompt,
            token_strings=token_strings, content=content, padding=padding,
        )
        content_tokens = [token_strings[i] for i in content]
        print(
            f"  {name}: {len(content)} content / {len(padding)} padding  "
            f"tokens={content_tokens}  -> {out_path}"
        )

    print(f"\nDone. Reports in {args.output_dir}/")


if __name__ == "__main__":
    main()
