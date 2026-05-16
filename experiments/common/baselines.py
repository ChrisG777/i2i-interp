"""TaskDefinition → reference image.

``load_or_make_reference(model, task)`` resolves a task's reference image
from one of three sources, in priority order (matches the rules baked into
TaskDefinition's invariants):

1. ``real_ref_name`` — load ``<task.real_ref_dir>/<name>.{png,jpg,jpeg}`` via
   ``utils.model_registry.load_real_reference``. ``real_ref_dir`` is a
   required, per-task field (no hardcoded default).
2. ``source_image_path`` — load the file from disk and resize to the task's
   ``(width, height)``.
3. ``ref_seed`` — generate a synthetic reference via
   ``model.generate(source_caption, seed=ref_seed, ...)``.

For legacy entries that set both ``real_ref_name`` and ``ref_seed``, real
takes precedence — same behavior as the pre-runner ``load_or_generate_reference``.
"""

from __future__ import annotations

from PIL import Image

from experiments.common.tasks import NUM_INFERENCE_STEPS, TaskDefinition
from utils.model_registry import load_real_reference


def load_or_make_reference(model, task: TaskDefinition) -> Image.Image:
    if task.real_ref_name is not None:
        assert task.real_ref_dir is not None  # invariant
        return load_real_reference(task.real_ref_name, task.real_ref_dir)

    if task.source_image_path is not None:
        img = Image.open(task.source_image_path).convert("RGB")
        if img.size != (task.width, task.height):
            img = img.resize((task.width, task.height))
        return img

    assert task.ref_seed is not None, (
        f"task {task.task_id}: TaskDefinition invariant violated — no ref source"
    )
    assert task.source_caption is not None and task.source_caption.strip(), (
        f"task {task.task_id}: ref_seed requires a non-empty source_caption to "
        f"generate from"
    )
    print(f"  Generating reference: {task.ref_label}")
    return model.generate(
        task.source_caption,
        seed=task.ref_seed,
        num_inference_steps=NUM_INFERENCE_STEPS,
        height=task.height,
        width=task.width,
    )
