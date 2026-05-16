"""Model registry — factory for loading diffusion model wrappers by name."""

import os

from PIL import Image

from utils.model_base import DiffusionModel


MODEL_CHOICES = ("flux2_klein",)

_REAL_REF_EXTS = (".png", ".jpg", ".jpeg")


def load_model(name: str, **kwargs) -> DiffusionModel:
    """Load a diffusion model by name. Currently only ``"flux2_klein"``."""
    if name == "flux2_klein":
        from utils.flux2_klein import Flux2KleinModel
        return Flux2KleinModel(**kwargs)
    raise ValueError(
        f"Unknown model: {name!r}. Choose from {MODEL_CHOICES}."
    )


def generate_i2i(
    model: DiffusionModel,
    prompt: str,
    seed: int,
    ref_image: Image.Image,
    *,
    num_inference_steps: int = 1,
    height: int | None = None,
    width: int | None = None,
) -> Image.Image:
    """Generate an i2i image. ``height``/``width`` are forwarded when set so
    the output matches the task's intended dims rather than the model's
    1024² default."""
    print(f"  Generating i2i image: '{prompt}' (seed={seed})")
    kwargs: dict = {}
    if height is not None:
        kwargs["height"] = height
    if width is not None:
        kwargs["width"] = width
    return model.generate(
        prompt, seed=seed,
        num_inference_steps=num_inference_steps, image=ref_image,
        **kwargs,
    )


def load_real_reference(
    name: str,
    real_references_dir: str,
) -> Image.Image:
    """Load a real (non-generated) reference image by name.

    Looks for ``<real_references_dir>/<name>.{png,jpg,jpeg}`` (first hit wins).
    Returns the image at its native size — any size is supported. The Flux2
    VAE has a hard 16-pixel-multiple requirement, so dimensions that aren't
    already aligned are silently center-cropped to the nearest 16-multiple
    (≤15px lost per side; aspect ratio preserved). Color modes other than RGB
    are auto-converted.
    """
    matches = [
        os.path.join(real_references_dir, name + ext)
        for ext in _REAL_REF_EXTS
        if os.path.exists(os.path.join(real_references_dir, name + ext))
    ]
    assert matches, (
        f"Real reference {name!r} not found under {real_references_dir}. "
        f"Expected one of: {[name + ext for ext in _REAL_REF_EXTS]}"
    )
    path = matches[0]
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    new_w = (w // 16) * 16
    new_h = (h // 16) * 16
    assert new_w > 0 and new_h > 0, (
        f"Real reference {path} is too small ({w}x{h}); both dimensions must "
        f"be at least 16 pixels."
    )
    if (new_w, new_h) != (w, h):
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        img = img.crop((left, top, left + new_w, top + new_h))
        print(
            f"  Loading real reference: {path} "
            f"(center-cropped {w}x{h} -> {new_w}x{new_h} for 16-multiple alignment)"
        )
    else:
        print(f"  Loading real reference: {path}")
    return img


def load_or_generate_reference(
    model: DiffusionModel,
    prompt: str,
    seed: int,
    *,
    num_inference_steps: int = 1,
    real_ref: str | None = None,
    real_ref_dir: str | None = None,
) -> Image.Image:
    """Load a real reference photo or generate a fresh image from ``(prompt, seed)``."""
    if real_ref is not None:
        assert real_ref_dir is not None, (
            f"load_or_generate_reference: real_ref={real_ref!r} requires real_ref_dir"
        )
        return load_real_reference(real_ref, real_ref_dir)

    print(f"  Generating image: '{prompt}' (seed={seed})")
    return model.generate(
        prompt, seed=seed, num_inference_steps=num_inference_steps,
    )
