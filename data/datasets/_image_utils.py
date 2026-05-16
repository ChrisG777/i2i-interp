"""Image utilities shared across dataset extractors."""

from __future__ import annotations

from PIL import Image

PATCH = 16


def round_down_to_multiple(x: int, m: int) -> int:
    return x - (x % m)


def center_crop_to_multiple(img: Image.Image, patch: int = PATCH) -> Image.Image:
    """Center-crop to the largest WxH that is a multiple of ``patch`` on each side."""
    w, h = img.size
    new_w = round_down_to_multiple(w, patch)
    new_h = round_down_to_multiple(h, patch)
    if (new_w, new_h) == (w, h):
        return img
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))
