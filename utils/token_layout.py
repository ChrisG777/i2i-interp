"""Model-agnostic joint-attention token layout.

Every supported model concatenates a text stream and an image stream into one
joint attention sequence ``[text | noise | ref]`` (i2i) or ``[text | noise]``
(t2i). :class:`TokenLayout` records the per-task band lengths; all slicing in
the knockout/patching stack reads token counts from a layout instance.

For VL-conditioned models (Qwen-Image-Edit) the text stream additionally
contains a contiguous band of vision tokens produced by the VL text encoder.
``vision_start``/``vision_end`` record that band (empty span for text-only
encoders like FLUX.2-klein's Qwen3 embedder).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenLayout:
    """Per-task sequence-length layout for the joint attention sequence.

    The joint stream is ``[text | noise | ref]`` (i2i) or ``[text | noise]``
    (t2i, ``ref_seq_len == 0``). ``vision_start``/``vision_end`` delimit the
    VL vision-token band *within* the text band (both 0 when the text encoder
    never sees the image).
    """

    text_seq_len: int
    noise_seq_len: int
    ref_seq_len: int  # 0 for t2i
    vision_start: int = 0  # first vision-token position within the text band
    vision_end: int = 0    # one past the last vision-token position

    def __post_init__(self) -> None:
        assert self.text_seq_len > 0, f"text_seq_len must be positive, got {self.text_seq_len}"
        assert self.noise_seq_len > 0, f"noise_seq_len must be positive, got {self.noise_seq_len}"
        assert self.ref_seq_len >= 0, f"ref_seq_len must be >= 0, got {self.ref_seq_len}"
        assert 0 <= self.vision_start <= self.vision_end <= self.text_seq_len, (
            f"vision span [{self.vision_start}, {self.vision_end}) must sit "
            f"inside the text band [0, {self.text_seq_len})"
        )

    @property
    def has_ref(self) -> bool:
        return self.ref_seq_len > 0

    @property
    def has_vision(self) -> bool:
        return self.vision_end > self.vision_start

    @property
    def vision_slice(self) -> slice:
        """Vision-token band, in joint-stream coordinates (text starts at 0)."""
        assert self.has_vision, "layout has no vision tokens"
        return slice(self.vision_start, self.vision_end)

    @property
    def prompt_slice(self) -> slice:
        """Post-vision text band (prompt tokens + closing template scaffold),
        in joint-stream coordinates. Template tokens *before* the vision band
        belong to neither subset."""
        assert self.has_vision, "layout has no vision tokens"
        assert self.vision_end < self.text_seq_len, (
            f"no prompt tokens after the vision band "
            f"(vision_end={self.vision_end} == text_seq_len={self.text_seq_len})"
        )
        return slice(self.vision_end, self.text_seq_len)

    @property
    def total_t2i(self) -> int:
        return self.text_seq_len + self.noise_seq_len

    @property
    def total_i2i(self) -> int:
        return self.text_seq_len + self.noise_seq_len + self.ref_seq_len

    @property
    def total(self) -> int:
        """Joint-stream length: includes ref tokens iff ``has_ref``."""
        return self.total_i2i if self.has_ref else self.total_t2i


def seq_len_for_image(h: int, w: int, *, patch: int) -> int:
    """Token count for an image of size ``(h, w)`` after the VAE+patchify path."""
    assert h > 0 and w > 0, f"image size must be positive, got ({h}, {w})"
    assert h % patch == 0 and w % patch == 0, (
        f"image size ({h}, {w}) must be a multiple of patch={patch}"
    )
    return (h // patch) * (w // patch)


def get_category_slices(layout: TokenLayout) -> dict[str, slice]:
    """Return slices for text, image (noise), and (for i2i) ref categories.

    Operates on the joint stream ``[text | noise | ref]``. For t2i layouts
    (``ref_seq_len == 0``) the ``"ref"`` key is omitted.
    """
    txt_end = layout.text_seq_len
    img_end = txt_end + layout.noise_seq_len
    if layout.ref_seq_len == 0:
        return {
            "text": slice(0, txt_end),
            "image": slice(txt_end, img_end),
        }
    return {
        "text": slice(0, txt_end),
        "image": slice(txt_end, img_end),
        "ref": slice(img_end, img_end + layout.ref_seq_len),
    }
