"""Rewrite a generic instruction so it explicitly points at the reference image.

Without this, prompts like ``"A photo of a kitten chasing a butterfly"`` read
as generic t2i and the i2i model often ignores the reference image entirely
(its ``t2i_clean`` and ``i2i_baseline`` come out near-identical). Anchoring
the subject to the reference gives the model a reason to attend to it.

The anchor phrase is ``"in this image"`` (rather than ``"in the reference
image"``) — short, natural, and matches the phrasing used by the
hand-built solid_color and property_manual customize tasks.
"""

from __future__ import annotations

import re


def anchor_to_reference(instruction: str, subject: str, category: str) -> str:
    """Rewrite ``instruction`` so it explicitly references the reference image.

    - ``style`` category: append ``", in the style of this image"``. The subject
      *is* the style descriptor, so substituting in mid-sentence reads awkwardly.
    - everything else: replace the first ``[a|an|the] <subject>`` mention with
      ``the <subject> in this image``.
    """
    if category == "style":
        return f"{instruction.rstrip('.').rstrip()}, in the style of this image"

    pattern = re.compile(
        rf"\b(?:a |an |the )?{re.escape(subject)}\b",
        flags=re.IGNORECASE,
    )
    replacement = f"the {subject} in this image"
    rewritten, n = pattern.subn(replacement, instruction, count=1)
    assert n == 1, f"subject {subject!r} not found in instruction {instruction!r}"
    return rewritten
