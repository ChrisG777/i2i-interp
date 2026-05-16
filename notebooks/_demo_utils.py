"""Display helpers for demo.ipynb."""
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def show_labeled(items, *, ncols=4, figsize_per=4, fontsize=20):
    """Display a labeled image grid. ``items`` is a list of ``(path, label)`` pairs."""
    items = [(Path(p), label) for p, label in items]
    nrows = (len(items) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per * ncols, figsize_per * nrows))
    axes = list(axes.flat) if nrows * ncols > 1 else [axes]
    for ax, (path, label) in zip(axes, items):
        if path.exists():
            ax.imshow(Image.open(path))
        else:
            ax.text(0.5, 0.5, f'(missing)\n{path.name}', ha='center', va='center', fontsize=fontsize - 2)
        ax.set_title(label, fontsize=fontsize)
        ax.axis('off')
    for ax in axes[len(items):]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def auto_dims(path):
    """Return (height, width) for ``path``, each snapped down to a multiple of 16."""
    w, h = Image.open(path).size
    return (h // 16) * 16, (w // 16) * 16
