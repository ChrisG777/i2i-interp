"""Image-grid plotting helper for sweep visualizations."""

import matplotlib.pyplot as plt
import numpy as np


def create_image_grid(
    images, titles, save_path,
    n_rows=None, ncols=None,
    cell_size=None, fontsize=None,
    suptitle=None,
    highlight_indices=None,
    highlight_color="#E67E22",
    circle_indices=None,
    circle_color="#D62728",
    group_pairs=None,
):
    """Create a grid of images with titles.

    Layout control — specify at most one of:
        n_rows: number of rows (computes columns); legacy default when
                neither is given.
        ncols:  number of columns (computes rows).

    Styling defaults depend on which layout parameter is used:
        n_rows mode  → cell_size=2.5, fontsize=6   (dense grids)
        ncols  mode  → cell_size=4,   fontsize=9   (sweep grids)

    Args:
        highlight_indices: Set of cell indices to highlight with a colored
            border and bold title.
        highlight_color: Color for highlighted borders/titles.
        circle_indices: Set of cell indices to ring with a colored ellipse,
            for calling out a chosen cell.
        circle_color: Color for the rings drawn around ``circle_indices``.
        group_pairs: List of ``(idx_a, idx_b)`` tuples.  A thin outline
            rectangle is drawn around each pair of cells.
    """
    import matplotlib.patches as mpatches
    from matplotlib.transforms import Bbox

    n_images = len(images)

    if ncols is not None:
        _ncols = ncols
        _nrows = (n_images + ncols - 1) // ncols
        _cell_size = cell_size if cell_size is not None else 4
        _fontsize = fontsize if fontsize is not None else 9
    else:
        _nrows = n_rows if n_rows is not None else 4
        _ncols = (n_images + _nrows - 1) // _nrows
        _cell_size = cell_size if cell_size is not None else 2.5
        _fontsize = fontsize if fontsize is not None else 6

    fig, axes = plt.subplots(
        _nrows, _ncols,
        figsize=(_cell_size * _ncols, _cell_size * _nrows),
    )
    axes = np.atleast_1d(axes).flatten()

    for ax in axes:
        ax.axis("off")
    for idx, (ax, img, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(img)
        if highlight_indices and idx in highlight_indices:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(highlight_color)
                spine.set_linewidth(3)
            ax.set_title(
                title, fontsize=_fontsize, pad=2,
                color=highlight_color, fontweight="bold",
            )
        else:
            ax.set_title(title, fontsize=_fontsize, pad=2)
        if circle_indices and idx in circle_indices:
            ax.add_patch(mpatches.Ellipse(
                (0.5, 0.5), 1.08, 1.08,
                transform=ax.transAxes,
                fill=False, edgecolor=circle_color, linewidth=3,
                clip_on=False,
            ))
    if suptitle:
        fig.suptitle(suptitle, fontsize=_fontsize, y=1.02)

    plt.tight_layout()

    # Draw pair-grouping outlines after tight_layout so positions are final.
    if group_pairs:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for idx_a, idx_b in group_pairs:
            if idx_a >= n_images or idx_b >= n_images:
                continue
            ax_a, ax_b = axes[idx_a], axes[idx_b]
            bb_a = ax_a.get_tightbbox(renderer).transformed(
                fig.transFigure.inverted(),
            )
            bb_b = ax_b.get_tightbbox(renderer).transformed(
                fig.transFigure.inverted(),
            )
            union = Bbox.union([bb_a, bb_b])
            pad = 0.004
            rect = mpatches.FancyBboxPatch(
                (union.x0 - pad, union.y0 - pad),
                union.width + 2 * pad,
                union.height + 2 * pad,
                boxstyle="round,pad=0",
                linewidth=1.5,
                edgecolor=highlight_color,
                facecolor="none",
                transform=fig.transFigure,
                clip_on=False,
            )
            fig.patches.append(rect)

    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
