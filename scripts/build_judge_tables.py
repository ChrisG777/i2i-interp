"""Build the paper's judge tables (T2I Lens, Attention Knockout main +
appendix, I2I->I2I patching) from the VLM-judge CSVs in
``results_v4/vlm_judge/``.

Layout: tasks are columns, knockout/patching settings are rows. Tables share
the column headers ``Color Transfer / Style Transfer / Human Customization``;
T2I Lens additionally has ``Object Addition`` and ``Object Removal``.

``--model`` picks which judge model's verdicts to read (default: the paper's
Claude judge). ``--model gpt-5.6-terra`` renders the second-judge appendix
tables from ``results_v4/vlm_judge/gpt-5.6-terra/``. ``--agreement`` emits
the judge-agreement appendix table instead (per-cell percent agreement
between the paper judge and ``--model``, via
:mod:`scripts.compare_judges`).

Inversion convention (one answer key for every cell): high score => the
intervention "succeeded" at its stated goal. The judge questions are framed
inconsistently (some ask "did it LOSE?" some ask "did it KEEP?"), so we flip
exactly the "did it KEEP?" judges with ``invert=True``. See per-cell comments
below and the question text in ``results_v4/vlm_judge/README.md``. Agreement
and kappa are invariant under the flip, so the agreement table ignores it.

Error bars: 95% Wilson score interval on the pooled binary verdicts in
each cell, rendered as ${\\bar p}_{-\\delta_{lo}}^{+\\delta_{hi}}$. The
interval is bounded in [0, 100] by construction.

The LaTeX output is *only* the ``\\begin{tabularx}...\\end{tabularx}``
block per table, designed to live between ``% AUTO-TABLE`` markers
inside a user-owned ``\\begin{table}`` env that carries the caption
and label. Captions are author-written and stay outside the markers.

Outputs (stdout): an ASCII text preview followed by the tabular blocks
delimited by ``=== LATEX:<name> ===`` ... ``=== END:<name> ===`` markers
(``<name>`` gains an ``@<model>`` suffix for non-default models).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from scipy.stats import binomtest

from scripts.compare_judges import compare_judge
from scripts.judge.configs import get
from scripts.judge.csv_io import load_existing_rows
from utils.vlm import DEFAULT_MODEL, MODELS


@dataclass(frozen=True)
class Cell:
    table: str
    row: str
    col: str
    judge: str        # judge name in scripts.judge.configs
    invert: bool      # True if judge asks "did property survive?" -> flip


# fmt: off
# Inversion comment per group:
#   ko_*_ref_to_text(_padding|_content): "did i2i LOSE the property?"  -> direct
#   ko_*_ref_to_image:                   "did i2i KEEP the property?"  -> INVERT
#   i2i2i_*:                             "did target take on source's?" -> direct
#   i2i_unc_*_text_lens and i2i_unc_{add,remove}:
#                                         "did patched t2i pick it up?" -> direct
#   i2i_unc_dreambench_human_identity:   "is patched person DIFFERENT?" -> INVERT
CELLS: list[Cell] = [
    # Attention Knockout (rows = KO setting; cols = task). All 4 rows live in
    # the appendix table; the main-body table picks Ref->Text + Ref->Image only.
    Cell("attention_knockout", "Ref->Text",          "Color Transfer",      "ko_color_ref_to_text",                    False),
    Cell("attention_knockout", "Ref->Text[Padding]", "Color Transfer",      "ko_color_ref_to_text_padding",            False),
    Cell("attention_knockout", "Ref->Text[Content]", "Color Transfer",      "ko_color_ref_to_text_content",            False),
    Cell("attention_knockout", "Ref->Image",         "Color Transfer",      "ko_color_ref_to_image",                   True),
    Cell("attention_knockout", "Ref->Text",          "Style Transfer",      "ko_style_ref_to_text",                  False),
    Cell("attention_knockout", "Ref->Text[Padding]", "Style Transfer",      "ko_style_ref_to_text_padding",          False),
    Cell("attention_knockout", "Ref->Text[Content]", "Style Transfer",      "ko_style_ref_to_text_content",          False),
    Cell("attention_knockout", "Ref->Image",         "Style Transfer",      "ko_style_ref_to_image",                 True),
    Cell("attention_knockout", "Ref->Text",          "Human Customization", "ko_dreambench_human_ref_to_text",         False),
    Cell("attention_knockout", "Ref->Text[Padding]", "Human Customization", "ko_dreambench_human_ref_to_text_padding", False),
    Cell("attention_knockout", "Ref->Text[Content]", "Human Customization", "ko_dreambench_human_ref_to_text_content", False),
    Cell("attention_knockout", "Ref->Image",         "Human Customization", "ko_dreambench_human_ref_to_image",        True),

    # I2I->I2I patching (rows = which text tokens were patched; cols = task)
    Cell("i2i_to_i2i", "Text Tokens (All)",          "Color Transfer",      "i2i2i_color",                          False),
    Cell("i2i_to_i2i", "Text Tokens (Padding Only)", "Color Transfer",      "i2i2i_color_text_padding",             False),
    Cell("i2i_to_i2i", "Text Tokens (Content Only)", "Color Transfer",      "i2i2i_color_text_content",             False),
    Cell("i2i_to_i2i", "Text Tokens (All)",          "Style Transfer",      "i2i2i_style",                        False),
    Cell("i2i_to_i2i", "Text Tokens (Padding Only)", "Style Transfer",      "i2i2i_style_text_padding",           False),
    Cell("i2i_to_i2i", "Text Tokens (Content Only)", "Style Transfer",      "i2i2i_style_text_content",           False),
    Cell("i2i_to_i2i", "Text Tokens (All)",          "Human Customization", "i2i2i_dreambench_humans",              False),
    Cell("i2i_to_i2i", "Text Tokens (Padding Only)", "Human Customization", "i2i2i_dreambench_humans_text_padding", False),
    Cell("i2i_to_i2i", "Text Tokens (Content Only)", "Human Customization", "i2i2i_dreambench_humans_text_content", False),

    # T2I Lens (single row, 5 task columns).
    Cell("t2i_lens", "VLM Judge Observation Rate", "Color Transfer",      "i2i_unc_color_text_lens",           False),
    Cell("t2i_lens", "VLM Judge Observation Rate", "Style Transfer",      "i2i_unc_style_text_lens",         False),
    Cell("t2i_lens", "VLM Judge Observation Rate", "Object Addition",     "i2i_unc_add",                       False),
    Cell("t2i_lens", "VLM Judge Observation Rate", "Object Removal",      "i2i_unc_remove",                    False),
    Cell("t2i_lens", "VLM Judge Observation Rate", "Human Customization", "i2i_unc_dreambench_human_identity", True),
]

ROW_ORDER: dict[str, list[str]] = {
    "attention_knockout":      ["Ref->Text", "Ref->Image"],
    "attention_knockout_full": ["Ref->Text", "Ref->Text[Padding]", "Ref->Text[Content]", "Ref->Image"],
    "i2i_to_i2i":              ["Text Tokens (All)", "Text Tokens (Padding Only)", "Text Tokens (Content Only)"],
    "t2i_lens":                ["VLM Judge Observation Rate"],
}
COL_ORDER: dict[str, list[str]] = {
    "attention_knockout":      ["Color Transfer", "Style Transfer", "Human Customization"],
    "attention_knockout_full": ["Color Transfer", "Style Transfer", "Human Customization"],
    "i2i_to_i2i":              ["Color Transfer", "Style Transfer", "Human Customization"],
    "t2i_lens":                ["Color Transfer", "Style Transfer", "Object Addition", "Object Removal", "Human Customization"],
}
# Column headers as plain LaTeX. Tables are wrapped in `tabularx` with equal-
# width X columns, so headers wrap automatically when they overflow.
COL_HEADERS_LATEX: dict[str, str] = {
    "Color Transfer":      r"Color Transfer (\%)",
    "Style Transfer":      r"Style Transfer (\%)",
    "Object Addition":     r"Object Addition (\%)",
    "Object Removal":      r"Object Removal (\%)",
    "Human Customization": r"Human Customization (\%)",
}
# Row labels rendered as LaTeX. KO labels use \textsubscript so "KO" stays in
# upright text (not italic math); only the arrow is math-mode.
ROW_LABELS_LATEX: dict[str, str] = {
    "Ref->Text":                  r"KO\textsubscript{ref$\rightarrow$text}",
    "Ref->Text[Padding]":         r"KO\textsubscript{ref$\rightarrow$text[padding]}",
    "Ref->Text[Content]":         r"KO\textsubscript{ref$\rightarrow$text[content]}",
    "Ref->Image":                 r"KO\textsubscript{ref$\rightarrow$image}",
    "VLM Judge Observation Rate": r"\makecell[l]{VLM Judge\\Observation Rate}",
}
TABLE_TITLES = {
    "attention_knockout":      "Attention Knockout (main body)",
    "attention_knockout_full": "Attention Knockout (full, appendix)",
    "i2i_to_i2i":              "I2I->I2I patching",
    "t2i_lens":                "T2I Lens",
}
LATEX_TABLES = ("t2i_lens", "attention_knockout", "attention_knockout_full", "i2i_to_i2i")
# Agreement-table blocks: (block header, table whose cells the block lists).
AGREEMENT_BLOCKS = (
    ("T2I Lens", "t2i_lens"),
    ("Attention Knockout", "attention_knockout_full"),
    ("I2I-to-I2I Patching", "i2i_to_i2i"),
)
# fmt: on


@dataclass
class CellResult:
    score: float | None     # 0..1, None if no data
    delta_lo: float | None  # percentage points below score (>=0); None if can't compute
    delta_hi: float | None  # percentage points above score (>=0); None if can't compute
    n_judged: int
    n_errored: int
    csv_exists: bool


# ---------------------------------------------------------------------------
# CSV loading + statistics
# ---------------------------------------------------------------------------


def _wilson_ci(successes: int, n: int) -> tuple[float, float]:
    """Return (lo, hi) of the 95% Wilson score interval as fractions in [0, 1]."""
    ci = binomtest(successes, n).proportion_ci(confidence_level=0.95, method="wilson")
    return ci.low, ci.high


def evaluate(cell: Cell, model: str) -> CellResult:
    p = get(cell.judge).csv_path_for(model)
    if not p.exists():
        return CellResult(None, None, None, 0, 0, csv_exists=False)
    rows = load_existing_rows(p)
    verdicts = [int(r["pass"]) for r in rows.values() if r.get("pass", "") in ("0", "1")]
    n = len(verdicts)
    errored = len(rows) - n
    if n == 0:
        return CellResult(None, None, None, 0, errored, csv_exists=True)
    successes = sum(verdicts)
    score = successes / n
    lo, hi = _wilson_ci(successes, n)
    if cell.invert:
        score = 1 - score
        lo, hi = 1 - hi, 1 - lo
    delta_lo = (score - lo) * 100
    delta_hi = (hi - score) * 100
    return CellResult(score, delta_lo, delta_hi, n, errored, csv_exists=True)


def table_cells(table: str) -> list[Cell]:
    """The ``attention_knockout_full`` table reuses the underlying
    ``attention_knockout`` cells."""
    src = "attention_knockout" if table == "attention_knockout_full" else table
    return [c for c in CELLS if c.table == src]


def cell_grid(table: str, model: str) -> dict[tuple[str, str], CellResult]:
    return {(c.row, c.col): evaluate(c, model) for c in table_cells(table)}


# ---------------------------------------------------------------------------
# Text rendering (stdout preview)
# ---------------------------------------------------------------------------


def _fmt_text_cell(res: CellResult | None) -> str:
    if res is None or not res.csv_exists or res.score is None:
        return "    --    "
    s = f"{res.score * 100:>5.1f}%"
    if res.delta_lo is None or res.delta_hi is None:
        return f"{s:>10s}"
    return f"{s} -{res.delta_lo:>4.1f}/+{res.delta_hi:>4.1f}"


def render_text_table(table: str, model: str) -> str:
    grid = cell_grid(table, model)
    rows = ROW_ORDER[table]
    cols = COL_ORDER[table]
    row_w = max(28, max(len(r) for r in rows) + 2)
    col_w = max(14, max(len(c) for c in cols) + 2)

    out = [f"{TABLE_TITLES[table]} [{model}]"]
    out.append("".ljust(row_w) + "".join(c.ljust(col_w) for c in cols))
    for r in rows:
        line = r.ljust(row_w)
        for c in cols:
            line += _fmt_text_cell(grid.get((r, c))).ljust(col_w)
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------


def _fmt_latex_cell(res: CellResult | None) -> str:
    """Format a cell value as $\\bar{p}_{-\\delta_{lo}}^{+\\delta_{hi}}$.

    The `(\\%)` lives in the column header so the cell body carries only the
    numbers and the asymmetric Wilson half-widths.
    """
    if res is None or not res.csv_exists or res.score is None:
        return r"\textemdash"
    s = res.score * 100
    if res.delta_lo is None or res.delta_hi is None:
        return f"{s:.1f}"
    return f"${s:.1f}_{{-{res.delta_lo:.1f}}}^{{+{res.delta_hi:.1f}}}$"


def render_latex_tabular(table: str, model: str) -> str:
    """Emit a tabularx block sized to \\linewidth so headers can wrap."""
    grid = cell_grid(table, model)
    rows = ROW_ORDER[table]
    cols = COL_ORDER[table]

    col_spec = "l " + " ".join([r">{\centering\arraybackslash}X"] * len(cols))
    header_cells = [""] + [COL_HEADERS_LATEX[c] for c in cols]
    header = " & ".join(header_cells) + r" \\"
    body_lines = []
    for r in rows:
        cells = [ROW_LABELS_LATEX.get(r, r)]
        for c in cols:
            cells.append(_fmt_latex_cell(grid.get((r, c))))
        body_lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(body_lines)

    return (
        rf"\begin{{tabularx}}{{\linewidth}}{{{col_spec}}}" "\n"
        r"\toprule" "\n"
        f"{header}\n"
        r"\midrule" "\n"
        f"{body}\n"
        r"\bottomrule" "\n"
        r"\end{tabularx}"
    )


# ---------------------------------------------------------------------------
# Judge-agreement table (baseline paper judge vs --model)
# ---------------------------------------------------------------------------


def render_agreement(model: str, latex: bool) -> str:
    """Per-cell agreement between the paper judge and ``model``, grouped by
    paper table, plus a pooled overall row (micro-averaged agreement over
    all shared verdicts)."""
    lines_text: list[str] = [f"Judge agreement: {DEFAULT_MODEL} vs {model}"]
    lines_tex: list[str] = [
        r"\begin{tabularx}{\linewidth}{l l >{\centering\arraybackslash}X "
        r">{\centering\arraybackslash}X}",
        r"\toprule",
        r"Setting & Task & $n$ & Agreement (\%) \\",
        r"\midrule",
    ]
    all_pairs: list[tuple[int, int]] = []
    for block_title, table in AGREEMENT_BLOCKS:
        lines_text.append(f"\n[{block_title}]")
        lines_tex.append(rf"\multicolumn{{4}}{{l}}{{\emph{{{block_title}}}}} \\")
        cells = sorted(
            table_cells(table),
            key=lambda c: (ROW_ORDER[table].index(c.row), COL_ORDER[table].index(c.col)),
        )
        for cell in cells:
            stats = compare_judge(get(cell.judge), DEFAULT_MODEL, model)
            assert stats is not None, f"no {model} CSV for judge {cell.judge}"
            n = stats["n_shared"]
            assert n > 0, f"no shared verdicts for judge {cell.judge}"
            agree = 100 * stats["n_agree"] / n
            all_pairs.append((stats["n_agree"], n))
            setting = "" if table == "t2i_lens" else cell.row
            lines_text.append(
                f"  {setting or block_title:<28} {cell.col:<22} n={n:<5} "
                f"agree={agree:5.1f}%"
            )
            row_label = "" if table == "t2i_lens" else ROW_LABELS_LATEX.get(cell.row, cell.row)
            lines_tex.append(
                f"{row_label} & {cell.col} & {n} & {agree:.1f} \\\\"
            )
        lines_tex.append(r"\midrule")
    total_agree = sum(a for a, _ in all_pairs)
    total_n = sum(n for _, n in all_pairs)
    lines_text.append(
        f"\nOverall: agree={100 * total_agree / total_n:.1f}% "
        f"({total_agree}/{total_n} shared verdicts)"
    )
    lines_tex += [
        rf"\textbf{{Overall}} & & {total_n} & {100 * total_agree / total_n:.1f} \\",
        r"\bottomrule",
        r"\end{tabularx}",
    ]
    return "\n".join(lines_tex if latex else lines_text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", choices=sorted(MODELS), default=DEFAULT_MODEL,
        help="Judge model whose verdict CSVs feed the tables "
        "(default: the paper's Claude judge).",
    )
    ap.add_argument(
        "--agreement", action="store_true",
        help="Emit the judge-agreement table (paper judge vs --model) "
        "instead of the result tables.",
    )
    ap.add_argument("--text-only", action="store_true", help="suppress LaTeX blocks")
    ap.add_argument("--latex-only", action="store_true", help="suppress text preview")
    args = ap.parse_args()

    suffix = "" if args.model == DEFAULT_MODEL else f"@{args.model}"

    if args.agreement:
        assert args.model != DEFAULT_MODEL, "--agreement needs a non-default --model"
        if not args.latex_only:
            print(render_agreement(args.model, latex=False))
            print()
        if not args.text_only:
            print(f"=== LATEX:judge_agreement{suffix} ===")
            print(render_agreement(args.model, latex=True))
            print(f"=== END:judge_agreement{suffix} ===")
        return

    if not args.latex_only:
        for tbl in LATEX_TABLES:
            print(render_text_table(tbl, args.model))
            print()

    if not args.text_only:
        for tbl in LATEX_TABLES:
            print(f"=== LATEX:{tbl}{suffix} ===")
            print(render_latex_tabular(tbl, args.model))
            print(f"=== END:{tbl}{suffix} ===")
            print()


if __name__ == "__main__":
    main()
