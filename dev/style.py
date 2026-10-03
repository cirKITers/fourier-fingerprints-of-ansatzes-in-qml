"""Figure styling and plot helpers shared by the study scripts.

Uses a ggplot2 ``theme_bw`` style with a white panel, grey border and grid,
serif font, top legend, and fixed palette. Figures are saved as PGF and PDF.
The scatter and error-table helpers
are shared by s2 and s3.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Fixed figure palette.
COLOURS = [
    "#000000",  # black
    "#E69F00",  # orange
    "#999999",  # grey
    "#009371",  # teal
    "#beaed4",  # purple
    "#ed665a",  # salmon
    "#1f78b4",  # blue
    "#002D4C",  # navy
]

# Physical figure widths in inches; PGF text renders at ``font.size``.
COLWIDTH = 6.3
FIGWIDTH = 3.67

# theme_bw rcParams: serif fonts, white panel with a full grey border, solid
# light-grey grid, top legend, and a PGF backend for pdflatex.
PLOT_RC = {
    "font.family": "serif",
    "font.serif": ["Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "#333333",  # grey20 panel border
    "axes.linewidth": 0.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.color": "#EBEBEB",  # grey92
    "grid.linestyle": "-",
    "grid.linewidth": 0.4,
    "xtick.color": "#4D4D4D",  # grey30
    "ytick.color": "#4D4D4D",
    "legend.frameon": False,
    "lines.linewidth": 0.8,
    "lines.markersize": 3,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}",
}

# Fixed roles for the two scatter series and the legend markers.
FCC_COLOR = COLOURS[3]  # teal
EXPR_COLOR = COLOURS[1]  # orange
LEGEND_COLOR = COLOURS[7]  # navy
MARKERS = ["o", "s", "D", "P", "X", "^", "d", "*"]


def style_axes(ax: plt.Axes) -> None:
    """Apply the per-axis theme_bw touches rcParams cannot express."""
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.5)
    ax.grid(True, which="major", color="#EBEBEB", linewidth=0.4)
    ax.grid(True, which="minor", axis="y", color="#EBEBEB", linewidth=0.2)


def save_fig(fig, figures: Path, name: str) -> None:
    """Write ``name`` as ``.pgf`` and ``.pdf`` to ``figures`` and close ``fig``."""
    figures.mkdir(parents=True, exist_ok=True)
    print(f"Saving figure to {figures / name}.pgf")
    fig.savefig(figures / f"{name}.pgf")
    fig.savefig(figures / f"{name}.pdf")  # quick-look copy alongside the .pgf
    plt.close(fig)


def beautify_circuit_name(circuit):

    if circuit.lower() == "hardware_efficient":
        circuit = "HEA"
    elif circuit.lower() == "circuit_yzy_entangling":
        circuit = "Circuit YZY Ent."

    circuit = circuit.replace("_", " ")
    circuit = circuit.replace("circuit", "C")
    circuit = circuit.replace("Circuit", "C")
    return circuit


def visualize_expr_scatter(df, ansatz_ids, metric, metric_name, weighted=False):
    corr_mean = "corr_mean" if not weighted else "corr_w_mean"

    fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGWIDTH * 0.73))
    ax2 = ax.twinx()

    handles = []
    for i, ansatz_id in enumerate(ansatz_ids):
        marker = MARKERS[i % len(MARKERS)]
        _df = df[(df.ansatz_id == ansatz_id)]
        if len(_df) == 0:
            print(f"No data for ansatz_id={ansatz_id}")
            continue
        ansatz = _df["ansatz"].unique()[0]
        x = _df[metric].mean()
        # FCC on the left axis (teal), expressibility on the right axis (orange).
        ax.scatter(x, _df[corr_mean].mean(), marker=marker, color=FCC_COLOR)
        ax2.scatter(x, _df["expressibility"].mean(), marker=marker, color=EXPR_COLOR)
        handles.append(
            Line2D(
                [], [], ls="", marker=marker, color=LEGEND_COLOR, markersize=6,
                label=beautify_circuit_name(ansatz),
            )
        )

    handles.append(
        Line2D([], [], ls="", marker=(6, 2, 0), color=FCC_COLOR, markersize=6, label="FCC")
    )
    handles.append(
        Line2D(
            [], [], ls="", marker=(6, 2, 0), color=EXPR_COLOR, markersize=6,
            label="1 - Expressibility",
        )
    )

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # default ticks overlap
    ax.set_xlabel(metric_name)
    ax.set_ylabel(
        "Fourier Coefficient Corr."
        if not weighted
        else "Weight. Fourier Coefficient Corr."
    )
    ax2.set_ylabel("1 - Expressibility")
    style_axes(ax)
    ax2.grid(False)
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    return fig


def calculate_errors(df, metric, weighted=False):
    corr_mean = "corr_mean" if not weighted else "corr_w_mean"
    variables = ["expressibility", corr_mean, metric]

    means_by_seed = df.groupby(["seed", "ansatz"], as_index=False)[variables].mean()

    std_across_seeds = means_by_seed.groupby("ansatz")[
        variables
    ].std()  # ddof=1 by default (sample variance)

    result = std_across_seeds.T
    result.index = variables  # give the rows human-readable names

    print(result)
    return result


def export_pandas_table(result, figures: Path, name: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    print(f"Saving csv table to {figures / name}.csv")
    result.to_csv(figures / f"{name}.csv")

    def wrap_num(x):
        # format in scientific notation with 2 significant figures
        return f"\\num{{{x:.1e}}}"

    df_num = result.map(wrap_num)

    latex = df_num.to_latex(
        escape=False,  # values are already escaped
        index=True,
        header=True,
        column_format="l"
        + "c" * len(result.columns),  # first column left-justified, rest centered
        position="htbp",
    )

    print(f"Saving latex table to {figures / name}.tex")
    (figures / f"{name}.tex").write_text(latex)
