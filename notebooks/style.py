"""Shared figure style for benchmark plots.

Uses a ggplot2 ``theme_bw`` style with a white panel, grey border and grid,
serif font, top legend, and fixed palette.
"""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt

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


def style_axes(ax: plt.Axes) -> None:
    """Apply the per-axis theme_bw touches rcParams cannot express."""
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.5)
    ax.grid(True, which="major", color="#EBEBEB", linewidth=0.4)
    ax.grid(True, which="minor", axis="y", color="#EBEBEB", linewidth=0.2)
