"""Fingerprint heatmaps of the Fourier coefficient correlations (s1).

Reads CSVs only, from ``results/`` or, with ``--reference``, from
``reference/`` (bundled CSVs), and writes
``.pgf`` and ``.pdf`` to ``figures/``:

    uv run --no-project --with matplotlib --with pandas --with numpy \\
        python dev/s1-fingerprints/figures.py --reference

Figures, one per qubit count ``q``:

- ``<scenario>_hm_q<q>``: one panel per ansatz on a shared colour scale
  (1dfs_rx_mse_uw, 1dfs_rx_mse_w, 1dfs_ry_mse_uw, 1dfs_ry_mse_w)
- ``<scenario>_hms_q<q>``: single panel (2dfs_mse_uw, 2dfs_mse_w and the
  random-coefficient surrogate 1dfs_rc)

CSV schema: ``heatmaps/<scenario>.csv`` with one row per matrix entry of the
correlation matrix of seed 1000, i.e. per (qubits, ansatz, yi, xi):

- qubits: number of qubits
- ansatz: circuit name, e.g. Hardware_Efficient; panels follow first appearance
- yi, xi: integer row and column position in the matrix
- y, x: coefficient labels of that row and column, ``c_+k`` (1D) or ``c_+k_+l`` (2D)
- z: absolute correlation $|r|$ of the two coefficients, empty where masked
  (upper triangle)
- zmax: colour-scale ceiling of the panel, $\\max |r|$ rounded up at its first
  significant digit (1.0 for the surrogate)

Scenario suffixes: ``_uw`` holds the unweighted, ``_w`` the weighted
correlation; ``rx``/``ry`` the RX/RY encoding. The 1dfs files hold all eight
ansatzes; single-panel files hold one ansatz (Hardware_Efficient for 2dfs).
File names match the heatmap CSV scenarios.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

STUDY = Path(__file__).resolve().parent
FIGURES = STUDY / "figures"

sys.path.insert(0, str(STUDY.parent))  # dev/style.py
from style import FIGWIDTH, PLOT_RC, beautify_circuit_name, save_fig  # noqa: E402

plt.rcParams.update(PLOT_RC)

# plotly's sequential "Sunset" as a matplotlib colormap (identical hex stops).
SUNSET = LinearSegmentedColormap.from_list(
    "sunset",
    ["#f3e79b", "#fac484", "#f8a07e", "#eb7f86", "#ce6693", "#a059a0", "#5c53a5"],
)

# scenario -> plot type (hm: one panel per ansatz, hms: single panel)
HEATMAPS = {
    "1dfs_rx_mse_uw": "hm",
    "1dfs_ry_mse_uw": "hm",
    "1dfs_rx_mse_w": "hm",
    "1dfs_ry_mse_w": "hm",
    "2dfs_mse_uw": "hms",
    "2dfs_mse_w": "hms",
    "1dfs_rc": "hms",
}


def tickval_to_tex(tickvals, use_latex=False, optimize=True):
    ticktext = []
    ct = -1
    for tick in tickvals:
        t = tick.replace("+", "").split("_")
        if len(t) == 2:
            if use_latex:
                ticktext.append(f"${t[0]}_{{{t[1]}}}$")
            else:
                ticktext.append(f"{t[1]}")
        elif len(t) == 3:
            if optimize:
                if int(t[1]) > ct:
                    if use_latex:
                        ticktext.append(f"${t[0]}_{{{t[1]},*}}$")
                    else:
                        ticktext.append(f"{t[1]},*")
                    ct = int(t[1])
                else:
                    ticktext.append("")
            else:
                ticktext.append(f"${t[0]}_{{{t[1]},{t[2]}}}$")

    return ticktext


def heatmap_panel(hdf, ansatz):
    """Rebuild ``(z, x, y, zmax)`` for one panel from the long-format CSV.

    The pivot runs on the integer ``yi``/``xi`` positions, since the ``x``/``y``
    tick labels would sort lexicographically and scramble the matrix.
    """
    g = hdf[hdf.ansatz == ansatz]
    z = g.pivot(index="yi", columns="xi", values="z").sort_index()
    z = z.sort_index(axis=1).to_numpy()
    x = g.drop_duplicates("xi").sort_values("xi").x.tolist()
    y = g.drop_duplicates("yi").sort_values("yi").y.tolist()

    return z, x, y, float(g.zmax.iloc[0])


def visualize_heatmap(hdf):
    ansaetze = hdf.ansatz.unique()

    rows = 2
    cols = len(ansaetze) // rows

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(FIGWIDTH, FIGWIDTH * rows / cols + 1.0),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    # First pass: collect raw matrices to share one colour scale.
    panels = []
    for it, ansatz in enumerate(ansaetze):
        row_idx = 0 if it < cols else 1
        col_idx = it % cols
        z, x, y, zmax = heatmap_panel(hdf, ansatz)
        panels.append((axes[row_idx, col_idx], z, x, y, zmax, ansatz, row_idx, col_idx))

    vmax = max(p[4] for p in panels)

    for ax, z, x, y, zmax, ansatz, row_idx, col_idx in panels:
        im = ax.pcolormesh(z, cmap=SUNSET, vmin=0.0, vmax=vmax)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # matches plotly's autorange="reversed"
        ax.set_title(beautify_circuit_name(ansatz))
        ax.grid(False)
        ax.set_xticks(np.arange(len(x)) + 0.5)
        ax.set_yticks(np.arange(len(y)) + 0.5)
        # plain indices, $c_k$ labels overlap at this panel width
        if row_idx == rows - 1:
            ax.set_xticklabels(tickval_to_tex(x))
        else:
            ax.set_xticklabels([])
        if col_idx == 0:
            ax.set_yticklabels(tickval_to_tex(y))
        else:
            ax.set_yticklabels([])

    # Single figure-level labels; per-panel labels overlap at \figurewidth.
    fig.supxlabel("Coefficients", fontsize=plt.rcParams["axes.labelsize"])
    fig.supylabel("Coefficients", fontsize=plt.rcParams["axes.labelsize"])

    cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    cbar.solids.set_rasterized(False)  # keep the .pgf self-contained (no png sidecar)

    return fig


def visualize_single_heatmap(hdf):
    fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGWIDTH), constrained_layout=True)

    z, x, y, zmax = heatmap_panel(hdf, hdf.ansatz.iloc[0])

    im = ax.pcolormesh(z, cmap=SUNSET, vmin=0.0, vmax=zmax)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # matches plotly's autorange="reversed"
    ax.grid(False)
    ax.set_xticks(np.arange(len(x)) + 0.5)
    ax.set_yticks(np.arange(len(y)) + 0.5)
    ax.set_xticklabels(tickval_to_tex(x, use_latex=True))
    ax.set_yticklabels(tickval_to_tex(y, use_latex=True))
    ax.set_xlabel("Coefficients")
    ax.set_ylabel("Coefficients")
    cbar = fig.colorbar(im, ax=ax)
    cbar.solids.set_rasterized(False)  # keep the .pgf self-contained (no png sidecar)

    return fig


def main(data: Path) -> None:
    for scenario, plot in HEATMAPS.items():
        path = data / "heatmaps" / f"{scenario}.csv"
        if not path.exists():
            print(f"Skipping {scenario}: {path} not found")
            continue

        hdf = pd.read_csv(path)
        for q in hdf.qubits.unique():
            _hdf = hdf[hdf.qubits == q]
            fig = visualize_heatmap(_hdf) if plot == "hm" else visualize_single_heatmap(_hdf)
            save_fig(fig, FIGURES, f"{scenario}_{plot}_q{q}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reference", action="store_true", help="read reference/ instead of results/"
    )
    args = parser.parse_args()
    main(STUDY / ("reference" if args.reference else "results"))
