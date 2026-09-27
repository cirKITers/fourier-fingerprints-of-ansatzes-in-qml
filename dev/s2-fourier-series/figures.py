"""FCC and expressibility against the Fourier-series training error (s2).

Reads CSVs only, from ``results/`` or, with ``--reference``, from
``reference/`` (the CSVs behind the paper and thesis figures), and writes
``.pgf`` and ``.pdf`` plus the error tables (``.csv``, ``.tex``) to
``figures/``:

    uv run --no-project --with matplotlib --with pandas --with numpy --with jinja2 \\
        python dev/s2-fourier-series/figures.py --reference

Outputs, one per scenario in SCENARIOS and qubit count ``q``:

- ``<scenario>_sce_q<q>``: seed-averaged FCC (left axis) and expressibility
  (right axis) against the minimal validation error, one marker per ansatz
- ``<scenario>_err_q<q>``: standard deviation across seeds of the per-seed
  means of expressibility, FCC and error, one column per ansatz
- ``<scenario>_var_q<q>``: seed-averaged variance of the real (lines) and
  imaginary (markers) part of each coefficient (``_uw`` 1dfs scenarios only)

CSV schema: ``<scenario>.csv`` with one row per (qubits, ansatz, seed), i.e.
one training configuration averaged over its training runs, joined with the s1
FCC run of the same encoding and the s1 expressibility run on
(ansatz, qubits, seed):

- qubits: number of qubits
- ansatz: circuit name, e.g. Hardware_Efficient
- ansatz_id: integer id per ansatz, sets the marker order
- seed: seed shared by the joined runs
- corr_mean: FCC, mean absolute coefficient correlation
- corr_w_mean: weighted FCC (plotted by the ``_w`` scenarios)
- expressibility: logged expressibility metric, mean KL divergence to the Haar
  distribution
- mse_valid_min: minimal validation MSE over training, averaged over runs
- coeff_var_abs, coeff_var_real, coeff_var_imag: per-frequency variance of
  $|c_\\omega|$, $\\Re(c_\\omega)$ and $\\Im(c_\\omega)$ over parameter samples,
  as a list string ``[v_0, v_1, ...]`` (used by ``var``)
- coeff_mean_abs: per-frequency mean of $|c_\\omega|$, same format (read by
  the thesis plot.py only)

``1dfs_{rx,ry}`` is the 1D series with RX/RY encoding, ``2dfs`` the 2D series.
The ``_uw`` and ``_w`` files of one series carry the same rows; ``_w`` selects
``corr_w_mean``. Further columns are ignored; the reference files keep the
original export (MLflow run ids, ``corr_min``, ``corr_max``, ``corr_var``,
``steps``, empty ``*_var`` columns). The file names match those the thesis
plot.py reads.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

STUDY = Path(__file__).resolve().parent
FIGURES = STUDY / "figures"

sys.path.insert(0, str(STUDY.parent))  # dev/style.py
from style import (  # noqa: E402
    COLOURS,
    FIGWIDTH,
    PLOT_RC,
    beautify_circuit_name,
    calculate_errors,
    export_pandas_table,
    save_fig,
    style_axes,
    visualize_expr_scatter,
)

plt.rcParams.update(PLOT_RC)

# scenario -> (metric, metric name, weighted, plots)
SCENARIOS = {
    "1dfs_rx_mse_uw": ("mse_valid", "Mean Squared Error", False, ["sce", "var"]),
    "1dfs_ry_mse_uw": ("mse_valid", "Mean Squared Error", False, ["sce", "var"]),
    "1dfs_rx_mse_w": ("mse_valid", "Mean Squared Error", True, ["sce"]),
    "1dfs_ry_mse_w": ("mse_valid", "Mean Squared Error", True, ["sce"]),
    "2dfs_mse_uw": ("mse_valid", "Mean Squared Error", False, ["sce"]),
    "2dfs_mse_w": ("mse_valid", "Mean Squared Error", True, ["sce"]),
}


def visualize_coeff_variance(df):
    ansaetze = df.ansatz.unique()

    fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGWIDTH * 0.73))
    colors = itertools.cycle(COLOURS)
    handles = []
    for ansatz in ansaetze:
        color = next(colors)
        _df = df[(df.ansatz == ansatz)]

        # the coeff_* columns are per-frequency vectors serialised as strings
        coeff_var_abs = _df["coeff_var_abs"].apply(lambda x: np.array(json.loads(x))).mean(axis=0)
        coeff_var_real = _df["coeff_var_real"].apply(lambda x: np.array(json.loads(x))).mean(axis=0)
        coeff_var_imag = _df["coeff_var_imag"].apply(lambda x: np.array(json.loads(x))).mean(axis=0)
        coeff_var_real = np.array(coeff_var_real, dtype=float)
        coeff_var_imag = np.array(coeff_var_imag, dtype=float)

        coeff_var_real[coeff_var_abs < 1e-10] = np.nan
        coeff_var_imag[coeff_var_imag < 1e-10] = np.nan

        ax.plot(coeff_var_real, color=color)
        ax.plot(coeff_var_imag, ls="", marker="o", color=color)
        handles.append(
            Line2D(
                [], [], marker="o", color=color, markersize=6,
                label=beautify_circuit_name(ansatz),
            )
        )

    ax.set_yscale("log")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Coefficient Variance")
    style_axes(ax)
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    return fig


def main(data: Path) -> None:
    for scenario, (metric, metric_name, weighted, plots) in SCENARIOS.items():
        path = data / f"{scenario}.csv"
        if not path.exists():
            print(f"Skipping {scenario}: {path} not found")
            continue

        df = pd.read_csv(path)
        ansatz_ids = df.ansatz_id.unique()
        for q in df.qubits.unique():
            _df = df[df.qubits == q]

            if "sce" in plots:
                errors_table = calculate_errors(_df, f"{metric}_min", weighted=weighted)
                export_pandas_table(errors_table, FIGURES, f"{scenario}_err_q{q}")

                fig = visualize_expr_scatter(
                    _df, ansatz_ids, f"{metric}_min", metric_name, weighted=weighted
                )
                save_fig(fig, FIGURES, f"{scenario}_sce_q{q}")

            if "var" in plots:
                save_fig(visualize_coeff_variance(_df), FIGURES, f"{scenario}_var_q{q}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reference", action="store_true", help="read reference/ instead of results/"
    )
    args = parser.parse_args()
    main(STUDY / ("reference" if args.reference else "results"))
