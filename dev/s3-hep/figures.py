"""FCC and expressibility against the HEP training error, QFM against MLP (s3).

Reads CSVs only, from ``results/`` or, with ``--reference``, from
``reference/`` (the CSVs behind the paper and thesis figures), and writes
``.pgf`` and ``.pdf`` plus the error tables (``.csv``, ``.tex``) to
``figures/``:

    uv run --no-project --with matplotlib --with pandas --with numpy --with jinja2 \\
        python dev/s3-hep/figures.py --reference

Outputs, per qubit count ``q``:

- ``<scenario>_sce_q<q>`` and ``<scenario>_err_q<q>``: as in s2, for each
  scenario in SCENARIOS
- ``2dhep_dist_{train,valid}_q<q>``: distribution of the differences between
  predicted and true $p_T$ for the QFM and the classical MLP, of the definition
  selected by ``--definition`` (pooled, or paper with ``--reference``)

CSV schemas:

``<scenario>.csv`` with one row per (qubits, ansatz, seed), as in s2 (training
runs of that seed averaged, joined with the s1 FCC and expressibility runs).
Required columns: qubits, ansatz, ansatz_id, seed, corr_mean, expressibility
and ``<metric>_min`` (mse_valid_min, kl_divergence_valid_min or
huber_loss_valid_min, per SCENARIOS). Further columns are ignored; the
reference files keep the original export, where each file fills only its own
``<metric>_min`` column.

``dist_hist.csv`` with one row per histogram bin, i.e. per (definition,
identifier, qubits, ansatz, series, left):

- scenario: constant ``2dhepc``, kept for the thesis plot.py
- definition: paper (differences averaged elementwise over the training runs)
  or pooled (differences of all runs pooled); rows without the column (the
  reference files) count as paper
- identifier: fig_distribution_train or fig_distribution_valid
- qubits, ansatz: QFM configuration
- series: QFM or MLP
- left, right: bin edges (width 2)
- density: normalised bin height, as ``np.histogram(density=True)``
- mu, sigma: mean and standard deviation of the differences of the series

``dist_kde.csv`` with one row per KDE grid point, i.e. per (definition,
identifier, qubits, ansatz, series, x), with scenario, definition, identifier,
qubits, ansatz and series as above plus:

- x: grid point, 200 points from min to max of the differences
- y: Gaussian KDE (scipy default bandwidth) of the differences at ``x``

The differences come from the training runs of all seeds (88 QFM and 100 MLP
runs in the reference, extracted from MLflow). Averaging the differences of
different events across $N$ runs shrinks their spread by about $\\sqrt{N}$,
hence pooled is the default.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent
FIGURES = STUDY / "figures"

sys.path.insert(0, str(STUDY.parent))  # dev/style.py
from style import (  # noqa: E402
    EXPR_COLOR,
    FCC_COLOR,
    FIGWIDTH,
    PLOT_RC,
    calculate_errors,
    export_pandas_table,
    save_fig,
    style_axes,
    visualize_expr_scatter,
)

plt.rcParams.update(PLOT_RC)

# scenario -> (metric, metric name)
SCENARIOS = {
    "2dhep_mse_uw": ("mse_valid", "Mean Squared Error"),
    "2dhep_kl_uw": ("kl_divergence_valid", "KL Divergence"),
    "2dhep_hl_uw": ("huber_loss_valid", "Huber Loss"),
}


def visualize_distribution(hist, kde):
    """Redraw the baked histogram and KDE curve.

    ``ax.stairs`` reproduces what ``ax.hist(density=True)`` drew at extraction
    time, so no re-binning happens here.
    """
    fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGWIDTH * 0.67))

    for (ansatz, name), g in hist.groupby(["ansatz", "series"], sort=False):
        color = FCC_COLOR if name == "QFM" else EXPR_COLOR
        label = rf"{name}: $\mu$={g.mu.iloc[0]:.3f}, $\sigma$={g.sigma.iloc[0]:.2f}"
        edges = np.append(g.left.to_numpy(), g.right.to_numpy()[-1])
        ax.stairs(
            g.density.to_numpy(), edges, fill=True, alpha=0.5, color=color, label=label
        )

        k = kde[(kde.ansatz == ansatz) & (kde.series == name)]
        if len(k) > 0:
            ax.plot(k.x, k.y, color=color)

    ax.set_xlabel("Absolute difference of transverse momenta")
    ax.set_ylabel("Frequency")
    style_axes(ax)
    ax.legend(loc="upper right")

    return fig


def main(data: Path, definition: str) -> None:
    for scenario, (metric, metric_name) in SCENARIOS.items():
        path = data / f"{scenario}.csv"
        if not path.exists():
            print(f"Skipping {scenario}: {path} not found")
            continue

        df = pd.read_csv(path)
        ansatz_ids = df.ansatz_id.unique()
        for q in df.qubits.unique():
            _df = df[df.qubits == q]

            errors_table = calculate_errors(_df, f"{metric}_min")
            export_pandas_table(errors_table, FIGURES, f"{scenario}_err_q{q}")

            fig = visualize_expr_scatter(_df, ansatz_ids, f"{metric}_min", metric_name)
            save_fig(fig, FIGURES, f"{scenario}_sce_q{q}")

    if not (data / "dist_hist.csv").exists():
        print(f"Skipping dist: {data / 'dist_hist.csv'} not found")
        return

    hist = pd.read_csv(data / "dist_hist.csv")
    kde = pd.read_csv(data / "dist_kde.csv")
    for d in (hist, kde):
        if "definition" not in d:
            d["definition"] = "paper"  # the reference files predate the column
    hist = hist[hist.definition == definition]
    kde = kde[kde.definition == definition]
    for (identifier, q), h in hist.groupby(["identifier", "qubits"]):
        k = kde[(kde.identifier == identifier) & (kde.qubits == q)]
        split = identifier.removeprefix("fig_distribution_")
        save_fig(visualize_distribution(h, k), FIGURES, f"2dhep_dist_{split}_q{q}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reference", action="store_true", help="read reference/ instead of results/"
    )
    parser.add_argument(
        "--definition",
        choices=["paper", "pooled"],
        help="dist definition (default: pooled, paper with --reference)",
    )
    args = parser.parse_args()
    definition = args.definition or ("paper" if args.reference else "pooled")
    main(STUDY / ("reference" if args.reference else "results"), definition)
