"""FCC against training MSE for the encoding strategies (s4).

Reads CSVs only, from ``results/`` or, with ``--reference``, from
``reference/`` (a bundled CSV), and writes ``.pgf`` and
``.pdf`` to ``figures/``:

    uv run --no-project --with matplotlib --with pandas --with numpy \\
        python dev/s4-encodings/figures.py --reference

Output: ``encoding_strategy_sce``, the seed-averaged FCC against the
seed-averaged training MSE, one point per (ansatz, encoding_strategy), with a
least-squares fit of $\\log_{10}$ FCC on MSE per encoding strategy and the
Pearson correlation $r$ of the two in the legend. If the CSV has n_support and
n_params (the export does, the reference does not), also
``encoding_strategy_confounds.csv``, see `confounds`. With ``--excess`` (for the
rerun) the same for fcc_excess, the excess of the FCC over its null value, as
``encoding_strategy_excess_sce`` and ``encoding_strategy_excess_confounds.csv``.

CSV schema: ``encoding_strategy.csv`` with one row per training run, i.e. per
(ansatz, encoding_strategy, seed):

- ansatz: circuit name, e.g. Hardware_Efficient
- encoding_strategy: hamming, binary or ternary
- fcc: FCC of the ansatz under that encoding
- train_mse: training MSE of the run
- n_support, n_params (optional): size of the numerical support and number of
  trainable parameters of the model
- var_sum (optional): sum of the coefficient variances over all frequencies
- fcc_excess (for ``--excess``): fcc minus its null value

Further columns are ignored. The reference CSV also includes run IDs, seeds,
training steps, and train_fmse.
"""

import argparse
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
from style import COLOURS, FIGWIDTH, PLOT_RC, save_fig, style_axes  # noqa: E402

plt.rcParams.update(PLOT_RC)

# Fixed order and colour, so each encoding strategy keeps its colour.
ENCODING_STRATEGIES = {
    "hamming": COLOURS[3],  # teal
    "binary": COLOURS[1],  # orange
    "ternary": COLOURS[6],  # blue
}


def visualize_encoding_scatter(df, fcc="fcc"):
    """FCC against training MSE, one point per ansatz, coloured by encoding.

    The strategies sit in near-disjoint bands about a decade apart, so the FCC
    axis is logarithmic and the fits are least squares on $\\log_{10}$ FCC.
    fcc_excess can be negative and is shown on a linear axis.
    """
    log = fcc == "fcc"
    means = df.groupby(["ansatz", "encoding_strategy"], as_index=False)[
        [fcc, "train_mse"]
    ].mean()

    fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGWIDTH * 0.73))

    handles = []
    for strategy, color in ENCODING_STRATEGIES.items():
        _df = means[means.encoding_strategy == strategy]
        if len(_df) == 0:
            print(f"No data for encoding_strategy={strategy}")
            continue
        x = _df.train_mse.to_numpy()
        y = np.log10(_df[fcc].to_numpy()) if log else _df[fcc].to_numpy()

        ax.scatter(
            x, 10**y if log else y, marker="o", color=color, alpha=0.7, linewidths=0
        )
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        fit = slope * xs + intercept
        ax.plot(xs, 10**fit if log else fit, color=color)

        r = np.corrcoef(x, y)[0, 1]
        print(f"{strategy}: n={len(x)}, r(train_mse, {'log10 ' * log}{fcc})={r:+.3f}")
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                color=color,
                markersize=6,
                label=rf"{strategy} ($r$={r:.2f})",
            )
        )

    if log:
        ax.set_yscale("log")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel("Mean Squared Error")
    ax.set_ylabel("FCC" if log else "FCC excess")
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


def correlations(x, y, *zs):
    """Pearson correlation of x and y, and of their residuals after regressing on each z."""

    def residual(a, z):
        return a - z @ np.linalg.lstsq(z, a, rcond=None)[0]

    partial = [np.corrcoef(residual(x, z), residual(y, z))[0, 1] for z in zs]
    return np.corrcoef(x, y)[0, 1], *partial


def confounds(df, fcc="fcc", min_n=4, n_boot=2000, seed=0):
    """Correlation of FCC and training MSE with and without the model size.

    Per encoding strategy over the seed-averaged ansaetze: the Pearson $r$ of
    $\\log_{10}$ FCC (fcc_excess as is) and MSE as in the figure, and their
    partial correlation controlling for n_support and n_params and, if every
    row has var_sum, the one controlling for var_sum ($\\log_{10}$ like the FCC)
    and n_support (partial_r_var), each with the 2.5 and 97.5 percentiles over
    `n_boot` bootstrap resamples of the ansaetze. Computed for all ansaetze,
    for those with full support (the largest n_support of the encoding) and
    per support size, if at least `min_n`.
    """
    var = "var_sum" in df and bool(df.var_sum.notna().all())
    cols = [fcc, "train_mse", "n_support", "n_params", *["var_sum"] * var]
    means = df.groupby(["ansatz", "encoding_strategy"], as_index=False)[cols].mean()
    rng = np.random.default_rng(seed)

    rows = []
    for strategy in ENCODING_STRATEGIES:
        _df = means[means.encoding_strategy == strategy]
        subsets = {"all": _df, "full": _df[_df.n_support == _df.n_support.max()]}
        subsets.update((f"n_support={k:g}", s) for k, s in _df.groupby("n_support"))
        for subset, s in subsets.items():
            if len(s) < min_n:
                continue
            x = np.log10(s[fcc].to_numpy()) if fcc == "fcc" else s[fcc].to_numpy()
            y = s.train_mse.to_numpy()
            zs = [np.column_stack([np.ones(len(s)), s.n_support, s.n_params])]
            if var:
                v = np.log10(s.var_sum) if fcc == "fcc" else s.var_sum
                zs.append(np.column_stack([np.ones(len(s)), s.n_support, v]))

            # resamples with a constant column yield nan, skipped by the percentiles
            with np.errstate(invalid="ignore", divide="ignore"):
                boot = [
                    correlations(x[i], y[i], *(z[i] for z in zs))
                    for i in rng.integers(len(s), size=(n_boot, len(s)))
                ]
                lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
            row = {"encoding_strategy": strategy, "subset": subset, "n": len(s)}
            names = ["r", "partial_r", "partial_r_var"]
            for k, value in enumerate(correlations(x, y, *zs)):
                row.update(
                    {names[k]: value, f"{names[k]}_lo": lo[k], f"{names[k]}_hi": hi[k]}
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main(data: Path, excess: bool = False) -> None:
    path = data / "encoding_strategy.csv"
    if not path.exists():
        print(f"Skipping encoding_strategy: {path} not found")
        return

    df = pd.read_csv(path)
    fcc = "fcc_excess" if excess else "fcc"
    name = "encoding_strategy_excess" if excess else "encoding_strategy"
    save_fig(visualize_encoding_scatter(df, fcc), FIGURES, f"{name}_sce")
    if {"n_support", "n_params"} <= set(df):
        table = confounds(df, fcc)
        table.to_csv(FIGURES / f"{name}_confounds.csv", index=False)
        print(table.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reference", action="store_true", help="read reference/ instead of results/"
    )
    parser.add_argument(
        "--excess", action="store_true", help="fcc_excess instead of fcc (rerun)"
    )
    args = parser.parse_args()
    main(STUDY / ("reference" if args.reference else "results"), args.excess)
