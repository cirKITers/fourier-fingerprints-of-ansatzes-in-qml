"""Exports the s3 training runs, joined with s1, to the CSVs of figures.py.

    uv run python dev/s3-hep/export.py [--variant paper|revised]

Writes results/2dhep_{mse,kl,hl}_uw.csv, one row per (qubits, ansatz, seed)
with the QFM metrics averaged over the data seeds and the s1 2D FCC, and
results/dist_{hist,kde}.csv from the $p_T$ differences of the Circuit_15 QFM
and of the MLP runs, averaged elementwise over the runs (definition paper) and
pooled across the runs (definition pooled).
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fluksio.sdk.client import Client
from scipy.stats import gaussian_kde

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY.parent))  # dev/common.py
from common import (  # noqa: E402
    QUBITS,
    circuit,
    fcc_table,
    parser,
    runs,
    scenario_table,
)
from run import cells  # noqa: E402


def differences(train):
    """Differences of `train` per definition and split: paper, the elementwise
    mean over the runs, and pooled, all differences of all runs."""
    loaded = [
        np.load(io.BytesIO(Client().download(ref["digest"])))
        for ref in train.differences
    ]
    splits = ("train", "valid")
    return {
        "paper": {s: np.mean([d[s] for d in loaded], axis=0) for s in splits},
        "pooled": {s: np.concatenate([d[s] for d in loaded]) for s in splits},
    }


def distribution(data, identifier, series, definition):
    """Histogram (bin width 2) and KDE rows of one series."""
    keys = {
        "scenario": "2dhepc",
        "definition": definition,
        "identifier": identifier,
        "qubits": QUBITS,
        "ansatz": "Circuit_15",
        "series": series,
    }
    density, edges = np.histogram(
        data, bins=np.arange(data.min(), data.max() + 2, 2), density=True
    )
    hist = pd.DataFrame(
        {
            **keys,
            "left": edges[:-1],
            "right": edges[1:],
            "density": density,
            "mu": data.mean(),
            "sigma": data.std(),
        }
    )
    x = np.linspace(data.min(), data.max(), 200)
    return hist, pd.DataFrame({**keys, "x": x, "y": gaussian_kde(data)(x)})


def main(args):
    out = STUDY / "results"
    out.mkdir(parents=True, exist_ok=True)
    c = cells(args)
    qfm, mlp = runs("train", c["qfm"]), runs("train", c["mlp"])
    if len(qfm):
        circuits = [circuit(a, args.variant) for a in args.ansaetze]
        df = scenario_table(qfm, fcc_table("2dfs", circuits, args.seeds))
        for metric in ("mse", "kl", "hl"):
            df.to_csv(out / f"2dhep_{metric}_uw.csv", index=False)
        print(f"Wrote {out}/2dhep_*_uw.csv ({len(df)} rows)")

    qfm = qfm[qfm.circuit_type == circuit("Circuit_15", args.variant)]
    if not (len(qfm) and len(mlp)):
        print("Skipping dist: no Circuit_15 QFM or no MLP runs")
        return
    diffs = {"QFM": differences(qfm), "MLP": differences(mlp)}
    frames = [
        distribution(
            d[definition][split], f"fig_distribution_{split}", series, definition
        )
        for definition in ("paper", "pooled")
        for split in ("train", "valid")
        for series, d in diffs.items()
    ]
    for i, name in enumerate(("dist_hist", "dist_kde")):
        pd.concat([f[i] for f in frames]).to_csv(
            out / f"{name}.csv", index=False, float_format="%.6g"
        )
    print(f"Wrote {out}/dist_*.csv from {len(qfm)} QFM and {len(mlp)} MLP runs")


if __name__ == "__main__":
    main(parser(__doc__, data_seeds=True).parse_args())
