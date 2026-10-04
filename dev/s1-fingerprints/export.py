"""Exports the s1 fingerprints of seed 1000 to the heatmap CSVs of figures.py.

    uv run python dev/s1-fingerprints/export.py [--variant paper|revised]

Writes results/heatmaps/<series>_mse_{uw,w}.csv (all ansaetze for 1D, the
Hardware_Efficient ansatz for 2D) and results/heatmaps/1dfs_rc.csv (surrogate).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY.parent))  # dev/common.py
from common import (  # noqa: E402
    ANSAETZE,
    ENCODINGS,
    QUBITS,
    ansatz_of,
    circuit,
    fingerprint_cell,
    parser,
    runs,
)
from run import cells  # noqa: E402

SEED = 1000


def label(freq):
    """Format a coefficient label, such as c_+1 or c_+0_+2."""
    return "c_" + "_".join(f"+{f}" for f in freq)


def ceiling(z):
    """$\\max |r|$ rounded up at its first significant digit."""
    zmax = np.nanmax(z)
    for digits in (1, 2, 3):
        if zmax > 10.0**-digits:
            return math.ceil(zmax * 10**digits) / 10**digits
    return float(zmax)


def heatmap(stats, key, ansatz, zmax=None):
    """Long-format rows of one heatmap panel."""
    z = np.abs(np.array(stats[key], dtype=float))
    x = [label(f) for f in stats["fingerprint_cols"]]
    y = [label(f) for f in stats["fingerprint_rows"]]
    return pd.DataFrame(
        {
            "qubits": QUBITS,
            "ansatz": ansatz,
            "yi": np.repeat(np.arange(len(y)), len(x)),
            "xi": np.tile(np.arange(len(x)), len(y)),
            "y": np.repeat(y, len(x)),
            "x": np.tile(x, len(y)),
            "z": z.ravel(),
            "zmax": ceiling(z) if zmax is None else zmax,
        }
    )


def main(args):
    out = STUDY / "results" / "heatmaps"
    out.mkdir(parents=True, exist_ok=True)
    ansaetze = [a for a in ANSAETZE if a in args.ansaetze]
    files = {}
    for series in ENCODINGS:
        panels = ansaetze if series.startswith("1d") else ["Hardware_Efficient"]
        found = runs(
            "fingerprint",
            [fingerprint_cell(circuit(a, args.variant), series, SEED) for a in panels],
        )
        found = found.set_index(found.circuit_type.map(ansatz_of))
        panels = [a for a in panels if a in found.index]
        for suffix, key in (("uw", "fingerprint"), ("w", "fingerprint_w")):
            files[f"{series}_mse_{suffix}"] = [
                heatmap(found.stats[a], key, a) for a in panels
            ]

    surrogate = runs(
        "surrogate", [c for c in cells(args)["surrogate"] if c["seed"] == SEED]
    )
    files["1dfs_rc"] = [
        heatmap(s, "fingerprint", "Circuit_15", 1.0) for s in surrogate.stats
    ]

    for name, frames in files.items():
        if not frames:
            print(f"Skipping {name}: no runs of seed {SEED}")
            continue
        pd.concat(frames).to_csv(out / f"{name}.csv", index=False, float_format="%.6g")
        print(f"Wrote {out / name}.csv")


if __name__ == "__main__":
    main(parser(__doc__).parse_args())
