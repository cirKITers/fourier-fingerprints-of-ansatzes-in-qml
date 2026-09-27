"""Exports the s2 training runs, joined with s1, to the scenario CSVs of figures.py.

    uv run python dev/s2-fourier-series/export.py [--variant paper|revised]

Writes results/<series>_mse_{uw,w}.csv, one row per (qubits, ansatz, seed)
with the metrics averaged over the data seeds.
"""

import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY.parent))  # dev/common.py
from common import (  # noqa: E402
    ENCODINGS,
    circuit,
    fcc_table,
    parser,
    runs,
    scenario_table,
)
from run import cells  # noqa: E402


def main(args):
    out = STUDY / "results"
    out.mkdir(parents=True, exist_ok=True)
    circuits = [circuit(a, args.variant) for a in args.ansaetze]
    for series in ENCODINGS:
        train = runs("train", cells(args, [series]))
        if not len(train):
            print(f"Skipping {series}: no training runs")
            continue
        df = scenario_table(train, fcc_table(series, circuits, args.seeds))
        for suffix in ("uw", "w"):
            df.to_csv(out / f"{series}_mse_{suffix}.csv", index=False)
            print(f"Wrote {out / series}_mse_{suffix}.csv ({len(df)} rows)")


if __name__ == "__main__":
    main(parser(__doc__, data_seeds=True).parse_args())
