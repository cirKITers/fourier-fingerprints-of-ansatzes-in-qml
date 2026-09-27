"""Encoding strategies (s4): FCC and training error under each encoding strategy.

Submits the encoding cells of a variant to a running Fluksio engine
(dev/serve.sh), skipping every cell that already has an ok, queued or running
run:

    uv run python dev/s4-encodings/run.py --variant rerun --dry-run
    uv run python dev/s4-encodings/run.py --variant 1-18 --ansaetze Circuit_19

Cells: ansatz x encoding strategy x seed (model, FCC samples and series), see
VARIANTS for the settings that differ from the flow defaults and GRIDS for the
subsets of convergence (the rerun FCC at each sample size, without training)
and lr (the rerun at each learning rate).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # dev/common.py
from common import SEEDS, submit  # noqa: E402

# the qml-essentials ansaetze of the thesis figure
ANSAETZE = [
    *(f"Circuit_{i}" for i in (2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20)),
    "Strongly_Entangling",
    "Hardware_Efficient",
]
ENCODINGS = ["hamming", "binary", "ternary"]
RERUN = {
    "n_qubits": 6,
    "n_layers": 1,
    "prune": True,
    "n_samples": 8000,
    "learning_rate": 0.01,
}
VARIANTS = {
    # thesis figure as of study-1-18, train_fmse against the unnormalized target
    "1-18": {
        "n_qubits": 5,
        "n_layers": 2,
        "n_samples": 500,
        "learning_rate": 1e-4,
        "unnormalized_target": True,
    },
    "rerun": RERUN,
    "convergence": {**RERUN, "steps": 0},
    "lr": RERUN,
}
# ansaetze, seeds and the swept input of the subset variants
GRIDS = {
    "convergence": {
        "ansaetze": [
            "Circuit_2",
            "Circuit_19",
            "Strongly_Entangling",
            "Hardware_Efficient",
        ],
        "seeds": [1000, 1001],
        "n_samples": [500, 2000, 8000],
    },
    # small to full support
    "lr": {
        "ansaetze": [
            "Circuit_9",
            "Hardware_Efficient",
            "Circuit_15",
            "Strongly_Entangling",
        ],
        "seeds": [1000],
        "learning_rate": [1e-3, 3e-3, 1e-2, 3e-2],
    },
}


def parser(doc: str) -> argparse.ArgumentParser:
    """Command line options shared by the driver and the export."""
    p = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--variant", choices=list(VARIANTS), default="rerun")
    p.add_argument("--ansaetze", nargs="+", metavar="ANSATZ")
    p.add_argument("--encodings", nargs="+", choices=ENCODINGS, default=ENCODINGS)
    p.add_argument("--seeds", nargs="+", type=int, metavar="SEED")
    return p


def cells(args):
    """The encoding cells of `args.variant`."""
    variant, grid = VARIANTS[args.variant], GRIDS.get(args.variant, {})
    return [
        {
            **variant,
            "circuit_type": a,
            "encoding_strategy": e,
            "seed": seed,
            "n_samples": n,
            "learning_rate": lr,
        }
        for a in args.ansaetze or grid.get("ansaetze", ANSAETZE)
        for e in args.encodings
        for seed in args.seeds or grid.get("seeds", SEEDS)
        for n in grid.get("n_samples", [variant["n_samples"]])
        for lr in grid.get("learning_rate", [variant["learning_rate"]])
    ]


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--jobs", type=int, default=4, help="runs in flight")
    p.add_argument("--dry-run", action="store_true", help="list the cells only")
    args = p.parse_args()
    submit("encoding", cells(args), args.jobs, args.dry_run)
