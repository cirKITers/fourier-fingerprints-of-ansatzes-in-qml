"""HEP regression (s3): QFMs and the classical MLP on the $pp \\to Z \\to$ jets dataset.

Submits the train cells to a running Fluksio engine (dev/serve.sh), skipping
every cell that already has an ok, queued or running run. The export joins the
s1 2D fingerprint and expressibility runs of the same ansaetze and seeds, so
run s1 for them as well:

    uv run python dev/s3-hep/run.py --dry-run
    uv run python dev/s3-hep/run.py --ansaetze Circuit_15 --seeds 1000 --data-seeds 1000

Cells: QFM per ansatz x seed (model) x data seed (3000 steps), MLP per seed x
data seed (150 epochs). The paper variant runs Circuit_15_Paper and the MLP
with the broadcast targets of the paper.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # dev/common.py
from common import QUBITS, circuit, parser, submit  # noqa: E402

QFM = {
    "n_qubits": QUBITS,
    "encoding": ["RX", "RY"],
    "dataset": "hep",
    "steps": 3000,
    "learning_rate": 0.005,
    "loss_function": ["mse", "kl_divergence"],
}
MLP = {
    "dataset": "hep",
    "model": "mlp",
    "steps": 150,
    "learning_rate": 0.005,
    "loss_function": ["kl_divergence", "huber_loss"],
}


def cells(args):
    """The QFM and MLP train cells."""
    seeds = [(s, d) for s in args.seeds for d in args.data_seeds]
    qfm = [
        {**QFM, "circuit_type": circuit(a, args.variant), "seed": s, "data_seed": d}
        for a in args.ansaetze
        for s, d in seeds
    ]
    mlp = [
        {**MLP, "broadcast_targets": args.variant == "paper", "seed": s, "data_seed": d}
        for s, d in seeds
    ]
    return {"qfm": qfm, "mlp": mlp}


if __name__ == "__main__":
    p = parser(__doc__, data_seeds=True)
    p.add_argument("--jobs", type=int, default=2, help="runs in flight")
    p.add_argument("--dry-run", action="store_true", help="list the cells only")
    args = p.parse_args()
    c = cells(args)
    submit("train", c["mlp"] + c["qfm"], args.jobs, args.dry_run)
