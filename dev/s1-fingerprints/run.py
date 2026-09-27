"""Fourier fingerprints (s1): FCC, expressibility and the random-coefficient surrogate.

Submits the cells to a running Fluksio engine (dev/serve.sh), skipping every
cell that already has an ok, queued or running run, so an interrupted grid is
resumed by running this again:

    uv run python dev/s1-fingerprints/run.py --dry-run
    uv run python dev/s1-fingerprints/run.py --ansaetze Circuit_19 --seeds 1000

Cells: fingerprint per series (1dfs_ry, 1dfs_rx, 2dfs) x ansatz x seed,
expressibility per ansatz x seed, surrogate (Circuit_15) per seed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # dev/common.py
from common import (  # noqa: E402
    ENCODINGS,
    QUBITS,
    circuit,
    expressibility_cell,
    fingerprint_cell,
    parser,
    submit,
)


def cells(args):
    """The cells of each flow."""
    circuits = [circuit(a, args.variant) for a in args.ansaetze]
    return {
        "fingerprint": [
            fingerprint_cell(c, series, s)
            for series in ENCODINGS
            for c in circuits
            for s in args.seeds
        ],
        "expressibility": [
            expressibility_cell(c, s) for c in circuits for s in args.seeds
        ],
        "surrogate": [
            {
                "n_qubits": QUBITS,
                "circuit_type": circuit("Circuit_15", args.variant),
                "seed": s,
            }
            for s in args.seeds
        ],
    }


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--jobs", type=int, default=2, help="runs in flight per flow")
    p.add_argument("--dry-run", action="store_true", help="list the cells only")
    args = p.parse_args()
    for flow, flow_cells in cells(args).items():
        submit(flow, flow_cells, args.jobs, args.dry_run)
