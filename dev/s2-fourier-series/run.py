"""Fourier-series training (s2): QFMs fitted to random Fourier series.

Submits the train cells to a running Fluksio engine (dev/serve.sh), skipping
every cell that already has an ok, queued or running run. The export joins the
s1 fingerprint and expressibility runs of the same ansaetze and seeds, so run
s1 for them as well:

    uv run python dev/s2-fourier-series/run.py --dry-run
    uv run python dev/s2-fourier-series/run.py --ansaetze Circuit_19 --seeds 1000

Cells: series (1dfs_ry, 1dfs_rx, 2dfs) x ansatz x seed (model) x data seed,
2000 full-batch steps with the flow defaults.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # dev/common.py
from common import ENCODINGS, QUBITS, circuit, parser, submit  # noqa: E402


def cells(args, series=tuple(ENCODINGS)):
    """The train cells of `series`."""
    return [
        {
            "n_qubits": QUBITS,
            "circuit_type": circuit(a, args.variant),
            "encoding": ENCODINGS[s],
            "seed": seed,
            "data_seed": data_seed,
        }
        for s in series
        for a in args.ansaetze
        for seed in args.seeds
        for data_seed in args.data_seeds
    ]


if __name__ == "__main__":
    p = parser(__doc__, data_seeds=True)
    p.add_argument("--jobs", type=int, default=2, help="runs in flight")
    p.add_argument("--dry-run", action="store_true", help="list the cells only")
    args = p.parse_args()
    submit("train", cells(args), args.jobs, args.dry_run)
