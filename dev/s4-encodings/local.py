"""Local FCC (s4): `metrics.local_fcc` for the encoding cells, without training.

Runs in-process (no engine) and writes one row per ansatz, encoding strategy
and seed to results/local_fcc_<n_qubits>q<n_layers>l.csv:

    JAX_PLATFORMS=cpu uv run python dev/s4-encodings/local.py --n-qubits 5 --n-layers 2 --jobs 4

The seed seeds the model, the numerical support and the parameter draws. The
columns are those of `local_fcc` without fingerprint_local, see the Local FCC
section of the README.
"""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from run import ANSAETZE, ENCODINGS

from fourier_fingerprints.metrics import local_fcc
from fourier_fingerprints.model import create_model

STUDY = Path(__file__).resolve().parent
N_SAMPLES = 5


def cell(n_qubits, n_layers, ansatz, encoding, seed):
    """The local FCC of one cell."""
    model = create_model(
        n_qubits, n_layers, ansatz, encoding_strategy=encoding, seed=seed
    )
    stats = local_fcc(model, N_SAMPLES, seed)
    stats.pop("fingerprint_local")
    return {
        "ansatz": ansatz,
        "encoding_strategy": encoding,
        "seed": seed,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_samples": N_SAMPLES,
        **stats,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n-qubits", type=int, required=True)
    p.add_argument("--n-layers", type=int, required=True)
    p.add_argument("--ansaetze", nargs="+", default=ANSAETZE, metavar="ANSATZ")
    p.add_argument("--encodings", nargs="+", choices=ENCODINGS, default=ENCODINGS)
    p.add_argument("--seeds", nargs="+", type=int, default=[1000, 1001, 1002])
    p.add_argument("--jobs", type=int, default=1, help="processes")
    args = p.parse_args()

    cells = [
        (args.n_qubits, args.n_layers, a, e, s)
        for a in args.ansaetze
        for e in args.encodings
        for s in args.seeds
    ]
    # spawn, as JAX is multithreaded
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.jobs, mp_context=ctx) as ex:
        rows = list(ex.map(cell, *zip(*cells)))

    out = STUDY / "results" / f"local_fcc_{args.n_qubits}q{args.n_layers}l.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out} ({len(rows)} rows)")
