"""Shared grid constants, engine helpers and tables of the study drivers.

A cell holds the flow inputs that differ from the flow defaults; runs are
matched on all inputs, so a cell is only ever answered by a run of exactly its
configuration. The engine is found as by the fluksio CLI: FLUKSIO_URL and
FLUKSIO_TOKEN, or the client.json of the nearest .fluksio/ (see dev/serve.sh).
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import pandas as pd

QUBITS = 6
# order of the paper panels and ansatz ids (descending names)
ANSAETZE = [
    "Hardware_Efficient",
    "Circuit_YZY_Entangling",
    "Circuit_YZY",
    "Circuit_19",
    "Circuit_18",
    "Circuit_17",
    "Circuit_16",
    "Circuit_15",
]
SEEDS = list(range(1000, 1010))
ENCODINGS = {"1dfs_ry": ["RY"], "1dfs_rx": ["RX"], "2dfs": ["RX", "RY"]}
# FCC definition the paper reported per series, see the s1 README
PUBLISHED_FCC = {
    "1dfs_ry": "corr_mean",
    "1dfs_rx": "corr_full_mean",
    "2dfs": "corr_full_mean",
}
# the published variant ran the Circuit_15 of qml-essentials 0.1.35
PAPER_CIRCUITS = {"Circuit_15": "Circuit_15_Paper"}
OUTPUTS = {
    "fingerprint": ["stats"],
    "surrogate": ["stats"],
    "expressibility": ["expressibility"],
    "train": ["final_metrics", "differences"],
    "encoding": ["results"],
}
COEFF_STATS = [
    f"coeff_{s}_{p}" for s in ("mean", "var") for p in ("abs", "real", "imag")
]


def parser(doc: str, data_seeds: bool = False) -> argparse.ArgumentParser:
    """Command line options shared by the drivers and exports."""
    p = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--variant", choices=["paper", "revised"], default="paper")
    p.add_argument("--ansaetze", nargs="+", default=ANSAETZE, metavar="ANSATZ")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS, metavar="SEED")
    if data_seeds:
        p.add_argument(
            "--data-seeds", nargs="+", type=int, default=SEEDS, metavar="SEED"
        )
    return p


def circuit(ansatz: str, variant: str) -> str:
    """The circuit_type that runs `ansatz` in `variant`."""
    return PAPER_CIRCUITS.get(ansatz, ansatz) if variant == "paper" else ansatz


def ansatz_of(circuit_type: str) -> str:
    """The paper name of a circuit_type."""
    return {v: k for k, v in PAPER_CIRCUITS.items()}.get(circuit_type, circuit_type)


def fingerprint_cell(circuit_type: str, series: str, seed: int) -> dict[str, Any]:
    """The s1 fingerprint cell, $2^n \\cdot 500$ samples per series."""
    encoding = ENCODINGS[series]
    return {
        "n_qubits": QUBITS,
        "circuit_type": circuit_type,
        "encoding": encoding,
        "seed": seed,
        "n_samples": 500 // len(encoding),
    }


def expressibility_cell(circuit_type: str, seed: int) -> dict[str, Any]:
    """The s1 expressibility cell."""
    return {"n_qubits": QUBITS, "circuit_type": circuit_type, "seed": seed}


def _key(cell: dict[str, Any]) -> str:
    return json.dumps(cell, sort_keys=True)


def _complete(flow: str, cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from fluksio.sdk import FLOWS

    import fourier_fingerprints.pipeline  # noqa: F401, registers the flows

    defaults = {p.name: p.initial for p in FLOWS[flow].inputs}
    return [{**defaults, **cell} for cell in cells]


def runs(flow: str, cells: list[dict[str, Any]]) -> pd.DataFrame:
    """The newest ok run of each cell: its inputs, run (the id) and the outputs."""
    from fluksio.sdk.client import Client

    cells = _complete(flow, cells)
    wanted = {_key(c) for c in cells}
    keys = sorted(cells[0])
    found = {}
    # newest first, so the first run of a cell wins
    for row in Client().export_runs(
        flow=flow, status="ok", metrics=",".join(OUTPUTS[flow])
    ):
        cell = {k: row.get(f"param.{k}") for k in keys}
        if _key(cell) in wanted and _key(cell) not in found:
            outputs = {k: row[f"metric.{k}"] for k in OUTPUTS[flow]}
            found[_key(cell)] = {**cell, "run": row["id"], **outputs}
    return pd.DataFrame(list(found.values()), columns=[*keys, "run", *OUTPUTS[flow]])


def submit(flow: str, cells: list[dict[str, Any]], jobs: int, dry_run: bool) -> None:
    """Runs every cell not yet ok, queued or running, `jobs` at a time."""
    from fluksio.sdk.client import Client

    if dry_run:
        for cell in cells:
            print(flow, _key(cell))
        print(f"{flow}: {len(cells)} cells")
        return

    client = Client()
    cells = _complete(flow, cells)
    keys = sorted(cells[0])
    seen = {
        _key({k: row.get(f"param.{k}") for k in keys})
        for row in client.export_runs(flow=flow)
        if row["status"] in ("ok", "queued", "running")
    }
    queue = [c for c in cells if _key(c) not in seen]
    total = len(queue)
    print(
        f"{flow}: {total} to submit, {len(cells) - total} done or in flight", flush=True
    )

    inflight = []
    done = 0
    while queue or inflight:
        while queue and len(inflight) < jobs:
            cell = queue.pop(0)
            inflight.append((cell, client.submit(flow, cell, seed=cell["seed"])))
        time.sleep(5.0)
        for cell, handle in list(inflight):
            if not handle.refresh().done:
                continue
            inflight.remove((cell, handle))
            done += 1
            print(f"  [{done}/{total}] {handle.status} {handle.id}", flush=True)


def fcc_table(series: str, circuits: list[str], seeds: list[int]) -> pd.DataFrame:
    """
    The s1 statistics of `series` and the expressibility per (circuit_type, seed).

    corr_mean holds the FCC as published for the series (PUBLISHED_FCC),
    corr_tril_mean the mean over the strict lower triangle.
    """
    cells = [fingerprint_cell(c, series, s) for c in circuits for s in seeds]
    rows = []
    for _, r in runs("fingerprint", cells).iterrows():
        stats = r.stats
        row = {k: v for k, v in stats.items() if not isinstance(v, list)}
        row.update({k: json.dumps(stats[k]) for k in COEFF_STATS})
        row.update(
            circuit_type=r.circuit_type,
            seed=r.seed,
            corr_tril_mean=stats["corr_mean"],
            corr_mean=stats[PUBLISHED_FCC[series]],
            coeff_run_id=r.run,
        )
        rows.append(row)
    cells = [expressibility_cell(c, s) for c in circuits for s in seeds]
    expr = runs("expressibility", cells)[
        ["circuit_type", "seed", "expressibility", "run"]
    ]
    return pd.DataFrame(rows).merge(
        expr.rename(columns={"run": "expr_run_id"}), on=["circuit_type", "seed"]
    )


def scenario_table(train: pd.DataFrame, fcc: pd.DataFrame) -> pd.DataFrame:
    """Training runs averaged per (circuit_type, seed) and joined with `fcc`."""
    metrics = pd.DataFrame(list(train.final_metrics), index=train.index, dtype=float)
    g = pd.concat([train[["circuit_type", "seed", "run"]], metrics], axis=1).groupby(
        ["circuit_type", "seed"]
    )
    df = g[list(metrics)].mean()
    df["mse_valid_min_var"] = g.mse_valid_min.var()
    df["training_run_id"] = g.run.agg(list).astype(str)
    df = df.reset_index().merge(fcc, on=["circuit_type", "seed"])
    df.insert(0, "qubits", QUBITS)
    df.insert(1, "ansatz", df.circuit_type.map(ansatz_of))
    df.insert(2, "ansatz_id", df.ansatz.map(ANSAETZE.index))
    return df.sort_values(["seed", "ansatz_id"])
