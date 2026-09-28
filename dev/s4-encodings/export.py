"""Exports the s4 encoding runs of a variant to the CSVs of figures.py.

    uv run python dev/s4-encodings/export.py [--variant 1-18|rerun|convergence|power]

Writes results/encoding_strategy.csv (1-18, rerun and power), results/convergence.csv
or results/learning_rate.csv (lr), one row per run in the schema of the thesis
file (run_id, ansatz, encoding_strategy, data.seed, model.seed, fcc.seed,
train.steps, fcc, train_mse, train_fmse) followed by the further results and
settings of the run. fcc is the covariance FCC of the variant, restricted to the numerical
support with prune, fcc_excess its excess over the null value and at_floor
whether it is below 1.1 times the null value. train_pnmse is train_mse over
the power of the target (target_power_actual, the mean of its square).
"""

import sys
from pathlib import Path

import pandas as pd

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY.parent))  # dev/common.py
from common import runs  # noqa: E402
from run import cells, parser  # noqa: E402

COLUMNS = [
    "run_id",
    "ansatz",
    "encoding_strategy",
    "data.seed",
    "model.seed",
    "fcc.seed",
    "train.steps",
    "fcc",
    "train_mse",
    "train_fmse",
    "fcc_null",
    "fcc_excess",
    "at_floor",
    "fcc_pearson_excess",
    "train_nmse",
    "train_pnmse",
    "target_power_actual",
    "n_support",
    "n_params",
    "var_sum",
    *(f"fcc_{k}{s}" for k in ("unpruned", "pruned", "pearson") for s in ("", "_null")),
    "n_qubits",
    "n_layers",
    "n_samples",
    "learning_rate",
    "prune",
    "unnormalized_target",
    "target_power",
    "n_trainable",
]


def main(args):
    df = runs("encoding", cells(args))
    if not len(df):
        print(f"Skipping {args.variant}: no runs")
        return
    df = pd.concat([df, pd.DataFrame(list(df.results), index=df.index)], axis=1)
    df["fcc"] = df.fcc_pruned.where(df.prune, df.fcc_unpruned)
    df["fcc_null"] = df.fcc_pruned_null.where(df.prune, df.fcc_unpruned_null)
    df["fcc_excess"] = df.fcc - df.fcc_null
    df["at_floor"] = df.fcc < 1.1 * df.fcc_null
    df["fcc_pearson_excess"] = df.fcc_pearson - df.fcc_pearson_null
    for k in ("data", "model", "fcc"):
        df[f"{k}.seed"] = df.seed
    df = df.rename(
        columns={"run": "run_id", "circuit_type": "ansatz", "steps": "train.steps"}
    )
    df = df.reindex(columns=COLUMNS).sort_values(
        ["ansatz", "encoding_strategy", "model.seed", "n_samples", "learning_rate"]
    )
    df["train_pnmse"] = df.train_mse / df.target_power_actual

    out = STUDY / "results"
    out.mkdir(parents=True, exist_ok=True)
    name = {"convergence": "convergence", "lr": "learning_rate"}.get(
        args.variant, "encoding_strategy"
    )
    df.to_csv(out / f"{name}.csv", index=False)
    print(f"Wrote {out / name}.csv ({len(df)} rows)")


if __name__ == "__main__":
    main(parser(__doc__).parse_args())
