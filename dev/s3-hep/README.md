# s3 HEP regression

Does the FCC ranking carry over to real data? QFMs with two inputs regress the
transverse momentum $p_T$ of the leading jet in $pp \to Z \to$ jets events from the
center of mass energy and the parton energy difference, compared to a small
classical MLP. The dataset is data/pp-z-to-jets-500K-57246.h5 (40000 events,
split 80/10/10 by data_seed).

| flow | cells | settings |
| --- | --- | --- |
| train (QFM) | 8 ansaetze $\times$ 10 seeds $\times$ 10 data seeds | 6 qubits, RX and RY encoding, 3000 full-batch Adam steps, learning rate 0.005, MSE $+ 10^{-3}$ KL |
| train (MLP) | 10 seeds $\times$ 10 data seeds | width 8, depth 2, 150 epochs, batch 256, learning rate 0.005, KL $+ 10^{-3}$ Huber |

The export averages the minimal validation MSE, KL divergence and Huber loss of the
QFM over the data seeds and joins the s1 2dfs fingerprint and expressibility runs,
so s1 has to run for the same ansaetze and seeds. The distribution plot uses the
prediction errors in GeV of all Circuit_15 QFM runs and of all MLP runs, pooled
by default, since the elementwise mean over runs (`--definition paper`)
averages differences of different events and shrinks the spread by about
$\sqrt{N}$ for $N$ runs.

`--variant paper` (default) runs Circuit_15_Paper and the MLP with
broadcast targets (the Huber term compares all prediction-target pairs of a
batch); `--variant revised` runs the current Circuit_15 and the elementwise Huber
loss.

## How to re-run it

```sh
RUNS=4 DEVICES=4 dev/serve.sh                          # the engine, on ./.fluksio
uv run fluksio sync fourier_fingerprints/pipeline.py   # after every code change
uv run python dev/s1-fingerprints/run.py               # FCC and expressibility
uv run python dev/s3-hep/run.py --dry-run              # list the cells
uv run python dev/s3-hep/run.py                        # submit, resumable
uv run python dev/s3-hep/export.py                     # results/2dhep_*_uw.csv, dist_*.csv
uv run python dev/s3-hep/figures.py                    # figures/, --reference for bundled CSVs
```

A QFM run takes about an hour on 16 cores (1.2 s per step), an MLP run a few
minutes. `--ansaetze`, `--seeds` and `--data-seeds` restrict the grid (pass the
same to the export).
