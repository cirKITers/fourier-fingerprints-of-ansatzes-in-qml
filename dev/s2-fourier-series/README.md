# s2 Fourier-series training

Does the FCC (or the expressibility) of an ansatz predict how well a quantum
Fourier model (QFM) with it fits a random Fourier series of matching spectrum? Each
QFM is trained on a series whose coefficients are drawn uniformly from the unit
disc, on the equidistant grid resolving its highest frequency, and validated on the
same grid.

| flow | cells | settings |
| --- | --- | --- |
| train | 3 series $\times$ 8 ansaetze $\times$ 10 seeds $\times$ 10 data seeds | 6 qubits, 1 layer, 2000 full-batch Adam steps, learning rate 0.01, MSE |

Series, ansaetze and seeds as in s1; seed is the model (initialization) seed,
data_seed the seed of the series. The export averages the minimal validation
metrics over the data seeds and joins the s1 fingerprint and expressibility runs of
the same series, ansatz and seed, so s1 has to run for the same ansaetze and seeds.
The FCC columns are those of s1 (published corr_mean, corr_tril_mean, the
signal-only corr_mean_signal and corr_w_mean_signal).

## How to re-run it

```sh
RUNS=4 DEVICES=4 dev/serve.sh                          # the engine, on ./.fluksio
uv run fluksio sync fourier_fingerprints/pipeline.py   # after every code change
uv run python dev/s1-fingerprints/run.py               # FCC and expressibility
uv run python dev/s2-fourier-series/run.py --dry-run   # list the cells
uv run python dev/s2-fourier-series/run.py             # submit, resumable
uv run python dev/s2-fourier-series/export.py          # results/<series>_mse_{uw,w}.csv
uv run python dev/s2-fourier-series/figures.py         # figures/, --reference for the paper CSVs
```

`--ansaetze`, `--seeds` and `--data-seeds` restrict the grid (pass the same to
the export), `--variant` selects the Circuit_15 wiring as in s1.
