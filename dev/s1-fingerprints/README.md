# s1 Fourier fingerprints

How strongly are the Fourier coefficients of an ansatz correlated? The fingerprint
is the Pearson correlation matrix of the real parts of the non-negative Fourier
coefficients over random parameter samples, the Fourier coefficient correlation
(FCC) its mean absolute value. The expressibility (KL divergence to the Haar
distribution) and a random-coefficient surrogate serve as references.

| flow | cells | settings |
| --- | --- | --- |
| fingerprint | 3 series $\times$ 8 ansaetze $\times$ 10 seeds | 6 qubits, 1 layer, $2^6 \cdot 500$ samples |
| expressibility | 8 ansaetze $\times$ 10 seeds | $2^6 \cdot 500$ pairs, $6 \cdot 75$ bins |
| surrogate | Circuit_15 $\times$ 10 seeds | $2^6 \cdot 200$ samples |

Series: 1dfs_ry (RY encoding), 1dfs_rx (RX encoding) and 2dfs (RX, RY encoding of
two inputs, 250 samples per input). Ansaetze: Hardware_Efficient,
Circuit_YZY_Entangling, Circuit_YZY, Circuit_19, Circuit_18, Circuit_17,
Circuit_16, Circuit_15. Seeds 1000 to 1009; the heatmaps show seed 1000.

## Published and revised FCC

The paper reported the mean over the strict lower triangle (corr_mean) for 1dfs_ry,
but the mean over the full matrix including the diagonal (corr_full_mean) for
1dfs_rx and 2dfs. The s2 and s3 exports write the published definition as
corr_mean and keep the lower triangle mean as corr_tril_mean.

Both include the correlations of numerically zero coefficients, which are noise.
For 6 qubits and 1 layer the spectrum of Circuit_16, Circuit_17, Circuit_18,
Circuit_YZY, Circuit_YZY_Entangling and Hardware_Efficient is truncated at the
maximum frequency 1, 4, 1, 1, 2 and 3, so their FCC is noise-dominated. These
noise pairs depend on the rounding of the simulator, so the published values are
reproducible only statistically on the current stack. The revised signal-only FCC
(corr_mean_signal, corr_w_mean_signal) keeps only pairs of coefficients above the
tolerance tol.

`--variant paper` (default) runs Circuit_15_Paper, the Circuit_15 wiring of the
paper (qml-essentials 0.1.35), in place of Circuit_15; `--variant revised` runs the
current Circuit_15. Exports label both as Circuit_15.

## How to re-run it

```sh
RUNS=4 DEVICES=4 dev/serve.sh                         # the engine, on ./.fluksio
uv run fluksio sync fourier_fingerprints/pipeline.py  # after every code change
uv run python dev/s1-fingerprints/run.py --dry-run    # list the cells
uv run python dev/s1-fingerprints/run.py              # submit, resumable
uv run python dev/s1-fingerprints/export.py           # results/heatmaps/*.csv
uv run python dev/s1-fingerprints/figures.py          # figures/, --reference for the paper CSVs
```

`--ansaetze` and `--seeds` restrict the grid, `--jobs` bounds the runs in flight.
Running the drivers again submits only the cells without an ok, queued or running
run. The export and the figures read and write this folder's gitignored
`results/` and `figures/`; `reference/` holds the CSVs behind the paper figures.
