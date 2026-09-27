# s4 Encoding strategies

Does the relation between FCC and training error extend beyond Hamming encoding?
Every qml-essentials ansatz is combined with the Hamming, binary and ternary
encoding strategy of an RY encoding. The FCC is computed from random parameter
samples and a quantum Fourier model (QFM) with the same encoding is trained on a
random Fourier series with the spectrum of the encoding.

| variant | cells | settings |
| --- | --- | --- |
| 1-18 | 17 ansaetze $\times$ 3 encodings $\times$ 10 seeds | 5 qubits, 2 layers, full spectrum, learning rate $10^{-4}$ |
| rerun | 17 ansaetze $\times$ 3 encodings $\times$ 10 seeds | 6 qubits, 1 layer, FCC and target on the numerical support, 8000 samples, learning rate $0.01$ |
| convergence | 4 ansaetze $\times$ 3 encodings $\times$ 2 seeds $\times$ 3 sample sizes | FCC of rerun with 500, 2000 and 8000 samples, no training |
| lr | 4 ansaetze $\times$ 3 encodings $\times$ 4 learning rates | rerun with learning rates $10^{-3}$, $3 \cdot 10^{-3}$, $10^{-2}$ and $3 \cdot 10^{-2}$, seed 1000 |

All variants run the encoding flow. Ansaetze: Circuit_2, 3, 4, 7, 8, 9, 10, 13,
14, 15, 16, 17, 18, 19, 20, Strongly_Entangling and Hardware_Efficient as defined
by qml-essentials; convergence uses Circuit_2, Circuit_19, Strongly_Entangling and
Hardware_Efficient with seeds 1000 and 1001, lr Circuit_9, Hardware_Efficient,
Circuit_15 and Strongly_Entangling (small to full support). The seed (1000 to
1009) seeds the initial parameters, the FCC samples and the series. The models
re-upload the input in every layer, measure all qubits and draw their parameters
from $U[0, 2\pi]$. The FCC is the mean absolute covariance of the non-negative
Fourier coefficients over 500 parameter samples (8000 for rerun, not scaled with
the qubits), without weighting or numerical cap. The series draws its coefficients
uniformly from the unit disc, keeps the offset and is sampled on the equidistant
grid of the spectrum. Training is full-batch Adam (optax defaults) on the MSE for
3000 steps; the final step is reported. The Golomb encoding is left out, as its
spectrum is too expensive at these sizes.

1-18 reproduces the thesis figure (run study-1-18 of spectral-bias-x-fcc on
qml-essentials a2eee507); its train_fmse compares the model spectrum with the
unnormalized target coefficients as the original did (`unnormalized_target`).
rerun is the configuration the thesis text describes: coefficients outside the
numerical support (below tol $= 10^{-12}$ in every sample) are dropped from the
FCC and set to zero in the target. At 500 samples its FCC is at the floor of
the estimator (see Convergence), hence 8000 samples and fcc_excess as the
primary measure. The learning rate is chosen by lr (see Learning rate).

## Results

Per run the export writes the thesis columns (run_id, ansatz, encoding_strategy,
data.seed, model.seed, fcc.seed, train.steps, fcc, train_mse, train_fmse),
where fcc is the FCC of the variant (fcc_pruned for rerun, fcc_unpruned
otherwise), and further

- fcc_null: the null value of fcc
- fcc_excess: fcc minus fcc_null, the primary measure of rerun
- at_floor: whether fcc / fcc_null is below 1.1
- fcc_pearson_excess: fcc_pearson minus its null value
- train_nmse: train_mse divided by the variance of the target
- n_support: number of non-negative frequencies in the numerical support
  ($\vert \Omega_s \vert$), n_params: number of trainable parameters
- fcc_unpruned, fcc_pruned: covariance FCC over all non-negative frequencies and
  over the support
- fcc_pearson: mean absolute Pearson correlation of the real parts over the
  support (corr_mean_signal of s1)
- the null value of each FCC with suffix _null, and the settings n_qubits,
  n_layers, n_samples, learning_rate, prune and unnormalized_target

All FCC variants come from the same parameter samples. The null values repeat
each computation after permuting the samples independently per frequency, i.e.
for uncorrelated coefficients with the same marginals. An FCC close to its null
value is at the floor of the estimator: for $N$ samples the absolute covariance
of independent coefficients is of order $\sigma_{\omega} \sigma_{\omega'} / \sqrt{N}$
and the absolute Pearson correlation averages $\approx \sqrt{2 / (\pi N)}$.

figures.py plots the seed-averaged FCC against the seed-averaged MSE and writes
`figures/encoding_strategy_confounds.csv`: per encoding the Pearson $r$ of
$\log_{10}$ FCC and MSE, as in the figure, and their partial correlation
controlling for n_support and n_params, with 95 % bootstrap intervals over the
ansaetze. Each is computed for all ansaetze (subset all), for those with full
support (full, the largest n_support of the encoding) and per support size
(n_support=k), for subsets of at least 4 ansaetze; within a support size the
partial correlation controls for n_params only. `--excess` does the same for
fcc_excess on a linear axis (encoding_strategy_excess_*), for rerun.

## Reproduction of 1-18

The recomputed FCC of all 510 cells matches the reference to a relative
$3.4 \cdot 10^{-15}$, except for the 30 Circuit_9 cells ($1.0 \cdot 10^{-6}$):
qml-essentials a2eee507 built its Hadamard matrix on import, before x64 was
enabled, i.e. in single precision, and Circuit_9 is the only ansatz with
Hadamard gates. train_mse and train_fmse of 19 cells trained on the engine
(Circuit_9, Circuit_19 and Strongly_Entangling with seeds 1002 and 1007,
Hardware_Efficient with Hamming encoding and seed 1000) match to
$2 \cdot 10^{-16}$, for Circuit_9 to $8 \cdot 10^{-7}$.

figures.py reproduces the thesis values $r = 0.84$, $0.73$ and $0.09$. With
n_support and n_params of the recomputation the partial correlations are
$0.68$ $[0.34, 0.90]$ (Hamming), $0.71$ $[0.26, 0.91]$ (binary) and $0.59$
$[-0.14, 0.91]$ (ternary). Restricted to the 5 ansaetze with full support, $r$
is $0.90$ $[0.47, 1.00]$, $0.87$ $[0.62, 1.00]$ and $-0.30$ $[-1.00, 1.00]$.
Further support sizes with at least 4 ansaetze occur only under Hamming
encoding: 10 (5 ansaetze, $r = 0.71$ $[0.08, 1.00]$) and 5 (4 ansaetze,
$r = 0.83$ $[-1.00, 1.00]$).
Many values are at the floor: the median of
fcc / fcc_null is 1.23, 1.05 and 1.03, and 37 %, 66 % and 72 % of the cells
are below 1.1. Under binary encoding the null value alone correlates with the
MSE at $r = 0.62$, i.e. the relation follows mostly the coefficient variances.

## Convergence

The convergence cells at the rerun configuration, ratios averaged over the
ansaetze and seeds, and the median of fcc / fcc_null:

| encoding | fcc(2000) / fcc(500) | fcc(8000) / fcc(500) | fcc / fcc_null at 500, 2000, 8000 |
| --- | --- | --- | --- |
| hamming | 0.66 | 0.43 | 0.98, 1.47, 1.56 |
| binary | 0.53 | 0.30 | 1.00, 1.00, 1.05 |
| ternary | 0.52 | 0.28 | 1.01, 1.01, 1.01 |

The null value falls as $N^{-1/2}$ (ratios 0.48 to 0.50 and 0.25 to 0.27), and
the FCC nearly as fast: at 500 samples the rerun FCC is at the floor of the
estimator. Only Circuit_2 (fcc / fcc_null at 8000 samples: 4.3, 2.0 and 1.4)
and Circuit_19 under Hamming encoding (1.8) rise above it. For
Hardware_Efficient and Strongly_Entangling the ratio of fcc and of fcc_pearson
to its null value stays between 0.92 and 1.08 under binary and ternary encoding
and scatters between 0.7 and 1.5 under Hamming encoding (at most 21 pairs).

At 6 qubits and 1 layer the support sizes range from 2 to 7 (Hamming), 4 to 64
(binary) and 4 to 365 (ternary), the parameter counts from 12 (Circuit_9) to 72
(Strongly_Entangling).

## Learning rate

The lr cells (rerun at 8000 samples, seed 1000), median final train_nmse per
encoding over the 4 ansaetze, their mean, the median over all 12 runs and the
number of runs that diverged, still decreased by more than 1 % over the last
300 steps, or oscillated (more than 100 rising steps of a streamed train_mse
that is not yet at the numerical floor):

| learning rate | hamming | binary | ternary | mean | all | diverged | decreasing | oscillating |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $10^{-3}$ | 0.159 | 0.477 | 0.864 | 0.500 | 0.624 | 0 | 8 | 0 |
| $3 \cdot 10^{-3}$ | 0.114 | 0.430 | 0.796 | 0.447 | 0.557 | 0 | 2 | 0 |
| $10^{-2}$ | 0.097 | 0.367 | 0.745 | 0.403 | 0.524 | 0 | 1 | 2 |
| $3 \cdot 10^{-2}$ | 0.128 | 0.367 | 0.778 | 0.424 | 0.481 | 0 | 0 | 7 |

A run counts as diverged if its final train_mse is not finite, exceeds the
initial one or lost more than half of the decrease to its minimum. None did.
The criterion is the lowest median train_nmse per encoding: $10^{-2}$ is lowest
under Hamming and ternary encoding and tied with $3 \cdot 10^{-2}$ under binary
encoding, so the best value does not differ per encoding and rerun uses
$10^{-2}$. $3 \cdot 10^{-2}$ has the lowest median over all runs, but 7 of its
12 runs oscillate. Circuit_9 (support 2 or 4) stays above a train_nmse of 0.93
at every learning rate.

## Cost

Median seconds per cell on the engine with `RUNS=4 DEVICES=4` (16 cores):

| cells | hamming | binary | ternary |
| --- | --- | --- | --- |
| 1-18 | 27 | 40 | 79 |
| rerun, 500 samples | 26 | 38 | 94 |
| rerun, 8000 samples | 26 | 41 | 158 |
| convergence, 8000 samples | 3 | 6 | 64 |

A grid of 510 cells takes about 2 h with 4 runs at once, the rerun at 8000
samples about 2.7 h. The FCC evaluates the
spectra in chunks of 1000 parameter sets, a ternary run needs about 2.5 GB.

## How to re-run it

```sh
RUNS=4 DEVICES=4 dev/serve.sh                         # the engine, on ./.fluksio
uv run fluksio sync fourier_fingerprints/pipeline.py  # after every code change
uv run python dev/s4-encodings/run.py --variant rerun --dry-run  # list the cells
uv run python dev/s4-encodings/run.py --variant rerun            # submit, resumable
uv run python dev/s4-encodings/export.py --variant rerun         # results/encoding_strategy.csv
uv run python dev/s4-encodings/figures.py --excess    # figures/, --reference for the thesis CSV
```

`--variant 1-18` and `--variant rerun` export to `results/encoding_strategy.csv`
(the last export wins), `--variant convergence` to `results/convergence.csv` and
`--variant lr` to `results/learning_rate.csv`.
`--ansaetze`, `--encodings` and `--seeds` restrict the grid (pass the same to the
export), `--jobs` bounds the runs in flight. `reference/encoding_strategy.csv`
is the thesis file of 1-18.
