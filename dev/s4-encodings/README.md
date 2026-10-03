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
| power | 17 ansaetze $\times$ 3 encodings $\times$ 10 seeds | rerun on targets of power $P = 0.01$ |
| matched | 5 ansaetze $\times$ 3 encodings $\times$ 10 seeds | power at 5 qubits and 2 layers with 30 trainable parameters |

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

1-18 uses 5 qubits and 2 layers and compares the model spectrum with
unnormalized target coefficients for train_fmse (`unnormalized_target`).
rerun drops coefficients outside the numerical support (below tol $= 10^{-12}$
in every sample) from the FCC and sets them to zero in the target. At 500
samples, its FCC reaches the estimator's noise floor (see Convergence). The
rerun therefore uses 8000 samples and fcc_excess as the primary measure. The
learning rate is chosen by lr (see Learning rate).

power fixes the scale of the target. The series is
$\sum_\omega c_\omega e^{i \omega x} / K$ with $K$ the size of the frequency
grid (13, 127 and 729 for Hamming, binary and ternary encoding at 6 qubits and
1 layer; `Datasets.calculate_values` divides by the number of coefficients),
so its power $\sum_\omega |c_\omega / K|^2$, the
mean of $y^2$ over the grid, falls as $1 / K$. In rerun the median target power
is 0.021, 0.0017 and 0.00014, while the output power of the models at random
parameters (var_sum) has a median of 0.026 under every encoding. Under binary
and ternary encoding the training therefore mostly suppresses the model output,
and train_mse follows the output power of the model rather than the fit.
power rescales the pruned coefficients of every target to
$\sum_\omega |c_\omega / K|^2 = P$ (`target_power`). $P = 0.01$ is the lower
quartile of the rerun Hamming targets and below var_sum for 11 of the 17
ansaetze at 6 qubits and 1 layer (12 under Hamming encoding) and for 13 at 5
qubits and 2 layers.
`--n-qubits` and `--n-layers` set the size (6 and 1 by default),
`--target-power` sets $P$.

matched separates the FCC from the number of parameters. At 5 qubits and 2
layers Strongly_Entangling, Circuit_14, Circuit_19, Circuit_4 and Circuit_2
(90, 60, 45, 42 and 30 parameters) have the full support under every encoding,
and power orders their fcc_pearson and train_pnmse as their parameter count.
matched trains only a random subset of 30 parameters (`n_trainable`, the
subset seeded with the seed), the others keep their initial values, and samples
only this subset for the FCC, so that the FCC describes the trained model.
Circuit, support and target are those of power; matching by the number of
layers would change the spectrum. For Circuit_2 the cells equal those of power.

## Results

Per run the export writes run_id, ansatz, encoding_strategy, data.seed,
model.seed, fcc.seed, train.steps, fcc, train_mse and train_fmse. The fcc
column selects fcc_pruned for rerun and fcc_unpruned otherwise. It also writes:

- fcc_null: the null value of fcc
- fcc_excess: fcc minus fcc_null, the primary measure of rerun
- at_floor: whether fcc / fcc_null is below 1.1
- fcc_pearson_excess: fcc_pearson minus its null value
- train_nmse: train_mse divided by the variance of the target. The variance
  excludes the offset, so train_nmse can exceed 1 for a target with a large
  offset (Circuit_9 under Hamming encoding)
- train_pnmse: train_mse divided by target_power_actual, the power of the
  target (mean of its square, $P$ for power; empty for runs from before it was
  added)
- n_support: number of non-negative frequencies in the numerical support
  ($\vert \Omega_s \vert$), n_params: number of parameters of the model
- var_sum: $\sum_{\omega \in \Omega} \mathrm{Var}(c_{\omega})$ over all
  frequencies (outside the support the terms are numerically zero), from the
  same samples as the FCC. With zero-mean coefficients the expected MSE over
  the parameters is var_sum plus the squared norm of the target coefficients,
  i.e. it contains no correlations
- fcc_unpruned, fcc_pruned: covariance FCC over all non-negative frequencies and
  over the support
- fcc_pearson: mean absolute Pearson correlation of the real parts over the
  support (corr_mean_signal of s1)
- the null value of each FCC with suffix _null, and the settings n_qubits,
  n_layers, n_samples, learning_rate, prune, unnormalized_target,
  target_power and n_trainable (the number of trained parameters, 0 for all)

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
ansaetze. If every run has var_sum (runs from before it was added do not),
partial_r_var is the partial correlation controlling for $\log_{10}$ var_sum
(var_sum with `--excess`) and n_support, i.e. whether the FCC relates to the
MSE beyond the coefficient variances. Each is computed for all ansaetze (subset
all), for those with full support (full, the largest n_support of the
encoding) and per support size (n_support=k), for subsets of at least 4
ansaetze; within a support size partial_r controls for n_params only and
partial_r_var for var_sum only. `--excess` does the same for
fcc_excess on a linear axis (encoding_strategy_excess_*), for rerun.

## 1-18 validation

The recomputed FCC of all 510 cells matches the reference to a relative
$3.4 \cdot 10^{-15}$, except for the 30 Circuit_9 cells ($1.0 \cdot 10^{-6}$):
qml-essentials a2eee507 built its Hadamard matrix on import, before x64 was
enabled, i.e. in single precision, and Circuit_9 is the only ansatz with
Hadamard gates. train_mse and train_fmse of 19 cells trained on the engine
(Circuit_9, Circuit_19 and Strongly_Entangling with seeds 1002 and 1007,
Hardware_Efficient with Hamming encoding and seed 1000) match to
$2 \cdot 10^{-16}$, for Circuit_9 to $8 \cdot 10^{-7}$.

For 1-18, figures.py gives $r = 0.84$, $0.73$ and $0.09$. With
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

## Parameter matching

The matched cells (all 10 seeds, run in-process with the encoding node; four
power cells rerun this way match the engine to $6 \cdot 10^{-16}$) keep the full
support in every cell. Pearson $r$ of $\log_{10}$ fcc_pearson and train_pnmse
over the seed-averaged 5 ansaetze, with the 95 % interval over 2000 bootstrap
resamples of the seeds, the median and range of $r$ per seed, and the partial
correlation controlling for $\log_{10}$ var_sum:

| encoding | power $r$ | matched $r$ | matched $r$ per seed | matched partial $r$ |
| --- | --- | --- | --- | --- |
| hamming | 0.96 $[0.89, 0.99]$ | 0.74 $[0.48, 0.86]$ | 0.44 $[0.16, 0.95]$ | -0.29 |
| binary | 0.97 $[0.95, 0.98]$ | 0.91 $[0.78, 0.97]$ | 0.82 $[0.36, 0.99]$ | 0.90 |
| ternary | 0.93 $[0.91, 0.94]$ | -0.57 $[-0.80, -0.32]$ | -0.52 $[-0.85, 0.19]$ | -0.23 |

The FCC keeps its order Strongly_Entangling < Circuit_14 < Circuit_19 <
Circuit_4 < Circuit_2 under matching (Circuit_4 and Circuit_2 swap under
Hamming encoding), but train_pnmse no longer follows it: Strongly_Entangling
and Circuit_14 stay lowest under Hamming and binary encoding, Circuit_19 and
Circuit_4 are highest, and under ternary encoding train_pnmse is between 0.89
and 0.96 for every ansatz, i.e. 30 parameters barely fit the target. Under
matching, $\log_{10}$ fcc_pearson correlates with $\log_{10}$ var_sum at
$r = 0.89$, $0.64$ and $0.71$. The FCC predicts the error beyond the number of
parameters and var_sum only under binary encoding.

## Local FCC

The FCC correlates the coefficients over the whole parameter space, where they
are nearly uncorrelated for every ansatz, while a trained model can only move
within the coefficients reachable around its parameters. `metrics.local_fcc`
evaluates, without training, the Jacobian $J = \partial u / \partial \theta$
of the output coordinates $u$ in the orthonormal real Fourier basis of the
numerical support ($D_s$ dimensions, $2 \vert \Omega_s \vert - 1$ with the
offset) at 5 random parameter sets. $J J^\top$ is the covariance of the
coefficients under small parameter perturbations and its Pearson matrix
$R_{loc}$ the local fingerprint: fcc_local is the mean $|r|$ over its strict
lower triangle, r2_local the mean $r^2$, pr_local
$\mathrm{PR}(R_{loc}) / D_s = D_s / \lVert R_{loc} \rVert_F^2$ and rank the
numerical rank $m$ of $J$ (singular values above $10^{-9}$ times the largest),
with $\mathrm{PR}(R_{loc}) \le m \le$ n_params. For a linearized model the best
fit of an isotropic target of power $P$ leaves $P (1 - m / D_s)$ on average.

```python
from fourier_fingerprints.metrics import local_fcc
from fourier_fingerprints.model import create_model

model = create_model(5, 2, "Circuit_19", encoding_strategy="binary", seed=1000)
local_fcc(model, n_samples=5, seed=1000)  # mask as in fcc_variants for matched
```

local_fcc is not part of the encoding flow. local.py computes it in-process for
every ansatz, encoding and seed (1000 to 1002 by default, 5 draws each) and
writes results/local_fcc_<n_qubits>q<n_layers>l.csv, one row per cell with the
columns of local_fcc except fingerprint_local; with `--jobs 4` a grid of 153
cells takes about 7 min. Pearson $r$ with the seed-averaged train_pnmse of
power (5 qubits, 2 layers) and of the same grid at 4 qubits and 3 layers, with
the seed means of local_fcc_5q2l.csv and local_fcc_4q3l.csv:

| cells | fcc_local | pr_local | $m / D_s$ | $\log_{10}$ fcc_pearson | $\log_{10}$ var_sum |
| --- | --- | --- | --- | --- | --- |
| all (102) | 0.10 | -0.87 | -0.95 | -0.11 | -0.12 |
| $m < D_s$ (72) | 0.28 | -0.83 | -0.92 | -0.06 | -0.13 |
| $m = D_s$ (30, Hamming) | 0.49 | -0.44 | | -0.09 | -0.10 |
| per size and encoding (17 each) | 0.45 to 0.86 | -0.53 to -0.84 | -0.60 to -0.97 | -0.31 to 0.40 | -0.35 to 0.10 |

$m / D_s$ explains most of the variation ($R^2 = 0.90$ over all cells, 0.93
with fcc_local). fcc_local orders the ansaetze within a size and encoding, but
tracks $1 / m$ rather than $m / D_s$ (mean $r^2 \ge (D_s / m - 1) / (D_s - 1)$),
so it does not carry the differences between the encodings. Given $m / D_s$ its
partial $r$ is 0.53 $[0.29, 0.71]$ (95 % bootstrap interval over the cells) for
$m < D_s$; for $m = D_s$ given n_params / $D_s$ it is 0.33 $[-0.11, 0.66]$.

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
uv run python dev/s4-encodings/run.py --variant power --n-qubits 5 --n-layers 2  # 6 and 1 without the options
uv run python dev/s4-encodings/export.py --variant power --n-qubits 5 --n-layers 2
uv run python dev/s4-encodings/run.py --variant matched     # 5 qubits and 2 layers by default
uv run python dev/s4-encodings/export.py --variant matched
uv run python dev/s4-encodings/figures.py --excess    # figures/, --reference for bundled CSV
JAX_PLATFORMS=cpu uv run python dev/s4-encodings/local.py --n-qubits 5 --n-layers 2 --jobs 4  # no engine
JAX_PLATFORMS=cpu uv run python dev/s4-encodings/local.py --n-qubits 4 --n-layers 3 --jobs 4
```

`--variant 1-18`, `--variant rerun`, `--variant power` and `--variant matched`
export to `results/encoding_strategy.csv` (the last export wins), `--variant convergence`
to `results/convergence.csv` and `--variant lr` to `results/learning_rate.csv`.
`--ansaetze`, `--encodings` and `--seeds` restrict the grid, `--n-qubits`,
`--n-layers` and `--target-power` override the settings of the variant (pass
the same to the export), `--jobs` bounds the runs in flight. figures.py plots
train_mse, which is $P$ times train_pnmse for power, so its correlations are
those of train_pnmse. A run from before `target_power` was added counts as
`target_power` 0, the behaviour it had. `reference/encoding_strategy.csv`
contains the bundled 1-18 results.
