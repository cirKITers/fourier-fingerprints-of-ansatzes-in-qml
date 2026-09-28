"""Fourier coefficient correlation (FCC) and expressibility of quantum models."""

import math
import warnings
from typing import Dict, Optional, Tuple

import jax
import numpy as np
from qml_essentials.coefficients import FCC, Coefficients
from qml_essentials.expressibility import Expressibility
from qml_essentials.model import Model


def paper_coefficients(
    model: Model, n_samples: int, seed: int, chunk_size: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Samples the non-negative Fourier coefficients as in the paper.

    Uses $2^n \\cdot$ `n_samples` $\\cdot D$ parameter sets for $n$ qubits and $D$
    input features, evaluated in chunks of `chunk_size` to bound the memory.

    Parameters
    ----------
    model : Model
        The quantum Fourier model, its parameters are restored afterwards.
    n_samples : int
        Number of samples per basis state and input feature.
    seed : int
        Seed of the parameter samples.
    chunk_size : int, optional
        Number of parameter sets evaluated at once.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Complex coefficients of shape (K, samples) and their integer frequency
        vectors of shape (K, D), for the K frequencies that are non-negative in
        every input dimension (row-major order, first input outermost).
    """
    n_total = 2**model.n_qubits * n_samples * model.n_input_feat
    params = model.params
    model.initialize_params(jax.random.PRNGKey(seed), repeat=n_total)
    samples = model.params

    coeffs = []
    for chunk in np.array_split(np.arange(n_total), -(-n_total // chunk_size)):
        c, freqs = Coefficients.get_spectrum(
            model, shift=True, trim=True, params=samples[chunk]
        )
        coeffs.append(np.asarray(c))
    model.params = params

    freqs = np.rint(np.reshape(freqs, (model.n_input_feat, -1))).astype(int)
    grid = np.stack(np.meshgrid(*freqs, indexing="ij"), axis=-1)
    grid = grid.reshape(-1, model.n_input_feat)
    coeffs = np.concatenate(coeffs, axis=-1).reshape(len(grid), n_total)

    keep = (grid >= 0).all(axis=1)
    return coeffs[keep], grid[keep]


def random_coefficients(
    model: Model, n_samples: int, seed: int, mean: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Samples the random-coefficient surrogate of the paper.

    Real coefficients are drawn from normal distributions whose scales follow
    the L2 normalized binomial coefficients $\\binom{2L}{L + k}$ for $k = 0 \\dots L$,
    with $L$ the highest frequency of `model`, using $2^n \\cdot$ `n_samples`
    samples.

    Parameters
    ----------
    model : Model
        Model providing the number of qubits and the highest frequency.
    n_samples : int
        Number of samples per basis state.
    seed : int
        Seed of the numpy random generator.
    mean : float, optional
        Mean of the normal distributions.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Coefficients of shape (L, samples) and frequencies of shape (L, 1) for
        $k = 1 \\dots L$. The zero frequency is sampled but dropped, as the paper's
        column filter did not match it.
    """
    degree = int(np.max(model.frequencies[0]))
    rng = np.random.default_rng(seed)

    pascal = np.array([math.comb(2 * degree, k) for k in range(2 * degree + 1)])
    scales = (pascal / np.linalg.norm(pascal))[degree:]
    coeffs = np.stack(
        [rng.normal(mean, s, 2**model.n_qubits * n_samples) for s in scales]
    )

    return coeffs[1:], np.arange(1, degree + 1)[:, None]


def correlation_stats(
    coeffs: np.ndarray, freqs: np.ndarray, tol: float = 1e-12
) -> Dict:
    """
    Computes the paper FCC and related statistics of sampled coefficients.

    Reproduces the paper pipeline: Pearson correlation of the real parts only
    (pandas `DataFrame.corr` silently dropped the imaginary part of complex
    columns), restricted to the strict lower triangle. The weighted variant
    uses the linear weights $w_{ij} = 1 - \\frac{i + j}{2 (K - 1)}$ over the
    flat frequency index.

    The signal-only variants keep only pairs of frequencies within the
    numerical support, i.e. whose real part exceeds `tol` in any sample, and
    thereby drop the correlations of numerically zero coefficients.

    Parameters
    ----------
    coeffs : np.ndarray
        Coefficients of shape (K, samples).
    freqs : np.ndarray
        Frequency vectors of shape (K, D).
    tol : float, optional
        Magnitude below which a coefficient counts as numerically zero.

    Returns
    -------
    Dict
        corr_mean (the FCC), corr_max, corr_min and corr_var (variance over the
        columns of the column-wise variances) of $|r|$, corr_w_mean of the
        weighted $|r|$, corr_full_mean of $|r|$ over the full matrix including
        the diagonal (reported as FCC for 1D RX and 2D in the paper),
        corr_mean_signal and corr_w_mean_signal restricted to the support
        (nan for less than two frequencies), the support frequencies support,
        per-frequency coefficient statistics
        coeff_{mean,var}_{abs,real,imag}, the signed fingerprints fingerprint
        and fingerprint_w of shape (K - 1, K - 1) (nan outside the strict lower
        triangle) and their row and column frequencies fingerprint_rows and
        fingerprint_cols.
    """
    n = len(freqs)
    idx = np.arange(n)
    lower = idx[:, None] > idx[None, :]
    weights = 1 - (idx[:, None] + idx[None, :]) / (2 * (n - 1))

    with warnings.catch_warnings():
        # constant coefficients yield nan correlations, which are skipped
        warnings.simplefilter("ignore", RuntimeWarning)
        r = np.corrcoef(coeffs.real)
        fingerprint = np.where(lower, r, np.nan)[1:, :-1]
        fingerprint_w = np.where(lower, r * weights, np.nan)[1:, :-1]
        a = np.abs(fingerprint)
        corr_var = np.nanvar(np.nanvar(a, axis=0, ddof=1), ddof=1)

        support = np.abs(coeffs.real).max(axis=1) > tol
        signal = lower & np.outer(support, support)
        corr_mean_signal = np.nanmean(np.abs(r[signal]))
        corr_w_mean_signal = np.nanmean(np.abs(r * weights)[signal])

    return {
        "corr_mean": float(np.nanmean(a)),
        "corr_max": float(np.nanmax(a)),
        "corr_min": float(np.nanmin(a)),
        "corr_var": float(corr_var),
        "corr_w_mean": float(np.nanmean(np.abs(fingerprint_w))),
        "corr_full_mean": float(np.nanmean(np.abs(r))),
        "corr_mean_signal": float(corr_mean_signal),
        "corr_w_mean_signal": float(corr_w_mean_signal),
        "support": freqs[support],
        "coeff_mean_abs": np.abs(coeffs).mean(axis=1),
        "coeff_var_abs": np.abs(coeffs).var(axis=1),
        "coeff_mean_real": coeffs.real.mean(axis=1),
        "coeff_var_real": coeffs.real.var(axis=1),
        "coeff_mean_imag": coeffs.imag.mean(axis=1),
        "coeff_var_imag": coeffs.imag.var(axis=1),
        "fingerprint": fingerprint,
        "fingerprint_w": fingerprint_w,
        "fingerprint_rows": freqs[1:],
        "fingerprint_cols": freqs[:-1],
    }


def fcc(
    model: Model,
    n_samples: int,
    seed: int,
    method: str = "covariance",
    weight: bool = False,
    numerical_cap: float = -1,
    prune: bool = False,
    tol: float = 1e-12,
) -> float:
    """
    Computes the FCC with the qml-essentials implementation.

    With `prune`, the fingerprint is restricted to the numerical support of
    the sampled coefficients before averaging, i.e. to frequencies whose
    coefficient exceeds `tol` in magnitude in any sample.

    Parameters
    ----------
    model : Model
        The quantum Fourier model, its parameters are restored afterwards.
    n_samples : int
        Number of parameter samples (not scaled).
    seed : int
        Seed of the parameter samples.
    method : str, optional
        Correlation method: pearson, complex_pearson, spearman or covariance.
    weight : bool, optional
        Whether to weight the correlations by the mean coefficient magnitudes.
    numerical_cap : float, optional
        Coefficients below this magnitude are set to zero, disabled if negative.
    prune : bool, optional
        Whether to drop the correlations of numerically zero coefficients.
    tol : float, optional
        Magnitude below which a coefficient counts as numerically zero.

    Returns
    -------
    float
        The FCC.
    """
    params = model.params
    fingerprint, _, (row_coeffs, col_coeffs) = FCC.get_fourier_fingerprint(
        model=model,
        n_samples=n_samples,
        random_key=jax.random.PRNGKey(seed),
        method=method,
        weight=weight,
        numerical_cap=numerical_cap,
    )
    model.params = params
    if prune:
        rows = np.abs(row_coeffs).max(axis=1) > tol
        cols = np.abs(col_coeffs).max(axis=1) > tol
        fingerprint = fingerprint[rows][:, cols]
    return float(FCC.calculate_fcc(fingerprint))


def fcc_variants(
    model: Model,
    n_samples: int,
    seed: int,
    method: str = "covariance",
    weight: bool = False,
    numerical_cap: float = -1,
    tol: float = 1e-12,
    chunk_size: int = 1000,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes FCC variants and their null values from one parameter sampling.

    fcc_unpruned and fcc_pruned equal `fcc` without and with `prune`,
    fcc_pearson is the support-restricted real-part Pearson FCC
    (corr_mean_signal of `correlation_stats`). The null values repeat each
    computation after permuting the samples independently per frequency,
    which removes the correlations and leaves the floor of the estimator.
    var_sum is $\\sum_{\\omega} \\mathrm{Var}(c_{\\omega})$ over all frequencies,
    including the negative ones, i.e. the separable term of the expected MSE
    over the parameters; frequencies outside the numerical support add
    numerically zero, so it is also the sum over the support. The spectra are
    evaluated in chunks of `chunk_size` parameter sets to bound the memory.
    With `mask`, only the masked parameters are sampled and the others keep
    their current values.

    Parameters
    ----------
    model : Model
        The quantum Fourier model, its parameters are restored afterwards.
    n_samples : int
        Number of parameter samples (not scaled).
    seed : int
        Seed of the parameter samples and of the permutations.
    method : str, optional
        Correlation method of fcc_unpruned and fcc_pruned.
    weight : bool, optional
        Whether to weight the correlations by the mean coefficient magnitudes.
    numerical_cap : float, optional
        Coefficients below this magnitude are set to zero, disabled if negative.
    tol : float, optional
        Magnitude below which a coefficient counts as numerically zero.
    chunk_size : int, optional
        Number of parameter sets evaluated at once.
    mask : Optional[np.ndarray], optional
        Boolean mask of the shape of `model.params`, True for the sampled
        parameters; None samples all.

    Returns
    -------
    Dict[str, float]
        fcc_unpruned, fcc_pruned, fcc_pearson, their null values with suffix
        _null, var_sum, and n_support, the number of non-negative frequencies
        in the numerical support.
    """
    params = model.params
    model.initialize_params(jax.random.PRNGKey(seed), repeat=n_samples)
    samples = model.params
    if mask is not None:
        samples = jax.numpy.where(mask, samples, params)
    coeffs = []
    for chunk in np.array_split(np.arange(n_samples), -(-n_samples // chunk_size)):
        c, freqs = Coefficients.get_spectrum(
            model,
            shift=True,
            trim=True,
            numerical_cap=numerical_cap,
            params=samples[chunk],
        )
        coeffs.append(np.asarray(c))
    model.params = params

    coeffs = np.concatenate(coeffs, axis=-1).reshape(-1, n_samples)
    var_sum = float(coeffs.var(axis=1, ddof=1).sum())
    # the non-negative frequencies in the order of `fcc`
    pos = FCC._calculate_mask(freqs)
    coeffs = coeffs[pos]
    freqs = np.asarray(FCC._flat_frequencies(freqs))[pos].reshape(len(coeffs), -1)
    support = np.abs(coeffs).max(axis=1) > tol
    lower = np.tri(len(coeffs), k=-1, dtype=bool)

    def variants(c):
        r = np.asarray(FCC._correlate(c.T, method=method))
        if weight:
            m = np.abs(c.mean(axis=1))
            r = r * m[:, None] * m[None, :]
        a = np.where(lower, np.abs(r), np.nan)
        return {
            "fcc_unpruned": float(np.nanmean(a)),
            "fcc_pruned": float(np.nanmean(a[support][:, support])),
            "fcc_pearson": correlation_stats(c, freqs, tol)["corr_mean_signal"],
        }

    null = np.random.default_rng(seed).permuted(coeffs, axis=1)
    return {
        **variants(coeffs),
        **{f"{k}_null": v for k, v in variants(null).items()},
        "var_sum": var_sum,
        "n_support": int(support.sum()),
    }


def numerical_support(
    model: Model, seed: int, n_samples: int = 10, tol: float = 1e-12
) -> np.ndarray:
    """
    Determines the frequencies with a coefficient that is not numerically zero.

    Parameters
    ----------
    model : Model
        The quantum Fourier model, its parameters are restored afterwards.
    seed : int
        Seed of the parameter samples.
    n_samples : int, optional
        Number of random parameter sets.
    tol : float, optional
        Magnitude below which a coefficient counts as numerically zero.

    Returns
    -------
    np.ndarray
        Boolean mask of shape `model.degree` over the ascending frequencies of
        each input feature, True where the coefficient exceeds `tol` in
        magnitude in any sample.
    """
    params = model.params
    model.initialize_params(jax.random.PRNGKey(seed), repeat=n_samples)
    coeffs, _ = Coefficients.get_spectrum(model, shift=True, params=model.params)
    model.params = params
    return np.abs(np.asarray(coeffs)).max(axis=-1) > tol


def local_fcc(
    model: Model,
    n_samples: int = 5,
    seed: int = 1000,
    tol: float = 1e-12,
    rtol: float = 1e-9,
    mask: Optional[np.ndarray] = None,
) -> Dict:
    """
    Computes the local FCC from the Jacobian of the Fourier coefficients.

    For a model with one input feature, $J = \\partial u / \\partial \\theta$ is
    evaluated at `n_samples` random parameter sets, where $u$ are the
    coordinates of the output in the orthonormal real Fourier basis of the
    numerical support (see `numerical_support`, seeded with `seed`), i.e. the
    normalized cosines of the non-negative and sines of the positive support
    frequencies on the grid. $J J^\\top$ is the covariance of $u$ under small isotropic parameter
    perturbations and $R_{loc}$ its Pearson matrix. The numerical rank $m$ of
    $J$ is the dimension of the coefficients reachable around $\\theta$, and
    $\\mathrm{PR}(R_{loc}) = D_s^2 / \\lVert R_{loc} \\rVert_F^2 \\le m$ for the
    support dimension $D_s$. Coordinates without local variation (Jacobian row
    norm below `rtol` times the largest) yield nan correlations, which are
    skipped. With `mask`, only the masked parameters are sampled and
    differentiated, the others keep their current values.

    Parameters
    ----------
    model : Model
        The quantum Fourier model, its parameters are restored afterwards.
    n_samples : int, optional
        Number of random parameter sets.
    seed : int, optional
        Seed of the parameter samples and of the numerical support.
    tol : float, optional
        Magnitude below which a coefficient counts as numerically zero.
    rtol : float, optional
        Singular values below `rtol` times the largest do not count to the rank.
    mask : Optional[np.ndarray], optional
        Boolean mask of the shape of `model.params`, True for the sampled
        parameters; None samples all.

    Returns
    -------
    Dict
        n_dims ($D_s$), rank ($m$), fcc_local (mean $|r|$ of $R_{loc}$ over
        the strict lower triangle), r2_local (mean $r^2$ over the same pairs)
        and pr_local ($\\mathrm{PR}(R_{loc}) / D_s$) as means over the samples,
        their standard deviations over the samples with suffix _std, and
        fingerprint_local, the mean of $|R_{loc}|$ over the samples, with the
        cosines of the ascending frequencies first and then the sines.
    """
    params = model.params
    freqs = model.frequencies[0][numerical_support(model, seed, tol=tol)]
    model.initialize_params(jax.random.PRNGKey(seed), repeat=n_samples)
    samples = model.params
    if mask is not None:
        samples = jax.numpy.where(mask, samples, params)
    model.params = params

    x = 2 * np.pi * np.arange(model.degree[0]) / model.degree[0]
    basis = np.hstack(
        [np.cos(np.outer(x, freqs[freqs >= 0])), np.sin(np.outer(x, freqs[freqs > 0]))]
    )
    basis /= np.linalg.norm(basis, axis=0)
    jac = jax.jit(
        jax.jacfwd(lambda p: model.apply(params=p, inputs=x[:, None], force_mean=True))
    )

    stats, fingerprints = [], []
    for p in samples:
        J = basis.T @ np.asarray(jac(p[None])).reshape(len(x), -1)
        if mask is not None:
            J = J[:, np.ravel(mask)]
        s = np.linalg.svd(J, compute_uv=False)
        norm = np.linalg.norm(J, axis=1)
        norm = np.where(norm > rtol * norm.max(), norm, np.nan)
        r = J @ J.T / np.outer(norm, norm)
        a = r[np.tri(len(r), k=-1, dtype=bool)]
        n = np.isfinite(norm).sum()
        stats.append(
            [
                (s > rtol * s[0]).sum(),
                np.nanmean(np.abs(a)),
                np.nanmean(a**2),
                n**2 / np.nansum(r**2) / len(r),
            ]
        )
        fingerprints.append(np.abs(r))
    model.params = params

    keys = ["rank", "fcc_local", "r2_local", "pr_local"]
    return {
        "n_dims": len(r),
        **{k: float(v) for k, v in zip(keys, np.mean(stats, axis=0))},
        **{f"{k}_std": float(v) for k, v in zip(keys, np.std(stats, axis=0))},
        "fingerprint_local": np.mean(fingerprints, axis=0),
    }


def expressibility(
    model: Model, n_samples: int = 500, n_bins: int = 75, seed: int = 1000
) -> float:
    """
    Computes the expressibility as KL divergence to the Haar distribution.

    As in the paper, $2^n \\cdot$ `n_samples` parameter pairs and $n \\cdot$ `n_bins`
    histogram bins are used for $n$ qubits and the inputs are zero. The
    fidelities $|\\langle \\psi | \\phi \\rangle|^2$ are computed from state vectors,
    which equals the density matrix fidelity of `Expressibility.state_fidelities`
    for noiseless models at a fraction of the memory.

    Parameters
    ----------
    model : Model
        The quantum Fourier model, its parameters are restored afterwards.
    n_samples : int, optional
        Number of parameter pairs per basis state.
    n_bins : int, optional
        Number of histogram bins per qubit.
    seed : int, optional
        Seed of the parameter samples.

    Returns
    -------
    float
        The KL divergence, lower values mean higher expressibility.
    """
    n_total = 2**model.n_qubits * n_samples
    params = model.params
    model.initialize_params(jax.random.PRNGKey(seed), repeat=2 * n_total)
    states = np.asarray(model(params=model.params, execution_type="state"))
    model.params = params

    fidelities = np.abs(np.sum(states[:n_total].conj() * states[n_total:], axis=1)) ** 2
    bins = np.linspace(0, 1, model.n_qubits * n_bins + 1)
    z = np.histogram(fidelities, bins=bins)[0] / n_total
    _, haar = Expressibility.haar_integral(
        model.n_qubits, n_bins, cache=False, scale=True
    )
    return float(Expressibility.kullback_leibler_divergence(z, haar)[0])
