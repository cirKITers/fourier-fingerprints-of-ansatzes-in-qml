import numpy as np

from fourier_fingerprints.metrics import correlation_stats


def test_correlation_stats():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    coeffs = np.stack(
        [
            x,
            -x + 1j * np.array([5.0, -3.0, 2.0, 7.0]),  # imaginary part is ignored
            np.array([1.0, -1.0, 1.0, -1.0]),
        ]
    )
    stats = correlation_stats(coeffs, np.arange(3)[:, None])

    r = 2 / np.sqrt(20)  # |r| of the third coefficient with the other two
    assert np.isclose(stats["corr_mean"], (1 + 2 * r) / 3)
    assert np.isclose(stats["corr_max"], 1.0)
    assert np.isclose(stats["corr_min"], r)
    assert np.isclose(stats["corr_w_mean"], (0.75 + 0.5 * r + 0.25 * r) / 3)
    assert np.isclose(stats["corr_full_mean"], (3 + 2 * (1 + 2 * r)) / 9)
    assert stats["fingerprint"].shape == (2, 2)

    # a numerically zero coefficient only enters the noise-inclusive FCC
    stats = correlation_stats(np.vstack([coeffs, 1e-17 * x]), np.arange(4)[:, None])
    assert stats["corr_mean"] > (1 + 2 * r) / 3
    assert np.isclose(stats["corr_mean_signal"], (1 + 2 * r) / 3)
    assert np.isclose(stats["corr_w_mean_signal"], (5 / 6 + 4 / 6 * r + 3 / 6 * r) / 3)
    assert (stats["support"].ravel() == [0, 1, 2]).all()


def test_numerical_support():
    from fourier_fingerprints.metrics import numerical_support
    from fourier_fingerprints.model import create_model

    model = create_model(4, 1, "Hardware_Efficient", encoding=["RY"])
    support = model.frequencies[0][numerical_support(model, 1000)]
    assert (support == model.exact_spectrum(method="dp")[0]).all()


def test_fcc_variants():
    from fourier_fingerprints.metrics import fcc, fcc_variants, numerical_support
    from fourier_fingerprints.model import create_model

    model = create_model(3, 1, "Hardware_Efficient", encoding=["RY"])
    variants = fcc_variants(model, 50, 1000)
    assert np.isclose(variants["fcc_unpruned"], fcc(model, 50, 1000), rtol=1e-12)
    assert np.isclose(
        variants["fcc_pruned"], fcc(model, 50, 1000, prune=True), rtol=1e-12
    )
    support = numerical_support(model, 1000)[model.frequencies[0] >= 0]
    assert variants["n_support"] == support.sum()
