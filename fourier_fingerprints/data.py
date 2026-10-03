"""Fourier series and high energy physics (HEP) datasets."""

from typing import Dict, Tuple

import h5py
import jax
import numpy as np
from qml_essentials.coefficients import Datasets
from qml_essentials.model import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from fourier_fingerprints.metrics import numerical_support


def fourier_series(
    model: Model,
    seed: int,
    coefficients_min: float = 0.0,
    coefficients_max: float = 1.0,
    zero_centered: bool = True,
    prune: bool = False,
    target_power: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a random Fourier series matching the spectrum of `model`.

    Coefficients are drawn uniformly from the disc with radii in
    [`coefficients_min`, `coefficients_max`] and the series is sampled on the
    equidistant grid that resolves the highest frequency of the model. With
    `prune`, coefficients outside the numerical support of the model (see
    `numerical_support`, seeded with `seed`) are set to zero, which preserves
    the conjugate symmetry as the support is symmetric. The series is
    $\\sum_\\omega c_\\omega e^{i \\omega x} / K$ for $K$ coefficients, so its
    power $\\sum_\\omega |c_\\omega / K|^2$ (the mean of $y^2$ on the grid) falls
    as $1 / K$; `target_power` rescales the coefficients to a fixed power.

    Parameters
    ----------
    model : Model
        Model defining the frequencies and the input dimensionality.
    seed : int
        Seed of the coefficients.
    coefficients_min : float, optional
        Minimum radius of the coefficients.
    coefficients_max : float, optional
        Maximum radius of the coefficients.
    zero_centered : bool, optional
        Whether to set the zero frequency coefficient (offset) to zero.
    prune : bool, optional
        Whether to restrict the series to the frequencies the model can express.
    target_power : float, optional
        Power $\\sum_\\omega |c_\\omega / K|^2$ of the series, including the
        offset, after pruning; 0 keeps the drawn coefficients.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Inputs of shape (N, D), targets of shape (N,) and the coefficients with
        one axis per input feature, ordered by ascending frequency.
    """
    x, y, coefficients = Datasets.generate_fourier_series(
        jax.random.PRNGKey(seed),
        model,
        coefficients_min,
        coefficients_max,
        zero_centered,
    )
    if prune:
        coefficients = coefficients * numerical_support(model, seed)
        y = Datasets.calculate_values(
            x.reshape(-1, model.n_input_feat),
            Datasets.construct_frequencies(model),
            coefficients.flatten(),
        )
    if target_power > 0:
        # the series is linear in the coefficients
        power = np.sum(np.abs(coefficients / coefficients.size) ** 2)
        scale = np.sqrt(target_power / power)
        coefficients, y = scale * coefficients, scale * y
    return (
        np.asarray(x).reshape(-1, model.n_input_feat),
        np.asarray(y).reshape(-1),
        np.asarray(coefficients),
    )


def hep_dataset(
    n_events: int = 40000,
    seed: int = 1000,
    path: str = "data/pp-z-to-jets-500K-57246.h5",
) -> Dict[str, np.ndarray]:
    """
    Load and preprocess a $pp \\to Z \\to$ jets dataset.

    The features are the center of mass energy $E_{CM}$ and the energy difference
    $|E^{(1)} - E^{(2)}|$ of the partons, transformed by a uniform quantile
    transformer. The label is the transverse momentum of the leading jet,
    min-max scaled to [-0.5, 0.5]. Both scalings are fitted on the first
    `n_events` events, afterwards events without jets are dropped and the
    remaining ones are split 80/10/10 into train, validation and test (unused).

    Parameters
    ----------
    n_events : int, optional
        Number of events read from the file.
    seed : int, optional
        Seed of the quantile subsampling and train/validation split.
    path : str, optional
        Path to the HDF5 file.

    Returns
    -------
    Dict[str, np.ndarray]
        x_train, y_train, x_valid and y_valid as float32 arrays, and pt_min and
        pt_max to invert the label scaling.
    """
    with h5py.File(path, "r") as f:
        partons = f["partons"][:n_events]  # (events, 2, [E, px, py, pz, id, charge])
        jets = f["jets"][:n_events]  # (events, 5, [E, px, py, pz])

    energy = partons[:, :, 0]
    pz = partons[:, :, 3].sum(axis=1)
    x = np.stack(
        [
            np.sqrt(energy.sum(axis=1) ** 2 - pz**2),
            np.abs(energy[:, 0] - energy[:, 1]),
        ],
        axis=1,
    )
    x = QuantileTransformer(random_state=seed).fit_transform(x)

    pt = np.hypot(jets[:, :, 1], jets[:, :, 2])
    y = pt.max(axis=1)
    pt_min, pt_max = y.min(), y.max()
    y = (y - pt_min) / (pt_max - pt_min) - 0.5

    has_jets = (pt > 0).any(axis=1)
    x_train, x_rest, y_train, y_rest = train_test_split(
        x[has_jets], y[has_jets], test_size=0.2, random_state=seed
    )
    x_valid, _, y_valid, _ = train_test_split(
        x_rest, y_rest, test_size=0.5, random_state=seed
    )

    return {
        "x_train": x_train.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "x_valid": x_valid.astype(np.float32),
        "y_valid": y_valid.astype(np.float32),
        "pt_min": pt_min,
        "pt_max": pt_max,
    }
