from qml_essentials.model import Model

from typing import List
import numpy as np

import logging

log = logging.getLogger(__name__)


def create_model(
    n_qubits: int,
    n_layers: int,
    circuit_type: str,
    data_reupload: bool,
    output_qubit: int,
    initialization: str,
) -> Model:
    return Model(
        n_qubits=n_qubits,
        n_layers=n_layers,
        circuit_type=circuit_type,
        data_reupload=data_reupload,
        output_qubit=output_qubit,
        initialization=initialization,
    )


def print_model(model: Model):
    return str(model)


def sample_domain(domain: List[float], omegas: List[List[float]]) -> np.ndarray:
    """
    Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.

    Parameters
    ----------
    sidelen : int
        Side length of the grid
    dim : int, optional
        Dimensionality of the grid, by default 2

    Returns
    -------
    np.Tensor
        Grid tensor of shape (sidelen^dim, dim)
    """
    dimensions = 1  # len(omega)

    if isinstance(omegas, int):
        omegas = [o for o in range(omegas)]
    # using the max of all dimensions because we want uniform sampling
    n_d = int(np.ceil(2 * np.max(np.abs(domain)) * np.max(omegas)))

    log.info(f"Using {n_d} data points on {len(omegas)} dimensions")

    tensors = tuple(dimensions * [np.linspace(domain[0], domain[1], num=n_d)])

    return np.meshgrid(*tensors)[0].reshape(-1)  # .reshape(-1, dimensions)


def generate_fourier_series(
    domain_samples: np.ndarray,
    omegas: List[List[float]],
    coefficients: List[List[float]],
) -> np.ndarray:
    """
    Generates the Fourier series representation of a function.

    Parameters
    ----------
    domain_samples : np.ndarray
        Grid of domain samples.
    omega : List[List[float]]
        List of frequencies for each dimension.

    Returns
    -------
    np.ndarray
        Fourier series representation of the function.
    """
    if not isinstance(omegas, list):
        omegas = [o for o in range(omegas)]
    if not isinstance(coefficients, list):
        coefficients = [coefficients for _ in omegas]

    assert len(omegas) == len(
        coefficients
    ), "Number of frequencies and coefficients must match"

    omegas = np.array(omegas)
    coefficients = np.array(coefficients)

    def y(x: np.ndarray) -> float:
        """
        Calculates the Fourier series representation of a function at a given point.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the function.

        Returns
        -------
        float
            Value of the Fourier series representation at the given point.
        """
        return (
            1 / np.linalg.norm(omegas) * np.sum(coefficients * np.cos(omegas.T * x))
        )  # transpose!

    values = np.stack([y(x) for x in domain_samples])

    return values
