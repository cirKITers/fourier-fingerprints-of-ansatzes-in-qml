from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze, Circuit

from typing import List, Optional
import pennylane as qml
import pennylane.numpy as np

import logging

log = logging.getLogger(__name__)


class OurAnsaetze(Ansaetze):
    class Bansatz(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            if n_qubits > 1:
                return n_qubits * 3
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 3

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            if n_qubits > 1:
                return [-n_qubits, None, None]
            else:
                return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Circuit19 ansatz.

            Length of flattened vector must be n_qubits*3-1
            because for >1 qubits there are three gates

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.RY(w[w_idx], wires=q)
                w_idx += 1
                qml.RZ(w[w_idx], wires=q)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits // 2):
                    qml.CRX(w[w_idx], wires=[(2 * q), (2 * q + 1)])
                    w_idx += 1

                for q in range((n_qubits - 1) // 2):
                    qml.CRX(w[w_idx], wires=[(2 * q + 1), (2 * q + 2)])
                    w_idx += 1

    class Circuit_XZX(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 3

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a XZX ansatz.

            Length of flattened vector must be n_qubits*2

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.RX(w[w_idx], wires=q)
                w_idx += 1
                qml.RZ(w[w_idx], wires=q)
                w_idx += 1
                qml.RX(w[w_idx], wires=q)
                w_idx += 1

    class Circuit_19_N(Ansaetze.Circuit_19):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            if n_qubits > 1:
                return n_qubits * 6
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 2

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Circuit19 ansatz.

            Length of flattened vector must be n_qubits*3-1
            because for >1 qubits there are three gates

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            n_params_per_layer = Ansaetze.Circuit_19.n_params_per_layer(n_qubits)
            for i in range(2):  # twice the number of params
                Ansaetze.Circuit_19.build(
                    w[i * n_params_per_layer : (i + 1) * n_params_per_layer], n_qubits
                )
                qml.Barrier(wires=range(n_qubits))

    class Circuit_XZX_N(Circuit_XZX):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 6

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a XZX ansatz.

            Length of flattened vector must be n_qubits*2

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            n_params_per_layer = OurAnsaetze.Circuit_XZX.n_params_per_layer(n_qubits)
            for i in range(2):  # twice the number of params
                OurAnsaetze.Circuit_XZX.build(
                    w[i * n_params_per_layer : (i + 1) * n_params_per_layer], n_qubits
                )
                qml.Barrier(wires=range(n_qubits))

    class Bansatz_N(Bansatz):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 6

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Bansatz ansatz.

            Length of flattened vector must be n_qubits*2

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            n_params_per_layer = OurAnsaetze.Bansatz.n_params_per_layer(n_qubits)
            for i in range(2):  # twice the number of params
                OurAnsaetze.Bansatz.build(
                    w[i * n_params_per_layer : (i + 1) * n_params_per_layer], n_qubits
                )
                qml.Barrier(wires=range(n_qubits))

    class Hardware_Efficient_N(Ansaetze.Hardware_Efficient):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 6

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Hardware-Efficient ansatz, as proposed in
            https://arxiv.org/pdf/2309.03279

            Length of flattened vector must be n_qubits*3

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3)
                n_qubits (int): number of qubits
            """
            n_params_per_layer = Ansaetze.Hardware_Efficient.n_params_per_layer(
                n_qubits
            )
            for i in range(2):  # twice the number of params
                Ansaetze.Hardware_Efficient.build(
                    w[i * n_params_per_layer : (i + 1) * n_params_per_layer], n_qubits
                )
                qml.Barrier(wires=range(n_qubits))


def create_model(
    n_qubits: int,
    n_layers: int,
    circuit_type: str,
    data_reupload: bool,
    initialization: str,
    initialization_domain: List[float],
    output_qubit: int,
    seed: int,
) -> Model:
    pqc = getattr(OurAnsaetze, circuit_type or "no_ansatz")

    log.info(
        f"Creating model with {n_qubits} qubits, {n_layers} layers, and {circuit_type} circuit."
    )

    model = Model(
        n_qubits=n_qubits,
        n_layers=n_layers,
        circuit_type=pqc,
        data_reupload=data_reupload,
        output_qubit=output_qubit,
        initialization=initialization,
        initialization_domain=initialization_domain,
        random_seed=seed,
    )
    # model.draw(figure=True)[0].savefig(f"{circuit_type}.png")
    return model


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
