from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze, Circuit

import torch
from torch.utils.data import TensorDataset, DataLoader

from typing import List, Optional, Union, Callable
import pennylane as qml
import pennylane.numpy as np
import pandas as pd

from saqml.helpers.hep_dataset import get_data, get_loaders

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
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
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

    class Circuit_YZY(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 3

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a YZY ansatz.

            Length of flattened vector must be n_qubits*2

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.RY(w[w_idx], wires=q)
                w_idx += 1
                qml.RZ(w[w_idx], wires=q)
                w_idx += 1
                qml.RY(w[w_idx], wires=q)
                w_idx += 1

    class Circuit_YZY_Entangling(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 3

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a YZY ansatz.

            Length of flattened vector must be n_qubits*2

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.RY(w[w_idx], wires=q)
                w_idx += 1
                qml.RZ(w[w_idx], wires=q)
                w_idx += 1
                qml.RY(w[w_idx], wires=q)
                w_idx += 1

            if n_qubits > 1:
                for q1 in range(n_qubits - 1):  # 0..n_qubits-2
                    for q2 in range(q1 + 1, n_qubits):  # q1..n_qubits-1
                        qml.CNOT(wires=[q1, q2])

    class Circuit_19_N(Ansaetze.Circuit_19):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            if n_qubits > 1:
                return n_qubits * 6
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 2

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
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
                    w[i * n_params_per_layer : (i + 1) * n_params_per_layer],
                    n_qubits,
                    noise_params,
                )
                qml.Barrier(wires=range(n_qubits))

    class Circuit_YZY_N(Circuit_YZY):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 6

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a YZY ansatz.

            Length of flattened vector must be n_qubits*2

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            n_params_per_layer = OurAnsaetze.Circuit_YZY.n_params_per_layer(n_qubits)
            for i in range(2):  # twice the number of params
                OurAnsaetze.Circuit_YZY.build(
                    w[i * n_params_per_layer : (i + 1) * n_params_per_layer],
                    n_qubits,
                    noise_params,
                )
                qml.Barrier(wires=range(n_qubits))

    class Bansatz_N(Bansatz):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 6

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
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
                    w[i * n_params_per_layer : (i + 1) * n_params_per_layer],
                    n_qubits,
                    noise_params,
                )
                qml.Barrier(wires=range(n_qubits))

    class Hardware_Efficient_N(Ansaetze.Hardware_Efficient):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 6

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
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
                    w[i * n_params_per_layer : (i + 1) * n_params_per_layer],
                    n_qubits,
                    noise_params,
                )
                qml.Barrier(wires=range(n_qubits))

    class Circuit_2(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 2

        @staticmethod
        def get_control_indices(n_qubits: int):
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a multi-layered Circuit19 ansatz.

            Length of flattened vector must be n_qubits*2

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            n_params_per_layer = OurAnsaetze.Circuit_2.n_params_per_layer(n_qubits)

            w_idx = 0
            for i in range(n_qubits):
                qml.RX(w[w_idx], wires=i)
                w_idx += 1
                qml.RZ(w[w_idx], wires=i)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[n_qubits - q - 2, n_qubits - q - 1])

    class Circuit_9_N(Ansaetze.Circuit_9):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 6

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a multi-layered Circuit19 ansatz.

            Length of flattened vector must be n_qubits*3*layer_multiplier

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3*layer_multiplier)
                n_qubits (int): number of qubits
            """
            n_params_per_layer = Ansaetze.Circuit_9.n_params_per_layer(n_qubits)

            for i in range(6):
                Ansaetze.Circuit_9.build(
                    w[i * n_params_per_layer : (i + 1) * n_params_per_layer],
                    n_qubits,
                    noise_params,
                )
                qml.Barrier(wires=range(n_qubits))

    class ML_Bansatz(Bansatz):
        layer_multiplier = 1

        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 3 * OurAnsaetze.ML_Bansatz.layer_multiplier

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, layer_multiplier=1, noise_params=None):
            """
            Creates a multi-layered Bansatz ansatz.

            Length of flattened vector must be n_qubits*3*layer_multiplier

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3*layer_multiplier)
                n_qubits (int): number of qubits
            """
            n_params_per_layer = OurAnsaetze.Bansatz.n_params_per_layer(n_qubits)

            for i in range(OurAnsaetze.ML_Bansatz.layer_multiplier):
                OurAnsaetze.Bansatz.build(
                    w[i * n_params_per_layer : (i + 1) * n_params_per_layer],
                    n_qubits,
                    noise_params,
                )
                qml.Barrier(wires=range(n_qubits))

    class ML_Hardware_Efficient(Ansaetze.Hardware_Efficient):
        layer_multiplier = 1

        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 3 * OurAnsaetze.ML_Hardware_Efficient.layer_multiplier

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a multi-layered Hardware-Efficient ansatz

            Length of flattened vector must be n_qubits*3*layer_multiplier

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3*layer_multiplier)
                n_qubits (int): number of qubits
            """
            n_params_per_layer = Ansaetze.Hardware_Efficient.n_params_per_layer(
                n_qubits
            )

            for i in range(OurAnsaetze.ML_Hardware_Efficient.layer_multiplier):
                Ansaetze.Hardware_Efficient.build(
                    w[i * n_params_per_layer : (i + 1) * n_params_per_layer],
                    n_qubits,
                    noise_params,
                )
                qml.Barrier(wires=range(n_qubits))


def create_model(
    n_qubits: int,
    n_layers: int,
    circuit_type: str,
    data_reupload: bool,
    encoding: Union[str, Callable, List[str], List[Callable]],
    initialization: str,
    initialization_domain: List[float],
    output_qubit: int,
    mp_threshold: int,
    seed: int,
    layer_multiplier: int,
) -> Model:
    pqc = getattr(OurAnsaetze, circuit_type or "no_ansatz")

    if layer_multiplier > 1:
        pqc.layer_multiplier = layer_multiplier

    log.info(
        f"Creating model with {n_qubits} qubits, {n_layers} layers, and {circuit_type} circuit."
    )

    model = Model(
        n_qubits=n_qubits,
        n_layers=n_layers,
        circuit_type=pqc,
        data_reupload=data_reupload,
        encoding=encoding,
        output_qubit=output_qubit,
        initialization=initialization,
        initialization_domain=initialization_domain,
        mp_threshold=mp_threshold,
        random_seed=seed,
    )

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
        omegas = [o for o in range(omegas + 1)]  # as zero frequency doesn't count
    # using the max of all dimensions because we want uniform sampling
    n_d = int(np.ceil(2 * np.max(np.abs(domain)) * np.max(omegas)))

    log.info(f"Using {n_d} data points on {len(omegas)} dimensions")

    tensors = tuple(dimensions * [np.linspace(domain[0], domain[1], num=n_d)])

    return np.meshgrid(*tensors)[0].reshape(-1)  # .reshape(-1, dimensions)


def generate_fourier_series(
    domain_samples: np.ndarray,
    omegas: List[List[float]],
    coefficients_mean: float = 0.5,
    coefficients_variance: float = 0.0,
    coefficients_distribution: Optional[str] = None,
    offset: bool = True,
    seed: Optional[int] = 1000,
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
    rng = np.random.default_rng(seed)
    if not isinstance(omegas, list):
        omegas = [o for o in range(omegas + 1)]  # zero frequency

    if coefficients_distribution is None:
        if isinstance(coefficients_mean, float):
            coefficients = np.array([coefficients for _ in omegas])
        elif isinstance(coefficients_mean, list):
            coefficients = np.array(coefficients_mean)
        else:
            raise ValueError(
                "coefficients_distribution must be specified if coefficients_mean is not a list or float"
            )
    elif coefficients_distribution == "uniform":
        coefficients = rng.uniform(
            coefficients_mean - coefficients_variance,
            coefficients_mean + coefficients_variance,
            len(omegas),
        )
    elif coefficients_distribution == "normal":
        coefficients = rng.normal(coefficients_mean, coefficients_variance, len(omegas))

    if offset:
        coefficients[0] = 0.0

    assert len(omegas) == len(
        coefficients
    ), "Number of frequencies and coefficients must match"

    omegas = np.array(omegas)

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

    return {
        "fourier_series": values,
        "target": pd.DataFrame({"omegas": omegas, "coefficients": coefficients}),
    }


def get_fourier_dataset(batch_size: int, domain_samples, fourier_series):
    if batch_size < 1:
        batch_size = len(domain_samples)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(domain_samples), torch.from_numpy(fourier_series)
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "valid_loader": train_loader,
    }


def get_hep_dataset(batch_size: int, n_events: int, features: List[str], seed: int):
    # train_partons_df, train_jets_df, test_partons_df = get_data()
    train_loader, valid_loader, test_loader, parton_scaler, jet_scaler = get_loaders(
        batch_size, n_events, features, labels=["leading_pt"]
    )

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "scalers": [parton_scaler, jet_scaler],
    }
