from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze, Circuit
from qml_essentials.coefficients import Coefficients

import torch
from torch.utils.data import TensorDataset, DataLoader

from typing import List, Optional, Union, Callable
import pennylane as qml
import pennylane.numpy as np
import pandas as pd
import itertools

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
    n_freqs: int = 2 * max(omegas) + 1
    n_input_feat = len(omegas)

    start, stop, step = domain[0], domain[1], 2 * np.pi / n_freqs
    # Stretch according to the number of frequencies
    inputs: np.ndarray = np.arange(start, stop, step)

    # permute with input dimensionality
    nd_inputs = np.array(np.meshgrid(*[inputs] * n_input_feat)).T.reshape(
        -1, n_input_feat
    )

    return nd_inputs


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
    mts = 1
    mfs = 1
    rng = np.random.default_rng(seed)
    omegas = np.array(omegas)
    dims = len(omegas)
    frequencies = np.stack(
        np.meshgrid(*[np.linspace(-omega, omega, 2 * omega + 1) for omega in omegas])
    ).T.reshape(-1, dims)

    n_freqs: int = int(2 * mfs * max(omegas) + 1)

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
        coefficients = 1.0 * rng.uniform(
            coefficients_mean - coefficients_variance,
            coefficients_mean + coefficients_variance,
            int(np.ceil(frequencies.shape[0] / 2)),
        ) + 1.0j * rng.uniform(
            coefficients_mean - coefficients_variance,
            coefficients_mean + coefficients_variance,
            int(np.ceil(frequencies.shape[0] / 2)),
        )
    elif coefficients_distribution == "normal":
        coefficients = 1.0 * rng.normal(
            coefficients_mean,
            coefficients_variance,
            int(np.ceil(frequencies.shape[0] / 2)),
        ) + 1.0j * rng.normal(
            coefficients_mean,
            coefficients_variance,
            int(np.ceil(frequencies.shape[0] / 2)),
        )

    coefficients = coefficients.flatten()
    if not offset:
        coefficients[0] = 0.0
    else:
        coefficients[0] = coefficients[0].real
    coefficients = np.concat(
        [np.flip(coefficients[1:]).conjugate(), coefficients],
    )

    # assert (
    #     omegas == coefficients.shape
    # ), "Number of frequencies and coefficients must match"

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
            np.real_if_close(np.sum(coefficients * np.exp(1j * frequencies.dot(x))))
            / coefficients.size
        )

    values = np.stack([y(x) for x in domain_samples])
    coefficients_hat = np.fft.fftshift(
        np.fft.fftn(values.reshape([n_freqs] * dims), axes=list(range(dims)))
    )
    freqs = np.fft.fftshift(np.fft.fftfreq(mts * n_freqs, 1 / n_freqs))

    assert np.allclose(
        coefficients, coefficients_hat.flatten(), atol=1e-6
    ), "Frequencies don't match"

    def str_sign(num: int):
        return f"{num:.2f}" if num < 0 else f"+{num:.2f}"

    # Build a pandas dataframe with the parameters and coefficients as columns
    df = pd.DataFrame(
        columns=[
            *[
                f"c_{'_'.join(str_sign(v) for v in tup)}"
                for tup in itertools.product(*[freqs] * dims)
            ],  # symmetric + zero frequency
        ]
    )
    df.loc[0] = coefficients.flatten()

    return {"fourier_series": values.flatten(), "target": df}


def sample_fourier_series(
    domain_samples: np.ndarray,
    omegas: List[List[float]],
    sample_mean: float = 0.5,
    sample_variance: float = 0.0,
    sample_distribution: Optional[str] = None,
    seed: Optional[int] = 1000,
):
    rng = np.random.default_rng(seed)

    dims = len(omegas)

    mfs = 1
    mts = 1

    n_freqs: int = 2 * mfs * max(omegas) + 1

    if sample_distribution == "uniform":
        values = rng.uniform(
            sample_mean - sample_variance,
            sample_mean + sample_variance,
            (n_freqs,) * dims,
        )
    elif sample_distribution == "normal":
        values = rng.normal(
            sample_mean,
            sample_variance,
            (n_freqs,) * dims,
        )
    else:
        raise ValueError(
            "sample_distribution must be specified if sample_mean is not a list or float"
        )
    Y = np.fft.fftshift(np.fft.fftn(values, axes=list(range(dims))))
    freqs = np.fft.fftshift(np.fft.fftfreq(mts * n_freqs, 1 / n_freqs))

    def str_sign(num: int):
        return f"{num:.2f}" if num < 0 else f"+{num:.2f}"

    # Build a pandas dataframe with the parameters and coefficients as columns
    df = pd.DataFrame(
        columns=[
            *[
                f"c_{'_'.join(str_sign(v) for v in tup)}"
                for tup in itertools.product(*[freqs] * dims)
            ],  # symmetric + zero frequency
        ]
    )
    df.loc[0] = Y.flatten()

    return {"fourier_series": values.flatten(), "target": df}


def get_fourier_dataset(batch_size: int, domain_samples, fourier_series):
    if batch_size < 1:
        batch_size = domain_samples.shape[0]
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(domain_samples),
            torch.from_numpy(fourier_series),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "valid_loader": train_loader,
    }


def get_hep_dataset(
    batch_size: int,
    n_events: int,
    features: List[str],
    scaling_methods: List[str],
    labels: List[str],
    seed: int,
):
    # train_partons_df, train_jets_df, test_partons_df = get_data()
    train_loader, valid_loader, test_loader, parton_scaler, jet_scaler = get_loaders(
        batch_size,
        n_events,
        features,
        scaling_methods=scaling_methods,
        labels=labels,
        seed=seed,
    )

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "scalers": [parton_scaler, jet_scaler],
    }


def calculate_hep_spectrum(data_loader, scalers, model, mts, mfs):
    x = data_loader.dataset.tensors[0]
    y = data_loader.dataset.tensors[1].squeeze()

    discretization_error = 0.0

    def closest_x(x1, x2):
        # find index of pair (x1, x2) that is closest to a tuple in x
        idx = np.argmin(np.linalg.norm(x - np.array([x1, x2]), axis=1))
        eps = np.linalg.norm(x[idx] - np.array([x1, x2]))
        return idx, eps

    n_samples = x.shape[0]
    n_freqs: int = 2 * mfs * model.degree + 1
    start, stop, step = 0, 2 * mts * np.pi, 2 * np.pi / n_freqs
    # Stretch according to the number of frequencies
    bins: np.ndarray = np.arange(start, stop, step)
    x = x * mts

    N = len(bins)
    y_hat = np.zeros([N, N])

    for i in range(N):
        for j in range(N):
            idx, eps = closest_x(bins[i], bins[j])
            y_hat[i, j] = y[idx]
            discretization_error += eps / n_samples

    log.info(f"Discretization error: {discretization_error}")

    Y = np.fft.fftn(y_hat)
    Y = np.fft.fftshift(Y)

    def str_sign(num: int):
        return f"{num:.2f}" if num < 0 else f"+{num:.2f}"

    freqs = np.fft.fftshift(np.fft.fftfreq(mts * n_freqs, 1 / n_freqs))

    # Build a pandas dataframe with the parameters and coefficients as columns
    df = pd.DataFrame(
        columns=[
            *[
                f"c_{'_'.join(str_sign(v) for v in tup)}"
                for tup in itertools.product(*[freqs] * model.n_input_feat)
            ],  # symmetric + zero frequency
        ]
    )
    df.loc[0] = Y.flatten()

    return {"target": df}
