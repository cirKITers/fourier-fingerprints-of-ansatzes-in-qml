from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze, Circuit
from qml_essentials.coefficients import Datasets
import qml_essentials.operations as op

import torch
from torch.utils.data import TensorDataset, DataLoader

import jax
import jax.numpy as jnp

from typing import List, Optional, Union, Callable
import pandas as pd
import itertools

from fourier_fingerprints.helpers.hep_dataset import get_data, get_loaders
from fourier_fingerprints.helpers.classical_model import HEPRegressor, set_torch_seed

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
        def get_control_indices(n_qubits: int) -> Optional[jnp.ndarray]:
            if n_qubits > 1:
                return [-n_qubits, None, None]
            else:
                return None

        @staticmethod
        def build(w: jnp.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a Circuit19 ansatz.

            Length of flattened vector must be n_qubits*3-1
            because for >1 qubits there are three gates

            Args:
                w (jnp.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                op.RY(w[w_idx], wires=q)
                w_idx += 1
                op.RZ(w[w_idx], wires=q)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits // 2):
                    op.CRX(w[w_idx], wires=[(2 * q), (2 * q + 1)])
                    w_idx += 1

                for q in range((n_qubits - 1) // 2):
                    op.CRX(w[w_idx], wires=[(2 * q + 1), (2 * q + 2)])
                    w_idx += 1

    class Circuit_YZY(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 3

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[jnp.ndarray]:
            return None

        @staticmethod
        def build(w: jnp.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a YZY ansatz.

            Length of flattened vector must be n_qubits*2

            Args:
                w (jnp.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                op.RY(w[w_idx], wires=q)
                w_idx += 1
                op.RZ(w[w_idx], wires=q)
                w_idx += 1
                op.RY(w[w_idx], wires=q)
                w_idx += 1

    class Circuit_YZY_Entangling(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 3

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[jnp.ndarray]:
            return None

        @staticmethod
        def build(w: jnp.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a YZY ansatz.

            Length of flattened vector must be n_qubits*2

            Args:
                w (jnp.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                op.RY(w[w_idx], wires=q)
                w_idx += 1
                op.RZ(w[w_idx], wires=q)
                w_idx += 1
                op.RY(w[w_idx], wires=q)
                w_idx += 1

            if n_qubits > 1:
                for q1 in range(n_qubits - 1):  # 0..n_qubits-2
                    for q2 in range(q1 + 1, n_qubits):  # q1..n_qubits-1
                        op.CNOT(wires=[q1, q2])

    class Circuit_19_N(Ansaetze.Circuit_19):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            if n_qubits > 1:
                return n_qubits * 6
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 2

        @staticmethod
        def build(w: jnp.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a Circuit19 ansatz.

            Length of flattened vector must be n_qubits*3-1
            because for >1 qubits there are three gates

            Args:
                w (jnp.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            n_params_per_layer = Ansaetze.Circuit_19.n_params_per_layer(n_qubits)
            for i in range(2):  # twice the number of params
                Ansaetze.Circuit_19.build(
                    w[i * n_params_per_layer : (i + 1) * n_params_per_layer],
                    n_qubits,
                    noise_params,
                )
                op.Barrier(wires=range(n_qubits))

    class Circuit_YZY_N(Circuit_YZY):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 6

        @staticmethod
        def build(w: jnp.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a YZY ansatz.

            Length of flattened vector must be n_qubits*2

            Args:
                w (jnp.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            n_params_per_layer = OurAnsaetze.Circuit_YZY.n_params_per_layer(n_qubits)
            for i in range(2):  # twice the number of params
                OurAnsaetze.Circuit_YZY.build(
                    w[i * n_params_per_layer : (i + 1) * n_params_per_layer],
                    n_qubits,
                    noise_params,
                )
                op.Barrier(wires=range(n_qubits))

    class Bansatz_N(Bansatz):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 6

        @staticmethod
        def build(w: jnp.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a Bansatz ansatz.

            Length of flattened vector must be n_qubits*2

            Args:
                w (jnp.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            n_params_per_layer = OurAnsaetze.Bansatz.n_params_per_layer(n_qubits)
            for i in range(2):  # twice the number of params
                OurAnsaetze.Bansatz.build(
                    w[i * n_params_per_layer : (i + 1) * n_params_per_layer],
                    n_qubits,
                    noise_params,
                )
                op.Barrier(wires=range(n_qubits))

    class Hardware_Efficient_N(Ansaetze.Hardware_Efficient):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 6

        @staticmethod
        def build(w: jnp.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a Hardware-Efficient ansatz, as proposed in
            https://arxiv.org/pdf/2309.03279

            Length of flattened vector must be n_qubits*3

            Args:
                w (jnp.ndarray): weight vector of size n_layers*(n_qubits*3)
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
                op.Barrier(wires=range(n_qubits))

    class Circuit_2(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 2

        @staticmethod
        def get_control_indices(n_qubits: int):
            return None

        @staticmethod
        def build(w: jnp.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a multi-layered Circuit19 ansatz.

            Length of flattened vector must be n_qubits*2

            Args:
                w (jnp.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            n_params_per_layer = OurAnsaetze.Circuit_2.n_params_per_layer(n_qubits)

            w_idx = 0
            for i in range(n_qubits):
                op.RX(w[w_idx], wires=i)
                w_idx += 1
                op.RZ(w[w_idx], wires=i)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    op.CNOT(wires=[n_qubits - q - 2, n_qubits - q - 1])

    class Circuit_9_N(Ansaetze.Circuit_9):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 6

        @staticmethod
        def build(w: jnp.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a multi-layered Circuit19 ansatz.

            Length of flattened vector must be n_qubits*3*layer_multiplier

            Args:
                w (jnp.ndarray): weight vector of size n_layers*(n_qubits*3*layer_multiplier)
                n_qubits (int): number of qubits
            """
            n_params_per_layer = Ansaetze.Circuit_9.n_params_per_layer(n_qubits)

            for i in range(6):
                Ansaetze.Circuit_9.build(
                    w[i * n_params_per_layer : (i + 1) * n_params_per_layer],
                    n_qubits,
                    noise_params,
                )
                op.Barrier(wires=range(n_qubits))

    class ML_Bansatz(Bansatz):
        layer_multiplier = 1

        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 3 * OurAnsaetze.ML_Bansatz.layer_multiplier

        @staticmethod
        def build(w: jnp.ndarray, n_qubits: int, layer_multiplier=1, noise_params=None):
            """
            Creates a multi-layered Bansatz ansatz.

            Length of flattened vector must be n_qubits*3*layer_multiplier

            Args:
                w (jnp.ndarray): weight vector of size n_layers*(n_qubits*3*layer_multiplier)
                n_qubits (int): number of qubits
            """
            n_params_per_layer = OurAnsaetze.Bansatz.n_params_per_layer(n_qubits)

            for i in range(OurAnsaetze.ML_Bansatz.layer_multiplier):
                OurAnsaetze.Bansatz.build(
                    w[i * n_params_per_layer : (i + 1) * n_params_per_layer],
                    n_qubits,
                    noise_params,
                )
                op.Barrier(wires=range(n_qubits))

    class ML_Hardware_Efficient(Ansaetze.Hardware_Efficient):
        layer_multiplier = 1

        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 3 * OurAnsaetze.ML_Hardware_Efficient.layer_multiplier

        @staticmethod
        def build(w: jnp.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a multi-layered Hardware-Efficient ansatz

            Length of flattened vector must be n_qubits*3*layer_multiplier

            Args:
                w (jnp.ndarray): weight vector of size n_layers*(n_qubits*3*layer_multiplier)
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
                op.Barrier(wires=range(n_qubits))


def generate_model(
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

    log.info(f"Created quantum model with {model.params.size} trainable parameters.")

    return model


def create_classical_model(
    width: int,
    depth: int,
    seed: int,
) -> Model:
    set_torch_seed(seed)

    model = HEPRegressor(2, width=width, depth=depth)

    log.info(
        f"Created classical model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
    )

    return model


def tikz_model(
    n_qubits: int,
    n_layers: int,
    circuit_type: str,
    output_qubit: int,
) -> Model:
    pqc = getattr(OurAnsaetze, circuit_type or "no_ansatz")

    log.info(
        f"Creating model with {n_qubits} qubits, {n_layers} layers, and {circuit_type} circuit."
    )

    model = Model(
        n_qubits=n_qubits,
        n_layers=n_layers,
        circuit_type=pqc,
        data_reupload=False,
        output_qubit=output_qubit,
    )
    fig = model.draw(figure="mpl")
    fig[0].savefig(f"{circuit_type}.svg")

    fig = model.draw(figure="tikz")
    fig.export(f"{circuit_type}.tex", full_document=False)

    return str(fig)


def print_model(model: Model):
    return str(model)


def generate_fourier_series(
    model: Model,
    coefficients_min: float,
    coefficients_max: float,
    zero_centered: bool,
    seed: int,
) -> jnp.ndarray:
    """
    Generates the Fourier series representation of a function.

    Parameters
    ----------
    domain_samples : jnp.ndarray
        Grid of domain samples.
    omega : List[List[float]]
        List of frequencies for each dimension.

    Returns
    -------
    jnp.ndarray
        Fourier series representation of the function.
    """
    random_key = jax.random.PRNGKey(seed)

    domain_samples, fourier_samples, coefficients = Datasets.generate_fourier_series(
        random_key=random_key,
        model=model,
        coefficients_min=coefficients_min,
        coefficients_max=coefficients_max,
        zero_centered=zero_centered,
    )

    return {
        "domain_samples": domain_samples,
        "fourier_samples": fourier_samples.flatten(),
        "coefficients": coefficients,
    }


def build_fourier_series_dataloader(
    batch_size: int, domain_samples, fourier_samples, coefficients: jnp.ndarray
):
    if batch_size < 1:
        batch_size = domain_samples.shape[0]
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(jnp.array(domain_samples)),
            torch.from_numpy(jnp.array(fourier_samples).squeeze()),
            torch.from_numpy(jnp.array(coefficients).squeeze()),
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
        idx = jnp.argmin(jnp.linalg.norm(x - jnp.array([x1, x2]), axis=1))
        eps = jnp.linalg.norm(x[idx] - jnp.array([x1, x2]))
        return idx, eps

    n_samples = x.shape[0]
    n_freqs: int = 2 * mfs * model.degree + 1
    start, stop, step = 0, 2 * mts * jnp.pi, 2 * jnp.pi / n_freqs
    # Stretch according to the number of frequencies
    bins: jnp.ndarray = jnp.arange(start, stop, step)
    x = x * mts

    N = len(bins)
    y_hat = jnp.zeros([N, N])

    for i in range(N):
        for j in range(N):
            idx, eps = closest_x(bins[i], bins[j])
            y_hat[i, j] = y[idx]
            discretization_error += eps / n_samples

    log.info(f"Discretization error: {discretization_error}")

    Y = jnp.fft.fftn(y_hat)
    Y = jnp.fft.fftshift(Y)

    def str_sign(num: int):
        return f"{num:.2f}" if num < 0 else f"+{num:.2f}"

    freqs = jnp.fft.fftshift(jnp.fft.fftfreq(mts * n_freqs, 1 / n_freqs))

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
