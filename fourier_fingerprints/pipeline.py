"""Fluksio flows for measurements and training.

Each cell runs one flow. Nodes wrap the library functions:

    dev/serve.sh                                    # the engine, once
    fluksio sync fourier_fingerprints/pipeline.py   # upload the flows
    fluksio run fingerprint --no-sync --defaults --circuit_type Circuit_19 --wait

- fingerprint: FCC statistics and fingerprint matrices of an ansatz
- surrogate: the same statistics for the random-coefficient surrogate
- expressibility: KL divergence to the Haar distribution
- train: QFM or MLP training on a Fourier series or the HEP dataset
- encoding: FCC variants and Fourier-series training of the encoding study

Inputs default to 6 qubits, one layer, and 1D Fourier-series training. The
study drivers in dev/ set other inputs.
"""

import io
import math
from pathlib import Path

import fluksio
import numpy as np
import torch
from fluksio import Flow, Port, node

from fourier_fingerprints import metrics
from fourier_fingerprints.data import fourier_series, hep_dataset
from fourier_fingerprints.model import create_model
from fourier_fingerprints.train import train_fourier_series, train_mlp, train_qfm

HEP_DATA = Path(__file__).resolve().parents[1] / "data" / "pp-z-to-jets-500K-57246.h5"
STREAMS = ("cost", "mse_valid", "kl_divergence_valid", "huber_loss_valid")


def jsonable(value):
    """Convert NumPy values to JSON types and non-finite floats to None."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray, list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    return value


def _finite_steps(loop, streams=STREAMS):
    """Yield finite stream metrics per step and return the loop result."""
    while True:
        try:
            step = next(loop)
        except StopIteration as stop:
            return stop.value
        yield {k: step[k] for k in streams if math.isfinite(step[k])}


# the arguments of create_model, required by every node
MODEL_INPUTS = [
    Port("n_qubits", "int", initial=6),
    Port("n_layers", "int", initial=1),
    Port("circuit_type", "str", initial="Hardware_Efficient"),
    Port("encoding", "json", initial=["RY"]),
    Port("encoding_strategy", "str", initial="hamming"),
    Port("seed", "int", initial=1000),
]


@node(
    requires=[*MODEL_INPUTS, Port("n_samples", "int"), Port("tol", "float")],
    provides=[Port("stats", "json")],
    timeout=3600,
)
def fingerprint(*, n_samples, tol, **circuit):
    """Compute FCC statistics from sampled Fourier coefficients."""
    model = create_model(**circuit)
    coeffs, freqs = metrics.paper_coefficients(model, n_samples, circuit["seed"])
    return {"stats": jsonable(metrics.correlation_stats(coeffs, freqs, tol))}


@node(
    requires=[*MODEL_INPUTS, Port("n_samples", "int"), Port("tol", "float")],
    provides=[Port("stats", "json")],
    timeout=3600,
)
def surrogate(*, n_samples, tol, **circuit):
    """Compute FCC statistics for the random-coefficient surrogate."""
    model = create_model(**circuit)
    coeffs, freqs = metrics.random_coefficients(model, n_samples, circuit["seed"])
    return {"stats": jsonable(metrics.correlation_stats(coeffs, freqs, tol))}


@node(
    requires=[*MODEL_INPUTS, Port("n_samples", "int"), Port("n_bins", "int")],
    provides=[Port("expressibility", "float")],
    timeout=3600,
)
def expressibility(*, n_samples, n_bins, **circuit):
    """Expressibility of an ansatz as KL divergence to the Haar distribution."""
    model = create_model(**circuit)
    kl = metrics.expressibility(model, n_samples, n_bins, circuit["seed"])
    return {"expressibility": kl}


@node(
    requires=[
        *MODEL_INPUTS,
        Port("dataset", "str"),
        Port("model", "str"),
        Port("data_seed", "int"),
        Port("n_events", "int"),
        Port("steps", "int"),
        Port("learning_rate", "float"),
        Port("loss_function", "json"),
        Port("loss_scaler", "json"),
        Port("batch_size", "int"),
        Port("width", "int"),
        Port("depth", "int"),
        Port("broadcast_targets", "bool"),
    ],
    provides=[
        *(Port(name, "float", stream=True) for name in STREAMS),
        Port("final_metrics", "json"),
        Port("differences", "artifact"),
    ],
    # the first step also pays for loading the data and the jit compilation
    timeout=3600,
)
def train(
    *,
    dataset,
    model,
    data_seed,
    n_events,
    steps,
    learning_rate,
    loss_function,
    loss_scaler,
    batch_size,
    width,
    depth,
    broadcast_targets,
    **circuit,
):
    """
    Train a QFM (`model` qfm) or MLP (`model` mlp) on a Fourier series
    (`dataset` fourier) or the HEP dataset (`dataset` hep). Fourier-series
    training uses the same data for validation. The MLP supports only HEP data
    and ignores circuit inputs. Returns the best validation metrics and an NPZ
    artifact of prediction errors for both splits (in GeV for HEP).
    """
    qfm = create_model(**circuit)
    if dataset == "fourier":
        x, y, _ = fourier_series(qfm, data_seed)
        data, scale = {"x_train": x, "y_train": y, "x_valid": x, "y_valid": y}, 1.0
    else:
        data = hep_dataset(n_events, data_seed, str(HEP_DATA))
        scale = float(data["pt_max"] - data["pt_min"])
    split = [data[k] for k in ("x_train", "y_train", "x_valid", "y_valid")]

    if model == "qfm":
        params, final = yield from _finite_steps(
            train_qfm(
                qfm,
                *split,
                steps=steps,
                learning_rate=learning_rate,
                loss_function=loss_function,
                loss_scaler=loss_scaler,
            )
        )

        def predict(x):
            return np.asarray(qfm.apply(params=params, inputs=x, force_mean=True))

    else:
        mlp, final = yield from _finite_steps(
            train_mlp(
                *split,
                seed=circuit["seed"],
                width=width,
                depth=depth,
                steps=steps,
                learning_rate=learning_rate,
                batch_size=batch_size,
                loss_function=loss_function,
                loss_scaler=loss_scaler,
                broadcast_targets=broadcast_targets,
            )
        )

        def predict(x):
            with torch.no_grad():
                return mlp(torch.from_numpy(x)).numpy()

    buffer = io.BytesIO()
    np.savez(
        buffer,
        **{
            s: (predict(data[f"x_{s}"]).reshape(-1) - data[f"y_{s}"]) * scale
            for s in ("train", "valid")
        },
    )
    return {
        "final_metrics": jsonable(final),
        "differences": fluksio.save_artifact(buffer.getvalue(), "differences.npz"),
    }


@node(
    requires=[
        *MODEL_INPUTS,
        Port("n_samples", "int"),
        Port("method", "str"),
        Port("weight", "bool"),
        Port("numerical_cap", "float"),
        Port("prune", "bool"),
        Port("tol", "float"),
        Port("steps", "int"),
        Port("learning_rate", "float"),
        Port("unnormalized_target", "bool"),
        Port("target_power", "float"),
        Port("n_trainable", "int"),
    ],
    provides=[Port("train_mse", "float", stream=True), Port("results", "json")],
    timeout=3600,
)
def encoding(
    *,
    n_samples,
    method,
    weight,
    numerical_cap,
    prune,
    tol,
    steps,
    learning_rate,
    unnormalized_target,
    target_power,
    n_trainable,
    **circuit,
):
    """
    Measure FCC variants and optionally train on a matching random Fourier series.

    The target keeps its offset; `prune` restricts it to numerical support, and
    positive `target_power` rescales its power. `seed` controls the model, FCC
    samples, and series. Positive `n_trainable` selects that many parameters
    for both sampling and training; the rest remain fixed. `results` contains
    FCC variants, var_sum, n_support, and n_params. With nonzero `steps`, it
    also contains train_mse, train_fmse, train_nmse (MSE divided by target
    variance), and target_power_actual (mean squared target).
    """
    model = create_model(**circuit)
    seed = circuit["seed"]
    mask = None
    if n_trainable:
        mask = np.zeros(model.params.size, dtype=bool)
        mask[np.random.default_rng(seed).choice(mask.size, n_trainable, False)] = True
        mask = mask.reshape(model.params.shape)
    results = metrics.fcc_variants(
        model, n_samples, seed, method, weight, numerical_cap, tol, mask=mask
    )
    results["n_params"] = model.params.size
    if steps:
        x, y, coefficients = fourier_series(
            model, seed, zero_centered=False, prune=prune, target_power=target_power
        )
        _, final = yield from _finite_steps(
            train_fourier_series(
                model,
                x,
                y,
                coefficients,
                steps,
                learning_rate,
                unnormalized_target,
                mask,
            ),
            ("train_mse",),
        )
        results.update(
            final,
            train_nmse=final["train_mse"] / np.var(y),
            target_power_actual=np.mean(y**2),
        )
    return {"results": jsonable(results)}


fingerprint_flow = Flow(
    "fingerprint",
    title="Fourier fingerprint of an ansatz",
    nodes=[fingerprint],
    inputs=[
        *MODEL_INPUTS,
        Port("n_samples", "int", initial=500),
        Port("tol", "float", initial=1e-12),
    ],
    outputs=["stats"],
)

surrogate_flow = Flow(
    "surrogate",
    title="Fourier fingerprint of the random-coefficient surrogate",
    nodes=[surrogate],
    inputs=[
        *MODEL_INPUTS,
        Port("n_samples", "int", initial=200),
        Port("tol", "float", initial=1e-12),
    ],
    outputs=["stats"],
)

expressibility_flow = Flow(
    "expressibility",
    title="Expressibility of an ansatz",
    nodes=[expressibility],
    inputs=[
        *MODEL_INPUTS,
        Port("n_samples", "int", initial=500),
        Port("n_bins", "int", initial=75),
    ],
    outputs=["expressibility"],
)

train_flow = Flow(
    "train",
    title="Train a QFM or MLP on a Fourier series or the HEP dataset",
    nodes=[train],
    inputs=[
        *MODEL_INPUTS,
        Port("dataset", "str", initial="fourier"),
        Port("model", "str", initial="qfm"),
        Port("data_seed", "int", initial=1000),
        Port("n_events", "int", initial=40000),
        Port("steps", "int", initial=2000),
        Port("learning_rate", "float", initial=0.01),
        Port("loss_function", "json", initial=["mse", "null_loss"]),
        Port("loss_scaler", "json", initial=[1.0, 0.001]),
        Port("batch_size", "int", initial=256),
        Port("width", "int", initial=8),
        Port("depth", "int", initial=2),
        Port("broadcast_targets", "bool", initial=False),
    ],
    outputs=["final_metrics", "differences"],
)

encoding_flow = Flow(
    "encoding",
    title="FCC and Fourier-series training under an encoding strategy",
    nodes=[encoding],
    inputs=[
        *MODEL_INPUTS,
        Port("n_samples", "int", initial=500),
        Port("method", "str", initial="covariance"),
        Port("weight", "bool", initial=False),
        Port("numerical_cap", "float", initial=-1.0),
        Port("prune", "bool", initial=False),
        Port("tol", "float", initial=1e-12),
        Port("steps", "int", initial=3000),
        Port("learning_rate", "float", initial=1e-4),
        Port("unnormalized_target", "bool", initial=False),
        Port("target_power", "float", initial=0.0),
        Port("n_trainable", "int", initial=0),
    ],
    outputs=["results"],
)
