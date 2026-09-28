"""Fluksio flows, one per kind of measurement.

Every study cell is one run of one flow, the nodes are thin wrappers around the
plain functions of the library:

    dev/serve.sh                                    # the engine, once
    fluksio sync fourier_fingerprints/pipeline.py   # upload the flows
    fluksio run fingerprint --no-sync --defaults --circuit_type Circuit_19 --wait

- fingerprint: paper FCC statistics and fingerprint matrices of an ansatz
- surrogate: the same statistics for the random-coefficient surrogate
- expressibility: KL divergence to the Haar distribution
- train: QFM or MLP training on a Fourier series or the HEP dataset
- encoding: FCC variants and Fourier-series training of the encoding study

Inputs default to the parameters of the paper runs (6 qubits, one layer, the
1D Fourier-series training), the study drivers in dev/ set everything else.
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
    """Converts numpy values to JSON types, non-finite floats become None."""
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
    """Yields the finite stream values of each step, returns the loop result."""
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
    """Paper FCC statistics of sampled Fourier coefficients."""
    model = create_model(**circuit)
    coeffs, freqs = metrics.paper_coefficients(model, n_samples, circuit["seed"])
    return {"stats": jsonable(metrics.correlation_stats(coeffs, freqs, tol))}


@node(
    requires=[*MODEL_INPUTS, Port("n_samples", "int"), Port("tol", "float")],
    provides=[Port("stats", "json")],
    timeout=3600,
)
def surrogate(*, n_samples, tol, **circuit):
    """Paper FCC statistics of the random-coefficient surrogate."""
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
    Trains a QFM (`model` qfm) or an MLP (`model` mlp) on the Fourier series
    (`dataset` fourier, validated on the training data as in the paper) or on
    the HEP dataset (`dataset` hep). The MLP ignores the circuit inputs and
    only supports the HEP dataset. Returns the minimal validation metrics and
    the differences between prediction and target on the training and
    validation data (in GeV for HEP) as npz artifact.
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
    **circuit,
):
    """
    One cell of the encoding study: the FCC variants of `metrics.fcc_variants`
    and the training on a Fourier series with the spectrum of the model (offset
    kept, restricted to the numerical support with `prune`, rescaled to the
    power `target_power` unless 0), `seed` seeds the model, the FCC samples and
    the series. `results` holds the FCC variants, var_sum, n_support, n_params
    and, unless `steps` is 0, the final train_mse, train_fmse, train_nmse
    (train_mse over the variance of the target) and target_power_actual (the
    mean of the squared target).
    """
    model = create_model(**circuit)
    seed = circuit["seed"]
    results = metrics.fcc_variants(
        model, n_samples, seed, method, weight, numerical_cap, tol
    )
    results["n_params"] = model.params.size
    if steps:
        x, y, coefficients = fourier_series(
            model, seed, zero_centered=False, prune=prune, target_power=target_power
        )
        _, final = yield from _finite_steps(
            train_fourier_series(
                model, x, y, coefficients, steps, learning_rate, unnormalized_target
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
    ],
    outputs=["results"],
)
