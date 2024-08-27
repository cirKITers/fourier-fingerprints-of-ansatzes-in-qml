from qml_essentials.entanglement import Entanglement
from qml_essentials.model import Model

import pennylane as qml
import pennylane.numpy as np
import mlflow
from typing import Dict
from rich.progress import track
import pandas as pd
import warnings
from typing import List

import logging

from entangling_the_waves.helpers.coefficients import Coefficients

log = logging.getLogger(__name__)


def validate_problem(omegas: List[List[float]], model: Model):
    if model.n_layers == 1 or model.n_qubits == 1:
        if model.degree < len(omegas):
            log.warning(
                f"Model is too small to use {len(omegas)} frequencies. Consider adjusting the model degree."
            )
        elif model.degree > len(omegas):
            log.warning(
                f"Model is too large to use {len(omegas)} frequencies. Consider adjusting the model degree."
            )
    else:
        log.warning("Problem validation not implemented yet.")


def train_model(
    model: Model,
    domain_samples: np.ndarray,
    fourier_series: np.ndarray,
    noise_params: Dict,
    epochs: int,
    learning_rate: float,
    batch_size: int,
):
    # Indices for logging params and gradients
    df_param_index_names = ["layer_dim", "param_dim"]
    df_params_index = pd.MultiIndex.from_product(
        [range(s) for s in model.params.shape], names=df_param_index_names
    )
    df_grads_index_names = ["out_dim", "layer_dim", "param_dim"]
    df_grads_index = pd.MultiIndex.from_product(
        [range(s) for s in (1, *model.params.shape)], names=df_grads_index_names
    )
    df_params = pd.DataFrame()
    df_grads = pd.DataFrame()
    df_coeffs = pd.DataFrame(
        columns=[f"c_{i}" for i in range(-model.degree, model.degree + 1)]
    )

    opt = qml.AdamOptimizer(stepsize=learning_rate)

    def mse(prediction, target):
        return np.mean((prediction - target) ** 2)

    def cost(params, **kwargs):
        return mse(model(params=params, **kwargs), fourier_series)

    log.info(f"Training model for {epochs} epochs")

    for epoch in track(range(epochs), description="Training..", total=epochs):
        with warnings.catch_warnings(action="ignore"):
            ent_cap = Entanglement.meyer_wallach(
                model=model,
                n_samples=0,  # disable sampling, use model params
                seed=None,  # set seed none to disable warnings
                noise_params=noise_params,
                cache=False,
            )
        log.debug(f"Entangling capability in epoch {epoch}: {ent_cap}")
        mlflow.log_metric("entangling_capability", ent_cap, epoch)

        # log params and gradients
        df_params_epoch = pd.DataFrame(
            {"param": model.params.flatten(), "epoch": epoch},
            index=df_params_index,
        )
        df_grads_epoch = pd.DataFrame(
            {"param": model.params.flatten(), "epoch": epoch},
            index=df_grads_index,
        )
        df_params = pd.concat([df_params, df_params_epoch])
        df_grads = pd.concat([df_grads, df_grads_epoch])

        model.params, cost_val = opt.step_and_cost(
            cost,
            model.params,
            inputs=domain_samples,
            noise_params=noise_params,
            cache=False,  # disable caching because currently no gradients are being stored
            execution_type="expval",
            force_mean=True,
        )

        log.debug(f"Cost in epoch {epoch}: {cost_val}")
        mlflow.log_metric("mse", cost_val, epoch)

        control_params = np.array(
            [
                model.pqc.get_control_angles(params, model.n_qubits)
                for params in model.params
            ]
        )
        if control_params.any() != None:
            control_rotation_mean = (
                np.sum(np.abs(control_params) % (2 * np.pi)) / control_params.size
            )

            mlflow.log_metric("control_rotation_mean", control_rotation_mean, epoch)

        # log coefficients
        df_coeffs = pd.concat(
            [df_coeffs, Coefficients.numerical(model, samples=0).filter(regex="c.*")],
            ignore_index=True,
        )

    # Convert indices to columns
    df_params = df_params.rename_axis(df_param_index_names).reset_index()
    df_grads = df_grads.rename_axis(df_grads_index_names).reset_index()
    return {
        "model": model,
        "params": df_params,
        "grads": df_grads,
        "coeffs": df_coeffs,
    }
