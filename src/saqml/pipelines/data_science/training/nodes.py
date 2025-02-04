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

from saqml.helpers.coefficients import Coefficients

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
    steps: int,
    learning_rate: float,
    batch_size: int,
    log_entangling: bool,
    log_coefficients: bool,
    convergence_threshold: float,
    convergence_gradient: float,
    convergence_steps: int,
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
    df_coeffs_index_names = ["freq"]
    df_coeffs_index = pd.MultiIndex.from_product(
        [range(model.degree + 1)], names=df_coeffs_index_names
    )
    df_params = pd.DataFrame()
    df_grads = pd.DataFrame()
    df_coeffs = pd.DataFrame()

    opt = qml.AdamOptimizer(stepsize=learning_rate)

    def mse(prediction, target):
        return np.mean((prediction - target) ** 2)

    def cost(params, **kwargs):
        return mse(model(params=params, **kwargs), fourier_series)

    log.info(f"Training model for {steps} steps")

    costs = np.zeros(steps)

    for step in track(range(steps), description="Training..", total=steps):
        if log_entangling:
            with warnings.catch_warnings(action="ignore"):
                ent_cap = Entanglement.meyer_wallach(
                    model=model,
                    n_samples=0,  # disable sampling, use model params
                    seed=None,  # set seed none to disable warnings
                    noise_params=noise_params,
                    cache=False,
                )
            log.debug(f"Entangling capability in step {step}: {ent_cap}")
            mlflow.log_metric("entangling_capability", ent_cap, step)

        # log params and gradients
        df_params = pd.concat(
            [
                df_params,
                pd.DataFrame(
                    {"param": model.params.flatten(), "step": step},
                    index=df_params_index,
                ),
            ]
        )
        df_grads = pd.concat(
            [
                df_grads,
                pd.DataFrame(
                    {"param": model.params.flatten(), "step": step},
                    index=df_grads_index,
                ),
            ]
        )

        # optimization step
        model.params, cost_val = opt.step_and_cost(
            cost,
            model.params,
            inputs=domain_samples,
            noise_params=noise_params,
            cache=False,  # disable caching because currently no gradients are being stored
            execution_type="expval",
            force_mean=True,
        )

        if log_coefficients:
            # log coefficients
            coeffs = Coefficients.calculate_coefficients(model, cache=False)
            df_coeffs = pd.concat(
                [
                    df_coeffs,
                    pd.DataFrame(
                        {
                            "coeffs": np.abs(coeffs[len(coeffs) // 2 :]),
                            "step": step,
                        },
                        index=df_coeffs_index,
                    ),
                ]
            )

        # log cost
        log.debug(f"Cost in step {step}: {cost_val}")
        mlflow.log_metric("mse", cost_val, step)
        costs[step] = cost_val

        # early stopping
        if cost_val < convergence_threshold:
            log.info(
                f"Convergence threshold {convergence_threshold} reached after {step} steps."
            )
            break
        elif (
            step >= convergence_steps
            and np.abs(np.gradient(costs)[step - convergence_steps : step].mean())
            < convergence_gradient
        ):
            log.info(
                f"Convergence gradient {convergence_gradient} reached after {step} steps."
            )
            break

    # Convert indices to columns
    df_params = df_params.rename_axis(df_param_index_names).reset_index()
    df_grads = df_grads.rename_axis(df_grads_index_names).reset_index()
    df_coeffs = df_coeffs.rename_axis(df_coeffs_index_names).reset_index()
    return {
        "model": model,
        "params": df_params,
        "grads": df_grads,
        "coeffs": df_coeffs,
    }
