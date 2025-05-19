from qml_essentials.entanglement import Entanglement
from qml_essentials.model import Model
from torch.utils.data import DataLoader
import torch
import pennylane as qml
import pennylane.numpy as np
import mlflow
from typing import Dict
from rich.progress import track
import pandas as pd
import warnings
from typing import List
from scipy.stats import wasserstein_distance, anderson_ksamp, energy_distance

# from torch.nn.functional import kl_div as kl_divergence
# from torch.nn.functional import huber_loss as huber_loss

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


def mse(prediction, target):
    return np.mean((prediction - target) ** 2)


def null_loss(prediction, target):
    return 0.0


def kl_divergence(prediction, target):
    pass
    var_pred = prediction.var()
    var_target = target.var()
    mean_pred = prediction.mean()
    mean_target = target.mean()
    return 0.5 * np.sum(
        np.log(var_target / var_pred)
        + (var_pred + (mean_pred - mean_target) ** 2) / var_target
        - 1
    )


def huber_loss(prediction, target, delta=1.0):
    a = prediction - target
    abs_a = np.abs(a)
    return np.mean(np.where(abs_a <= delta, 0.5 * a**2, delta * (abs_a - 0.5 * delta)))


def train_model(
    model: Model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    noise_params: Dict,
    steps: int,
    learning_rate: float,
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

    loss_1 = kl_divergence  # mse
    lambda_1 = 1
    loss_2 = huber_loss
    lambda_2 = 0.001

    def log_metrics(model, step):
        domain_samples = train_loader.dataset.tensors[0].numpy()
        fourier_series = train_loader.dataset.tensors[1].numpy().flatten()
        prediction = model(
            params=model.params,
            inputs=domain_samples,
            noise_params=noise_params,
            execution_type="expval",
            force_mean=True,
        )

        # scaler 1 is for jets
        mlflow.log_metric(
            "wasserstein_train",
            wasserstein_distance(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "anderson_ksamp_train",
            anderson_ksamp([prediction, fourier_series]).statistic,
            step=step,
        )
        mlflow.log_metric(
            "energy_distance_train",
            energy_distance(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "kl_divergence_train",
            kl_divergence(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "huber_loss_train",
            huber_loss(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "mse_train",
            mse(prediction, fourier_series),
            step=step,
        )

        domain_samples = valid_loader.dataset.tensors[0].numpy()
        fourier_series = valid_loader.dataset.tensors[1].numpy().flatten()
        prediction = model(
            params=model.params,
            inputs=domain_samples,
            noise_params=noise_params,
            execution_type="expval",
            force_mean=True,
        )

        mlflow.log_metric(
            "wasserstein_valid",
            wasserstein_distance(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "anderson_ksamp_valid",
            anderson_ksamp([prediction, fourier_series]).statistic,
            step=step,
        )
        mlflow.log_metric(
            "energy_distance_valid",
            energy_distance(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "kl_divergence_valid",
            kl_divergence(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "huber_loss_valid",
            huber_loss(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "mse_valid",
            mse(prediction, fourier_series),
            step=step,
        )

    def cost(params, targets, **kwargs):
        prediction = model(params=params, **kwargs)

        return lambda_1 * loss_1(targets, prediction) + lambda_2 * loss_2(
            targets, prediction
        )

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

        cost_val = 0
        for domain_samples, fourier_series in train_loader:
            if isinstance(domain_samples, torch.Tensor):
                domain_samples = domain_samples.numpy()
            if isinstance(fourier_series, torch.Tensor):
                fourier_series = fourier_series.numpy().flatten()

            # optimization step
            model.params, step_cost_val = opt.step_and_cost(
                cost,
                model.params,
                inputs=domain_samples,
                targets=fourier_series,
                noise_params=noise_params,
                execution_type="expval",
                force_mean=True,
            )

            cost_val += step_cost_val
        cost_val /= len(train_loader)

        if log_coefficients:
            # log coefficients
            coeffs, _ = Coefficients.calculate_coefficients(model, cache=False)
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

            # try:
            #     fcmse_val = fcmse(
            #         np.abs(coeffs[len(coeffs) // 2 :]),
            #         coeffs_target.coefficients.to_numpy(),
            #     )
            #     mlflow.log_metric("fcmse", fcmse_val, step)
            # except Exception as e:
            #     print(e)

        # log cost
        log.debug(f"Cost in step {step}: {cost_val}")
        # mlflow.log_metric("mse", cost_val, step)
        log_metrics(model, step)

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
