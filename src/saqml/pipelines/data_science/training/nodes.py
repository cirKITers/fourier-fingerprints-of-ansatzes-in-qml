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


class Losses:
    @staticmethod
    def mse(prediction, target):
        if type(target) == torch.Tensor and type(prediction) == torch.Tensor:
            return torch.mean((prediction - target) ** 2)
        else:
            return np.mean((prediction - target) ** 2)

    @staticmethod
    def null_loss(prediction, target):
        return 0.0

    @staticmethod
    def kl_divergence(prediction, target):
        var_pred = prediction.var()
        var_target = target.var()
        mean_pred = prediction.mean()
        mean_target = target.mean()

        if type(target) == torch.Tensor and type(prediction) == torch.Tensor:
            return 0.5 * torch.sum(
                torch.log(var_target / var_pred)
                + (var_pred + (mean_pred - mean_target) ** 2) / var_target
                - 1
            )
        else:
            return 0.5 * np.sum(
                np.log(var_target / var_pred)
                + (var_pred + (mean_pred - mean_target) ** 2) / var_target
                - 1
            )

    @staticmethod
    def huber_loss(prediction, target, delta=1.0):
        a = prediction - target
        if type(target) == torch.Tensor and type(prediction) == torch.Tensor:
            # return torch.nn.functional.huber_loss(prediction, target)
            abs_a = torch.abs(a)
            return torch.mean(
                torch.where(abs_a <= delta, 0.5 * a**2, delta * (abs_a - 0.5 * delta))
            )
        else:
            abs_a = np.abs(a)
            return np.mean(
                np.where(abs_a <= delta, 0.5 * a**2, delta * (abs_a - 0.5 * delta))
            )

    @staticmethod
    def wasserstein_distance(prediction, target):
        if type(target) == torch.Tensor and type(prediction) == torch.Tensor:
            target = target.detach().numpy()
            prediction = prediction.detach().numpy()
        return wasserstein_distance(prediction, target)

    @staticmethod
    def anderson_ksamp(prediction, target):
        if type(target) == torch.Tensor and type(prediction) == torch.Tensor:
            target = target.detach().numpy()
            prediction = prediction.detach().numpy()
        return anderson_ksamp([prediction, target]).statistic

    @staticmethod
    def energy_distance(prediction, target):
        if type(target) == torch.Tensor and type(prediction) == torch.Tensor:
            target = target.detach().numpy()
            prediction = prediction.detach().numpy()
        return energy_distance(prediction, target)


def train_model(
    model: Model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    loss_function: List,
    loss_scaler: List,
    noise_params: Dict,
    steps: int,
    learning_rate: float,
    log_entangling: bool,
    log_coefficients: bool,
    convergence_threshold: float,
    convergence_gradient: float,
    convergence_steps: int,
):
    if type(model) == Model:
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

        opt = qml.AdamOptimizer(stepsize=learning_rate)

        def cost(params, targets, **kwargs):
            prediction = model(params=params, **kwargs)
            return lambda_1 * loss_1(targets, prediction) + lambda_2 * loss_2(
                targets, prediction
            )

    else:
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.75, patience=6)
        epochs_before_decay = 4

        def cost(inputs, targets):
            prediction = model(inputs)

            return lambda_1 * loss_1(targets, prediction) + lambda_2 * loss_2(
                targets, prediction
            )

    df_params = pd.DataFrame()
    df_grads = pd.DataFrame()
    df_coeffs = pd.DataFrame()

    loss_1 = getattr(Losses, loss_function[0])
    loss_2 = getattr(Losses, loss_function[1])
    lambda_1 = loss_scaler[0]
    lambda_2 = loss_scaler[1]

    def log_metrics(model, step):
        domain_samples = train_loader.dataset.tensors[0].numpy()
        fourier_series = train_loader.dataset.tensors[1].numpy().flatten()
        if type(model) == Model:
            prediction = model(
                params=model.params,
                inputs=domain_samples,
                noise_params=noise_params,
                execution_type="expval",
                force_mean=True,
            )
        else:
            prediction = model(torch.Tensor(domain_samples)).detach().numpy()

        # scaler 1 is for jets
        mlflow.log_metric(
            "wasserstein_train",
            Losses.wasserstein_distance(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "anderson_ksamp_train",
            Losses.anderson_ksamp(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "energy_distance_train",
            Losses.energy_distance(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "kl_divergence_train",
            Losses.kl_divergence(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "huber_loss_train",
            Losses.huber_loss(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "mse_train",
            Losses.mse(prediction, fourier_series),
            step=step,
        )

        domain_samples = valid_loader.dataset.tensors[0].numpy()
        fourier_series = valid_loader.dataset.tensors[1].numpy().flatten()
        if type(model) == Model:
            prediction = model(
                params=model.params,
                inputs=domain_samples,
                noise_params=noise_params,
                execution_type="expval",
                force_mean=True,
            )
        else:
            prediction = model(torch.Tensor(domain_samples)).detach().numpy()

        mlflow.log_metric(
            "wasserstein_valid",
            Losses.wasserstein_distance(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "anderson_ksamp_valid",
            Losses.anderson_ksamp(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "energy_distance_valid",
            Losses.energy_distance(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "kl_divergence_valid",
            Losses.kl_divergence(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "huber_loss_valid",
            Losses.huber_loss(prediction, fourier_series),
            step=step,
        )
        mlflow.log_metric(
            "mse_valid",
            Losses.mse(prediction, fourier_series),
            step=step,
        )

    log.info(f"Training model for {steps} steps")

    costs = np.zeros(steps)

    for step in track(range(steps), description="Training..", total=steps):
        if type(model) == Model:
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
            if isinstance(domain_samples, torch.Tensor) and type(model) == Model:
                domain_samples = domain_samples.numpy()
            if isinstance(fourier_series, torch.Tensor) and type(model) == Model:
                fourier_series = fourier_series.numpy().flatten()

            if type(model) == Model:
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
            else:
                step_cost_val = cost(domain_samples, fourier_series)
                opt.zero_grad()
                step_cost_val.backward()
                opt.step()

            cost_val += step_cost_val
        cost_val /= len(train_loader)

        if step % epochs_before_decay == 0 and type(model) != Model:
            sched.step(cost_val)

        if type(cost_val) == torch.Tensor:
            cost_val = cost_val.item()

        if log_coefficients and type(model) == Model:
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

    if type(model) == Model:
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
