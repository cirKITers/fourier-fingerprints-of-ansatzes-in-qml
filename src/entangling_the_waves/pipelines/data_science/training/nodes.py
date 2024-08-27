from qml_essentials.entanglement import Entanglement
from qml_essentials.model import Model

import pennylane as qml
import pennylane.numpy as np
import mlflow
from typing import Dict
from rich.progress import track
from typing import List

import logging

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
    opt = qml.AdamOptimizer(stepsize=learning_rate)

    def mse(prediction, target):
        return np.mean((prediction - target) ** 2)

    def cost(params, **kwargs):
        return mse(model(params=params, **kwargs), fourier_series)

    log.info(f"Training model for {epochs} epochs")

    for epoch in track(range(epochs), description="Training..", total=epochs):
        ent_cap = Entanglement.meyer_wallach(
            model=model,
            n_samples=0,  # disable sampling, use model params
            noise_params=noise_params,
            cache=False,
        )
        log.debug(f"Entangling capability in epoch {epoch}: {ent_cap}")
        mlflow.log_metric("entangling_capability", ent_cap, epoch)

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

    return model
