from entangling_the_waves.helpers.entanglement import Entanglement
from entangling_the_waves.helpers.model import Model

import pennylane as qml
import pennylane.numpy as np
import mlflow
from typing import Dict
from rich.progress import track

import logging

log = logging.getLogger(__name__)


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
            samples=0,  # disable sampling, use model params
            inputs=[0],
            noise_params=noise_params,
            cache=False,
            state_vector=True,
        )
        log.debug(f"Entangling capability in epoch {epoch}: {ent_cap}")
        mlflow.log_metric("entangling_capability", ent_cap, epoch)

        model.params, cost_val = opt.step_and_cost(
            cost,
            model.params,
            inputs=domain_samples,
            noise_params=noise_params,
            cache=False,  # disable caching because currently no gradients are being stored
            state_vector=False,
            exp_val=True,
        )

        log.debug(f"Cost in epoch {epoch}: {cost_val}")
        mlflow.log_metric("mse", cost_val, epoch)

    return model
