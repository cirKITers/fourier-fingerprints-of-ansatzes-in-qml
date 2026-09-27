from qml_essentials.model import Model
from qml_essentials.expressibility import Expressibility
import mlflow
import jax.numpy as jnp

from typing import Dict, List

import logging

log = logging.getLogger(__name__)


def expressibility(
    model: Model,
    samples: int,
    seed: int,
    n_bins: int,
    input_domain: List[float],
    noise_params: Dict,
):
    log.info("Calculating expressibility...")
    _, _, z_model = Expressibility.state_fidelities(
        seed=seed,
        n_samples=samples,
        n_bins=n_bins,
        n_input_samples=None,
        input_domain=input_domain,
        model=model,
        noise_params=noise_params,
        scale=True,
    )

    log.info("Calculating haar integral...")
    _, y_haar = Expressibility.haar_integral(
        n_qubits=model.n_qubits, n_bins=n_bins, scale=True
    )

    log.info("Calculating divergence...")
    divergence = Expressibility.kullback_leibler_divergence(
        vqc_prob_dist=z_model, haar_dist=y_haar
    )

    mlflow.log_metric("expressibility", jnp.mean(divergence))

    return {"divergence": divergence}
