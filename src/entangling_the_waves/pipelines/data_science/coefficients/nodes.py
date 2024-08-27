from entangling_the_waves.helpers.coefficients import Coefficients
from qml_essentials.model import Model
import pennylane.numpy as np
from rich.progress import Progress, Task
import mlflow

import pandas as pd
from typing import Dict

import logging

log = logging.getLogger(__name__)


def calculate_coefficients(model: Model, samples: int, seed: int, noise_params: Dict):

    with Progress() as progress:
        sample_coeff_task = progress.add_task("Sampling...", total=samples)

        coefficients = Coefficients.numerical(
            model=model,
            samples=samples,
            seed=seed,
            progress=progress,
            sample_coeff_task=sample_coeff_task,
        )

    return coefficients


def correlate(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / (df.max() - df.min())


def sweep_control_values(
    model: Model, samples: int, seed: int, noise_params: Dict, n_control_values: int
):
    coefficients_correlated_control = pd.DataFrame(
        columns=["coeff_mean", "control_value"]
    )

    if model.pqc.get_control_indices(model.n_qubits) is None:
        return coefficients_correlated_control

    with Progress() as progress:
        control_value_it_task = progress.add_task(
            "Iterating control values...", total=n_control_values
        )
        sample_coeff_task = progress.add_task("Sampling...", total=samples)
        # coefficients_correlated_mean = []
        for i, cv in enumerate(np.linspace(0, np.pi, n_control_values, endpoint=True)):
            coefficients = Coefficients.numerical(
                model=model,
                samples=samples,
                seed=seed,
                control_value=cv,
                progress=progress,
                sample_coeff_task=sample_coeff_task,
                noise_params=noise_params,
            )
            df_correlated = correlate(coefficients)
            # df_correlated_normalized = normalize(df_correlated)
            coefficients_correlated = df_correlated.filter(regex="c_.*", axis=0).filter(
                regex="c_.*", axis=1
            )
            coefficients_correlated_control.loc[i] = {
                "coeff_mean": coefficients_correlated.mean().mean(),
                "control_value": cv.item(),
            }
            # coefficients_correlated_mean.append(coefficients.mean().mean())

            progress.advance(control_value_it_task, advance=1)

    return coefficients_correlated_control
