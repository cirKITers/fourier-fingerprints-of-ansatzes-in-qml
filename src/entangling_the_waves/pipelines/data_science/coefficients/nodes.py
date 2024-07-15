from entangling_the_waves.helpers.coefficients import Coefficients
from qml_essentials.model import Model
import pennylane.numpy as np
from rich.progress import Progress, Task

import pandas as pd
from typing import Dict

import logging

log = logging.getLogger(__name__)


def calculate_coefficients(model: Model, samples: int, seed: int, noise_params: Dict):

    coefficients = Coefficients.numerical(
        model=model,
        samples=samples,
        seed=seed,
        inputs=[0],
        noise_params=noise_params,
        cache=False,
    )

    return coefficients


def correlate(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / (df.max() - df.min())


def sweep_control_values(
    model: Model, samples: int, seed: int, noise_params: Dict, n_control_values: int
):
    with Progress() as progress:
        control_value_it_task = progress.add_task(
            "Iterating control values...", total=n_control_values
        )
        sample_coeff_task = progress.add_task("Sampling...", total=samples)
        coefficients_correlated_mean = []
        for cv in np.linspace(0, np.pi, n_control_values, endpoint=True):
            df = Coefficients.numerical(
                model=model,
                samples=samples,
                seed=seed,
                control_value=cv,
                progress=progress,
                sample_coeff_task=sample_coeff_task,
                inputs=None,
                noise_params=noise_params,
                cache=False,
            )
            df_correlated = correlate(df)
            df_correlated_normalized = normalize(df_correlated)
            coefficients = df_correlated_normalized.filter(
                regex="c_-.*", axis=0
            ).filter(regex="c_-.*", axis=1)
            coefficients_correlated_mean.append(coefficients.mean().mean())

            progress.advance(control_value_it_task, advance=1)

        print(coefficients_correlated_mean)
    return coefficients_correlated_mean
