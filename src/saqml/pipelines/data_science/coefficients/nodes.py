from saqml.helpers.coefficients import Coefficients
from qml_essentials.model import Model
import pennylane.numpy as np
from rich.progress import Progress
import dcor

import pandas as pd
from typing import Dict

import logging

log = logging.getLogger(__name__)


def calculate_coefficients(model: Model, samples: int, seed: int, noise_params: Dict):
    """
    Calculate the Fourier coefficients of the given model.

    Parameters
    ----------
    model : Model
        Model to calculate the coefficients for.
    samples : int
        Number of samples to use for each parameter.
    seed : int
        Seed for the random number generator.
    noise_params : Dict
        Parameters for the noise model.

    Returns
    -------
    np.ndarray
        The Fourier coefficients of the model.
    """
    total_samples = samples * model.params.size
    log.info(f"Total number of samples: {total_samples}")

    with Progress() as progress:
        sample_coeff_task = progress.add_task("Sampling...", total=total_samples)

        coefficients = Coefficients.numerical(
            model=model,
            samples=total_samples,
            seed=seed,
            progress=progress,
            sample_coeff_task=sample_coeff_task,
        )

    return coefficients


def correlate(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Calculate correlation matrix of given dataframe.
    Uses pandas correlation method if available, otherwise uses dcor
    as implemented here: https://github.com/vnmabus/dcor

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to calculate the correlation matrix of.
    method : str
        Correlation method to use. Supported methods are "pearson", "spearman", and
        "dcor".

    Returns
    -------
    pd.DataFrame
        Correlation matrix of the dataframe.

    Raises
    ------
    ValueError
        If the given method is not supported.
    """
    if method == "pearson" or method == "spearman":
        return df.corr(method=method)
    elif method == "dcor":
        result = dcor.rowwise(dcor.distance_correlation, df.to_numpy(), df.to_numpy())
    else:
        raise ValueError(f"Unknown correlation method: {method}")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # return (df - df.min()) / (df.max() - df.min())
    """
    Normalize the given dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to normalize.

    Returns
    -------
    pd.DataFrame
        Normalized dataframe.

    Notes
    -----
    Normalization is done by taking the absolute value of the dataframe.
    """
    return df.abs()


def sweep_control_values(
    model: Model, samples: int, seed: int, noise_params: Dict, n_control_values: int
):
    """
    Sweep over control values and calculate the correlated coefficients.

    Parameters
    ----------
    model : Model
        Model to calculate the coefficients for.
    samples : int
        Number of samples to use for each control value.
    seed : int
        Seed for the random number generator.
    noise_params : Dict
        Parameters for the noise model.
    n_control_values : int
        Number of control values to sweep over.

    Returns
    -------
    pd.DataFrame
        Dataframe with the correlated coefficients for each control value.

    Notes
    -----
    If the model does not have control values, an empty dataframe is returned.
    """
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
