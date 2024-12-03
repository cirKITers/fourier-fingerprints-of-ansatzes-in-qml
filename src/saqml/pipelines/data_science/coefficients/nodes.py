from saqml.helpers.coefficients import Coefficients
from qml_essentials.model import Model
from qml_essentials.expressibility import Expressibility
import pennylane.numpy as np
from rich.progress import Progress
import dcor
import mlflow

import pandas as pd
from typing import Dict, List

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


def sample_coefficients(
    model: Model, omegas: List[List[float]], samples: int, seed: int, mean: float = 0
):
    rng = np.random.default_rng(seed)
    total_samples = samples * model.params.size
    degree = omegas if isinstance(omegas, int) else len(omegas)

    def pascal_triangle(n):
        triangle = [[1]]
        for line in range(2, n + 1):
            row = [1]  # first entry
            for i in range(1, line - 1):
                row.append(triangle[-1][i - 1] + triangle[-1][i])
            row.append(1)  # last entry
            triangle.append(row)
        triangle = np.array(triangle[-1])
        return triangle / np.linalg.norm(triangle)

    # calculate variances using normalised pascal triangle
    variances = pascal_triangle(2 * degree + 1)

    # calculate positive coefficients including zero
    coeffs_pz = np.zeros((degree + 1, total_samples))
    for i, variance in enumerate(variances[len(variances) // 2 :]):
        coeffs_pz[i] = rng.normal(loc=mean, scale=variance, size=total_samples)

    # mirror the coefficients for the negative spectrum and transpose
    # such that we end up with a shape of [n_samples, 2*degree+1]
    coefficients = np.concatenate(
        (np.flip(coeffs_pz[1:], axis=0), coeffs_pz), axis=0
    ).transpose()

    df = pd.DataFrame(
        columns=[
            *[
                f"c_{i}" if i <= 0 else f"c_+{i}" for i in range(-degree, degree + 1)
            ],  # symmetric + zero frequency
        ]
    )

    for i in range(total_samples):
        df.loc[i] = coefficients[i]

    return df


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
        data = df.to_numpy().transpose()  # -> (n_rvs, n_samples)

        dcor_data = lambda rv: dcor.rowwise(
            dcor.distance_correlation,
            data,
            np.tile(rv, (data.shape[0], 1)),  # repeat over n_rvs
        )

        # TODO: this can get really slow for large n_rvs
        result = np.array([dcor_data(rv) for rv in data])
        return pd.DataFrame(result, index=df.columns, columns=df.columns)
    else:
        raise ValueError(f"Unknown correlation method: {method}")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
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

    mlflow.log_metric("expressibility", np.mean(divergence))

    return {"divergence": divergence}


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
