from saqml.helpers.coefficients import Coefficients
from qml_essentials.model import Model
import pennylane.numpy as np
import numpy as nnp
from rich.progress import Progress
import itertools

# import dcor

import pandas as pd
from typing import Dict, List

import logging

log = logging.getLogger(__name__)


def calculate_coefficients(
    model: Model,
    n_samples: int,
    seed: int,
    noise_params: Dict,
):
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
    n_params = model.params.size

    def str_sign(num: int):
        return f"{num}" if num < 0 else f"+{num}"

    # Build a pandas dataframe with the parameters and coefficients as columns
    df = pd.DataFrame(
        columns=[
            *[f"p_{i}" for i in range(n_params)],
            *[
                f"c_{'_'.join(str_sign(v) for v in tup)}"
                for tup in itertools.product(
                    *[range(-model.degree, model.degree + 1)] * model.n_input_feat
                )
            ],  # symmetric + zero frequency
        ]
    )

    if n_samples > 0:
        total_samples = int(
            np.power(2, model.n_qubits) * n_samples * model.n_input_feat
        )
        log.info(f"Total number of samples: {total_samples}")
        rng = np.random.default_rng(seed)
        model.initialize_params(rng=rng, repeat=total_samples)
    else:
        total_samples = 1

    coeffs, freqs = Coefficients.calculate_coefficients(
        model, noise_params=noise_params
    )
    log.info(f"Aggregating results..")
    concatenated = np.concatenate(
        [model.params.reshape(-1, total_samples), coeffs], axis=0
    )
    for i, c in enumerate(df.columns):
        df[c] = concatenated[i]

    # for i in range(total_samples):
    #     # append the parameters and absolute values of coefficients
    #     # calculation would raise an error if the imaginary part wouldn't sum up to 0
    #     df.loc[i] = [
    #         *model.params[..., i].flatten(),
    #         *coeffs[..., i].flatten(),
    #     ]

    df = df.astype({f"p_{i}": "float64" for i in range(n_params)})

    return df


def filter_coefficients(df: pd.DataFrame, model: Model) -> pd.DataFrame:
    """
    Filter the given dataframe to only include positive coefficients and the zero frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to filter.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    if len(df.index) == len(df.columns) and np.all(df.index == df.columns):
        return df.filter(regex=f"c(_\+\d+){{{model.n_input_feat}}}", axis=0).filter(
            regex=f"c(_\+\d+){{{model.n_input_feat}}}", axis=1
        )
    else:
        return df.filter(regex=f"c(_\+\d+){{{model.n_input_feat}}}", axis=1)


def sample_coefficients(model: Model, n_samples: int, seed: int, mean: float = 0):
    rng = np.random.default_rng(seed)
    total_samples = int(np.power(2, model.n_qubits) * n_samples)
    log.info(f"Total number of samples: {total_samples}")

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
    variances = pascal_triangle(2 * model.degree + 1)

    log.info(f"Using variances {variances[len(variances) // 2 :]}")

    # calculate positive coefficients including zero
    coeffs_pz = np.zeros((model.degree + 1, total_samples))
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
                f"c_{i}" if i <= 0 else f"c_+{i}"
                for i in range(-model.degree, model.degree + 1)
            ],  # symmetric + zero frequency
        ]
    )

    for i in range(total_samples):
        df.loc[i] = coefficients[i]

    return df


def calculate_decay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the decay of the coefficients in the given dataframe.

    The decay is calculated by taking the mean of the coefficients for each degree.
    The coefficients are filtered to only include positive coefficients and the zero frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the coefficients to calculate the decay of.

    Returns
    -------
    dict
        A dictionary containing the decay of the coefficients.
    """

    coefficients_decay = df.abs().apply(np.mean)

    return coefficients_decay


def weight_coefficients(
    df: pd.DataFrame,
    coefficients_decay: pd.DataFrame,
    weighting="coefficients",
) -> pd.DataFrame:
    """
    Weight the correlated coefficients by their decay.

    The weights are calculated as the product of the decay of the two coefficients
    being correlated. The weights are then normalized to have a maximum value of 1.
    The diagonal of the weights is set to 1.

    Parameters
    ----------
    coefficients_correlated : pd.DataFrame
        Dataframe containing the correlated coefficients.
    coefficients_decay : pd.DataFrame
        Dataframe containing the decay of the coefficients.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the weighted correlated coefficients.
    """
    nc = df.shape[0]
    if weighting == "coefficients_add":
        weights = np.ones((nc, nc))
        coefficients_decay = coefficients_decay / coefficients_decay.max()
        for i in range(nc):
            for j in range(nc):
                weights[i, j] = coefficients_decay.iloc[i] + coefficients_decay.iloc[j]
        np.fill_diagonal(weights, 0)
        weights /= weights.max()
        np.fill_diagonal(weights, 1)
    if weighting == "coefficients_prod":
        weights = np.ones((nc, nc))
        coefficients_decay = coefficients_decay / coefficients_decay.max()
        for i in range(nc):
            for j in range(nc):
                weights[i, j] = np.sqrt(
                    coefficients_decay.iloc[i] * coefficients_decay.iloc[j]
                )
        np.fill_diagonal(weights, 0)
        weights /= weights.max()
        np.fill_diagonal(weights, 1)
    elif weighting == "frequencies":
        weights = np.ones((nc, nc))
        for i in range(nc):
            for j in range(nc):
                weights[i, j] = 1 / np.sqrt((i + 1) * (j + 1))
        np.fill_diagonal(weights, 0)
        weights /= weights.max()
        np.fill_diagonal(weights, 1)
    elif weighting == "linear":
        weights = np.flip(nnp.mgrid[0:nc:1, 0:nc:1].sum(axis=0) / ((nc - 1) * 2))
    else:
        raise NotImplementedError(f"Weighting {weighting} not implemented.")

    df_weighted = df * weights

    return df_weighted


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
        result = df.abs().corr(method=method)
    elif method == "dcor":
        data = df.abs().to_numpy().transpose()  # -> (n_rvs, n_samples)

        raise NotImplementedError()
        # temporarily disabled because of issues with llvm
        # dcor_data = lambda rv: dcor.rowwise(
        #     dcor.distance_correlation,
        #     data,
        #     np.tile(rv, (data.shape[0], 1)),  # repeat over n_rvs
        # )

        # TODO: this can get really slow for large n_rvs
        result = pd.DataFrame(
            np.array([dcor_data(rv) for rv in data]),
            index=df.columns,
            columns=df.columns,
        )
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    return result


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


# def sweep_control_values(
#     model: Model, n_samples: int, seed: int, noise_params: Dict, n_control_values: int
# ):
#     """
#     Sweep over control values and calculate the correlated coefficients.

#     Parameters
#     ----------
#     model : Model
#         Model to calculate the coefficients for.
#     samples : int
#         Number of samples to use for each control value.
#     seed : int
#         Seed for the random number generator.
#     noise_params : Dict
#         Parameters for the noise model.
#     n_control_values : int
#         Number of control values to sweep over.

#     Returns
#     -------
#     pd.DataFrame
#         Dataframe with the correlated coefficients for each control value.

#     Notes
#     -----
#     If the model does not have control values, an empty dataframe is returned.
#     """
#     coefficients_correlated_control = pd.DataFrame(
#         columns=["coeff_mean", "control_value"]
#     )

#     if model.pqc.get_control_indices(model.n_qubits) is None:
#         return coefficients_correlated_control

#     with Progress() as progress:
#         control_value_it_task = progress.add_task(
#             "Iterating control values...", total=n_control_values
#         )
#         sample_coeff_task = progress.add_task("Sampling...", total=n_samples)
#         # coefficients_correlated_mean = []
#         for i, cv in enumerate(np.linspace(0, np.pi, n_control_values, endpoint=True)):

#             coefficients = Coefficients.numerical(
#                 model=model,
#                 n_samples=n_samples,
#                 seed=seed,
#                 control_value=cv,
#                 progress=progress,
#                 sample_coeff_task=sample_coeff_task,
#                 noise_params=noise_params,
#             )
#             df_correlated = correlate(coefficients)
#             # df_correlated_normalized = normalize(df_correlated)
#             coefficients_correlated = df_correlated.filter(regex="c_.*", axis=0).filter(
#                 regex="c_.*", axis=1
#             )
#             coefficients_correlated_control.loc[i] = {
#                 "coeff_mean": coefficients_correlated.mean().mean(),
#                 "coeff_max": coefficients_correlated.max().max(),
#                 "coeff_min": coefficients_correlated.min().min(),
#                 "control_value": cv.item(),
#             }
#             # coefficients_correlated_mean.append(coefficients.mean().mean())

#             progress.advance(control_value_it_task, advance=1)

#     return coefficients_correlated_control
