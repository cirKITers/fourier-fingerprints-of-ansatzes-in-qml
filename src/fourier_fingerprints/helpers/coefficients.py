from typing import Callable, Optional, List, Any
from qml_essentials.coefficients import Coefficients as QMLCoefficients

import pennylane.numpy as np
import pandas as pd
from rich.progress import Progress, Task

import logging

log = logging.getLogger(__name__)


class Coefficients:
    @staticmethod
    def calculate_coefficients(
        model: Callable[[np.ndarray], float], **kwargs: Any
    ) -> np.ndarray:
        """
        Calculate the Fourier coefficients of the given model.

        Args:
            model (Callable[[np.ndarray], float]): The model to calculate the Fourier coefficients for.
                The model should take in a parameter array and input values and return a float.

        Returns:
            np.ndarray: The Fourier coefficients of the model.
        """
        # freeze the model for the specific parameters
        coeffs, freqs = QMLCoefficients.get_spectrum(
            model, shift=True, trim=True, **kwargs
        )

        # reorder coefficients such that [..., c_-1, c_0, c_1, ...]
        # coeffs[: model.degree + 1] = [*coeffs[1 : model.degree + 1], coeffs[0]]
        return coeffs, freqs

    @staticmethod
    def numerical(
        model: Callable,  # type: ignore
        n_samples: int,
        seed: Optional[int] = None,
        control_value: int = None,
        progress: Optional[Progress] = None,
        sample_coeff_task: Optional[Task] = None,
        force_same: bool = False,
        **kwargs: Any,
    ) -> float:
        """
        Calculates the entangling capacity of a given quantum circuit
        using Meyer-Wallach measure.

        Parameters
        ----------
        model : Callable
            Function that models the quantum circuit.
            It must have a `n_qubits` attribute representing the number of qubits.
            It must accept a `params` argument representing the parameters of the circuit.
        samples : int
            Number of overall samples
        seed : Optional[int], optional
            Seed for the random number generator.
        control_value : int, optional
            Value of the controlled rotational gates. Allows to overwrite the
            default control value set during model initialization.
        **kwargs : Any
            Additional keyword arguments for the model function.

        Returns
        -------
        float
            Entangling capacity of the given circuit.
            It is guaranteed to be between 0.0 and 1.0.
        """

        if n_samples > 0:
            # TODO: maybe switch to JAX rng
            rng = np.random.default_rng(seed)
            # create empty samples and fill them later
            param_samples = np.ndarray((n_samples, *model.params.shape))

            for s in range(n_samples):
                if force_same:
                    param_samples[s] = rng.random() * np.ones(model.params.shape)
                else:
                    # sample using the model internal sampler, because it
                    # respects the init. domain and strategy
                    model.initialize_params(rng=rng)
                    param_samples[s] = model.params

                if control_value is not None:
                    indices = model.pqc.get_control_indices(model.n_qubits)
                    # special treatment for the control indices
                    if indices is not None:
                        param_samples[s, :, indices[0] : indices[1] : indices[2]] = (
                            np.ones_like(
                                param_samples[
                                    s, :, indices[0] : indices[1] : indices[2]
                                ]
                            )
                            * control_value
                        )
        else:
            if seed is not None:
                log.warning("Seed is ignored when samples is 0")
            n_samples = 1
            # add another dimension to "simulate" param space
            param_samples = model.params.reshape(n_samples, *model.params.shape)

        # Build a pandas dataframe with the parameters and coefficients as columns
        df = pd.DataFrame(
            columns=[
                *[f"p_{i}" for i in range(len(model.params.flatten()))],
                *[
                    f"c_{i}" if i <= 0 else f"c_+{i}"
                    for i in range(-model.degree, model.degree + 1)
                ],  # symmetric + zero frequency
            ]
        )

        if progress is not None:
            progress.reset(sample_coeff_task)

        for i, param_set in enumerate(param_samples):
            # Re-initialize model, because it triggers new sampling
            model.params = param_set
            coeffs, freqs = QMLCoefficients.get_spectrum(
                model, shift=True, trim=True, **kwargs
            )
            # append the parameters and absolute values of coefficients
            # calculation would raise an error if the imaginary part wouldn't sum up to 0
            df.loc[i] = [*param_set.flatten().tolist(), *np.abs(coeffs).tolist()]
            if progress is not None:
                progress.update(sample_coeff_task, advance=1)

        return df
