from typing import Callable, Optional, List, Any
import pennylane as qml
import pennylane.numpy as np
from functools import partial
import pandas as pd
from rich.progress import track


import logging

log = logging.getLogger(__name__)


class Coefficients:

    @staticmethod
    def numerical(
        model: Callable,  # type: ignore
        samples: int,
        seed: Optional[int] = None,
        control_value: int = None,
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
            Number of samples per qubit.
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

        def calculate_coefficients(model: Callable[[np.ndarray], float]) -> np.ndarray:
            """
            Calculate the Fourier coefficients of the given model.

            Args:
                model (Callable[[np.ndarray], float]): The model to calculate the Fourier coefficients for.
                    The model should take in a parameter array and input values and return a float.

            Returns:
                np.ndarray: The Fourier coefficients of the model.
            """
            partial_circuit = partial(model, model.params, execution_type="expval")

            num_inputs = 1

            coeffs = qml.fourier.coefficients(partial_circuit, num_inputs, model.degree)
            coeffs[: model.degree + 1] = [*coeffs[1 : model.degree + 1], coeffs[0]]
            return coeffs

        if samples > 0:
            # TODO: maybe switch to JAX rng
            rng = np.random.default_rng(seed)
            params = np.ndarray((samples, *model.params.shape))
            for s in range(samples):
                params[s] = rng.uniform(0, 2 * np.pi, size=model.params.shape)

                if control_value is not None:
                    indices = model.pqc.get_control_indices(model.n_qubits)
                    # special treatment for the control indices
                    if indices is not None:
                        params[s, :, indices[0] : indices[1] : indices[2]] = (
                            np.ones_like(
                                params[s, :, indices[0] : indices[1] : indices[2]]
                            )
                            * control_value
                        )
            # params = rng.uniform(0, 2 * np.pi, size=(samples, *model.params.shape))
        else:
            if seed is not None:
                log.warning("Seed is ignored when samples is 0")
            samples = 1
            params = model.params.reshape(1, *model.params.shape)

        # Build a pandas dataframe with the parameters and coefficients as columns
        df = pd.DataFrame(
            columns=[
                *[f"p_{i}" for i in range(len(model.params.flatten()))],
                *[
                    f"c_{i}" for i in range(-model.degree, model.degree + 1)
                ],  # symmetric + zero frequency
            ]
        )

        param_samples = np.random.uniform(
            0, 2 * np.pi, size=(samples, *model.params.shape), requires_grad=True
        )

        for i, params in track(
            enumerate(param_samples),
            description="Sampling..",
            total=samples,
        ):
            model.params = params
            coeffs = calculate_coefficients(model)
            # TODO: currently we're using the abs value -> maybe check if real/imag part has some contrib as well
            df.loc[i] = [*params.flatten().tolist(), *np.abs(coeffs).tolist()]

        return df
