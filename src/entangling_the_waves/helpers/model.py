from typing import Dict, Optional, Tuple, Callable, Union
import pennylane as qml
import pennylane.numpy as np
import hashlib
import os
from kedro.io import AbstractDataset

from entangling_the_waves.helpers.ansaetze import Ansaetze

import logging

log = logging.getLogger(__name__)


class Model:
    """
    A quantum circuit model.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        circuit_type: str,
        data_reupload: bool = True,
        output_qubit: int = 0,
    ) -> None:
        """
        Initialize the quantum circuit model.

        Args:
            n_qubits (int): The number of qubits in the circuit.
            n_layers (int): The number of layers in the circuit.
            circuit_type (str): The type of quantum circuit to use.
                If None, defaults to "no_ansatz".
            data_reupload (bool, optional): Whether to reupload data to the
                quantum device on each measurement. Defaults to True.

        Returns:
            None
        """
        self.n_qubits: int = n_qubits
        self.n_layers: int = n_layers
        self.data_reupload: bool = data_reupload
        self.output_qubit: int = output_qubit
        self.pqc: Callable[[Optional[np.ndarray], int], int] = getattr(
            Ansaetze, circuit_type or "no_ansatz"
        )()

        log.info(f"Using {circuit_type} circuit.")

        if data_reupload:
            impl_n_layers: int = n_layers + 1  # we need L+1 according to Schuld et al.
            self.degree = n_layers * n_qubits
        else:
            impl_n_layers: int = n_layers
            self.degree = 0

        params_shape: Tuple[int, int] = (
            impl_n_layers,
            self.pqc.n_params_per_layer(self.n_qubits),
        )
        self.params: np.ndarray = np.random.uniform(
            0, 2 * np.pi, params_shape, requires_grad=True
        )

        self.dev: qml.Device = qml.device("default.mixed", wires=n_qubits)

        self.circuit: qml.QNode = qml.QNode(self._circuit, self.dev)

        log.debug(self._draw())

    def _iec(
        self,
        inputs: np.ndarray,
        data_reupload: bool = True,
    ) -> None:
        """
        Creates an AngleEncoding using RX gates

        Args:
            inputs (np.ndarray): length of vector must be 1, shape (1,)
            data_reupload (bool, optional): Whether to reupload the data for the IEC
                or not, default is True.

        Returns:
            None
        """
        if data_reupload:
            for q in range(self.n_qubits):
                qml.RX(inputs, wires=q)
        else:
            qml.RX(inputs, wires=0)

    def _circuit(
        self,
        params: np.ndarray,
        inputs: np.ndarray,
        noise_params: Optional[Dict[str, float]] = None,
        state_vector: Optional[bool] = False,
        exp_val: Optional[bool] = False,
    ) -> Union[float, np.ndarray]:
        """
        Creates a circuit with noise.
        This involves, Bit Flip, Phase Flip, Amplitude Damping,
        Phase Damping and Depolarization.
        The Circuit consists of a PQC and IEC in each layer
        with the PQC as specified in the construction of the model.

        Args:
            inputs (np.ndarray): input vector of size 1
            params (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
            noise_params (Optional[Dict[str, float]]): dictionary with noise parameters
                - "BitFlip": float, default = 0.0
                - "PhaseFlip": float, default = 0.0
                - "AmplitudeDamping": float, default = 0.0
                - "PhaseDamping": float, default = 0.0
                - "DepolarizingChannel": float, default = 0.0
            state_vector (bool, optional): Whether to measure the state vector
                instead of the wave function. Defaults to False.

        Returns:
            Union[float, np.ndarray]: Expectation value of PauliZ(0) of the circuit if
                state_vector is False, otherwise the density matrix of all qubits.
        """
        if self.data_reupload:
            n_layers = self.n_layers - 1
        else:
            n_layers = self.n_layers

        for l in range(0, n_layers):
            self.pqc(params[l], self.n_qubits)

            if self.data_reupload or l == 0:
                self._iec(inputs, data_reupload=self.data_reupload)

            if noise_params is not None:
                for q in range(self.n_qubits):
                    qml.BitFlip(noise_params["BitFlip"], wires=q)
                    qml.PhaseFlip(noise_params["PhaseFlip"], wires=q)
                    qml.AmplitudeDamping(noise_params["AmplitudeDamping"], wires=q)
                    qml.PhaseDamping(noise_params["PhaseDamping"], wires=q)
                    qml.DepolarizingChannel(noise_params["Depolarizing"], wires=q)

        if self.data_reupload:
            self.pqc(params[-1], self.n_qubits)

        if state_vector:
            return qml.density_matrix(wires=list(range(self.n_qubits)))
        elif exp_val:
            return qml.expval(qml.PauliZ(self.output_qubit))
        else:
            if self.output_qubit == -1:
                return qml.probs(wires=list(range(self.n_qubits)))
            else:
                return qml.probs(wires=self.output_qubit)

    def _draw(self) -> None:
        return qml.draw(self.circuit)(params=self.params, inputs=[0])

    def __repr__(self) -> str:
        return self._draw()

    def __str__(self) -> str:
        return self._draw()

    def __call__(
        self,
        params: Optional[np.ndarray],
        inputs: np.ndarray,
        noise_params: Optional[Dict[str, float]] = None,
        cache: Optional[bool] = False,
        state_vector: bool = False,
        exp_val: bool = False,
    ) -> np.ndarray:
        """Perform a forward pass of the quantum circuit.

        Args:
            inputs (np.ndarray): input vector of size 1
            params (Optional[np.ndarray], optional): weight vector of size n_layers*(n_qubits*3-1). Defaults to None.
            noise_params (Optional[Dict[str, float]], optional): dictionary with noise parameters. Defaults to None.
            cache (Optional[bool], optional): cache the circuit. Defaults to False.
            state_vector (bool, optional): measure the state vector instead of the wave function. Defaults to False.

        Returns:
            np.ndarray: Expectation value of PauliZ(0) of the circuit.
        """
        # Call forward method which handles the actual caching etc.
        return self._forward(params, inputs, noise_params, cache, state_vector, exp_val)

    def _forward(
        self,
        params: Optional[np.ndarray],
        inputs: np.ndarray,
        noise_params: Optional[Dict[str, float]] = None,
        cache: Optional[bool] = False,
        state_vector: bool = False,
        exp_val: bool = False,
    ) -> np.ndarray:
        """Perform a forward pass of the quantum circuit.

        Args:
            inputs (np.ndarray): The input data.
            params (Optional[np.ndarray], optional): The weights of the quantum
                circuit. If None, uses the current weights of the Instructor instance.
                Defaults to None.
            noise_params (Optional[Dict[str, float]], optional): The noise parameters.
                Defaults to None.
            cache (Optional[bool], optional): Whether to cache the results. Defaults to False.
            state_vector (bool, optional): Whether to return the state vector instead of the
                expectation value. Defaults to False.

        Returns:
            np.ndarray: The output of the quantum circuit.
        """
        # the qasm representation contains the bound parameters, thus it is ok to hash that
        hs = hashlib.md5(
            repr(
                {
                    "n_qubits": self.n_qubits,
                    "n_layers": self.n_layers,
                    "pqc": self.pqc.__class__.__name__,
                    "dru": self.data_reupload,
                    "params": params,
                    "noise_params": noise_params,
                }
            ).encode("utf-8")
        ).hexdigest()

        result = None
        if cache:
            name = f"pqc_{hs}.npy"

            cache_folder = ".cache"
            if not os.path.exists(cache_folder):
                os.mkdir(cache_folder)

            file_path = os.path.join(cache_folder, name)

            if os.path.isfile(file_path):
                result = np.load(file_path)

        if result is None:
            # execute the PQC circuit with the current set of parameters
            result = self.circuit(
                params=params,
                inputs=inputs,
                noise_params=noise_params,
                state_vector=state_vector,
                exp_val=exp_val,
            )

        if cache:
            np.save(file_path, result)

        return result


class ModelDataset(AbstractDataset):
    def __init__(
        self,
        filepath: str,
        n_qubits: int,
        n_layers: int,
        circuit_type: str,
        data_reupload: bool,
    ) -> None:
        self.filepath = filepath
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit_type = circuit_type
        self.data_reupload = data_reupload

    def _save(self, model: Model):
        np.save(self.filepath, model.params)

    def _load(self):
        params = np.load(self.filepath)
        model = Model(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            circuit_type=self.circuit_type,
            data_reupload=self.data_reupload,
        )

        model.params = params
        return model

    def _describe(self):
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "circuit_type": self.circuit_type,
            "data_reupload": self.data_reupload,
        }
