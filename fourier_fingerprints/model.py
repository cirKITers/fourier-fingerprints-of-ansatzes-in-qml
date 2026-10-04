"""Quantum Fourier model construction and custom ansaetze."""

from typing import List, Sequence, Union

import numpy as np
from jaqsi.gates import Gates
from qml_essentials.ansaetze import Block, DeclarativeCircuit, Encoding
from qml_essentials.model import Model
from qml_essentials.topologies import Topology


class Circuit_YZY(DeclarativeCircuit):
    """RY, RZ and RY rotations on every qubit, without entanglement."""

    @classmethod
    def structure(cls):
        return (Block(gate=Gates.RY), Block(gate=Gates.RZ), Block(gate=Gates.RY))


class Circuit_YZY_Entangling(DeclarativeCircuit):
    """Circuit_YZY followed by a CX on every qubit pair (i, j) with i < j."""

    @classmethod
    def structure(cls):
        return (
            *Circuit_YZY.structure(),
            Block(gate=Gates.CX, topology=Topology.all_pairs),
        )


class Circuit_15_Paper(DeclarativeCircuit):
    """
    Circuit_15 wiring from qml-essentials 0.1.35.

    The second CX layer couples qubit $q - 1$ to $q - 2$, while qml-essentials'
    Circuit_15 couples $q - 1$ to $q + 2$ (identical only for 4 qubits).
    """

    @classmethod
    def structure(cls):
        return (
            Block(gate=Gates.RY),
            Block(
                gate=Gates.CX,
                topology=Topology.stairs,
                wrap=True,
                reverse=True,
                mirror=False,
            ),
            Block(gate=Gates.RY),
            Block(
                gate=Gates.CX,
                topology=Topology.stairs,
                offset=-1,
                span=-1,
                wrap=True,
                reverse=False,
                mirror=False,
            ),
        )


ANSAETZE = {
    c.__name__: c for c in (Circuit_YZY, Circuit_YZY_Entangling, Circuit_15_Paper)
}


def create_model(
    n_qubits: int,
    n_layers: int,
    circuit_type: str,
    encoding: Sequence[str] = ("RY",),
    encoding_strategy: str = "hamming",
    data_reupload: bool = True,
    output_qubit: Union[int, List[int]] = -1,
    initialization: str = "random",
    initialization_domain: Sequence[float] = (0.0, 2 * np.pi),
    seed: int = 1000,
) -> Model:
    """
    Creates a quantum Fourier model from plain config values.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of encoding layers (the model uses n_layers + 1 ansatz layers).
    circuit_type : str
        Name of a qml-essentials ansatz or of a custom ansatz in `ANSAETZE`.
    encoding : Sequence[str], optional
        Encoding gate per input feature, e.g. ["RX", "RY"] for 2D inputs.
    encoding_strategy : str, optional
        Encoding strategy: hamming, binary or ternary.
    data_reupload : bool, optional
        Whether to re-upload the inputs in every layer.
    output_qubit : Union[int, List[int]], optional
        Measured qubits, -1 measures all qubits.
    initialization : str, optional
        Parameter initialization strategy.
    initialization_domain : Sequence[float], optional
        Domain of the random parameter initialization.
    seed : int, optional
        Seed of the initial parameters.

    Returns
    -------
    Model
        The quantum Fourier model.
    """
    return Model(
        n_qubits=n_qubits,
        n_layers=n_layers,
        circuit_type=ANSAETZE.get(circuit_type, circuit_type),
        data_reupload=data_reupload,
        encoding=Encoding(encoding_strategy, list(encoding)),
        observables=output_qubit,
        initialization=initialization,
        initialization_domain=list(initialization_domain),
        random_seed=seed,
    )
