from typing import Any
import pennylane.numpy as np
import pennylane as qml

import logging

log = logging.getLogger(__name__)


class Circuit:
    def __init__(self):
        pass

    @staticmethod
    def n_params_per_layer(n_qubits: int):
        raise NotImplemented

    @staticmethod
    def get_control_indices(w: np.ndarray):
        raise NotImplemented

    @staticmethod
    def build(self, n_qubits: int, n_layers: int):
        raise NotImplemented

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.build(*args, **kwds)


class Ansaetze:
    def get_available():
        return [
            Ansaetze.Circuit_01,
            Ansaetze.Circuit_5,
            Ansaetze.Circuit_9,
            Ansaetze.Circuit_15,
            Ansaetze.Circuit_18,
            Ansaetze.Circuit_19,
            Ansaetze.No_Entangling,
            Ansaetze.Strongly_Entangling,
            Ansaetze.Hardware_Efficient,
        ]

    class Hardware_Efficient(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int):
            if n_qubits > 1:
                return n_qubits * 3
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 2

        @staticmethod
        def get_control_indices(w: np.ndarray):
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Circuit19 ansatz.

            Length of flattened vector must be n_qubits*3-1
            because for >1 qubits there are three gates

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.RY(w[w_idx], wires=q)
                w_idx += 1
                qml.RZ(w[w_idx], wires=q)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    qml.CZ(wires=[q, q + 1])

    class Circuit_19(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int):
            if n_qubits > 1:
                return n_qubits * 3
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 2

        @staticmethod
        def get_control_indices(w: np.ndarray):
            return w[2::3]

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Circuit19 ansatz.

            Length of flattened vector must be n_qubits*3-1
            because for >1 qubits there are three gates

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.RX(w[w_idx], wires=q)
                w_idx += 1
                qml.RZ(w[w_idx], wires=q)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    qml.CRX(
                        w[w_idx], wires=[n_qubits - q - 1, (n_qubits - q) % n_qubits]
                    )
                    w_idx += 1

    class Circuit_18(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int):
            if n_qubits > 1:
                return n_qubits * 3
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 2

        @staticmethod
        def get_control_indices(w: np.ndarray):
            return w[2::3]

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Circuit18 ansatz.

            Length of flattened vector must be n_qubits*3-1
            because for >1 qubits there are three gates

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.RX(w[w_idx], wires=q)
                w_idx += 1
                qml.RZ(w[w_idx], wires=q)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    qml.CRZ(
                        w[w_idx], wires=[n_qubits - q - 1, (n_qubits - q) % n_qubits]
                    )
                    w_idx += 1

    class Circuit_15(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int):
            if n_qubits > 1:
                return n_qubits * 3
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 2

        @staticmethod
        def get_control_indices(w: np.ndarray):
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Circuit15 ansatz.

            Length of flattened vector must be n_qubits*3-1
            because for >1 qubits there are three gates

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            raise NotImplementedError  # Did not figured out the entangling sequence yet

            w_idx = 0
            for q in range(n_qubits):
                qml.RX(w[w_idx], wires=q)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    qml.CNOT(wires=[n_qubits - q - 1, (n_qubits - q) % n_qubits])

            for q in range(n_qubits):
                qml.RZ(w[w_idx], wires=q)
                w_idx += 1

    class Circuit_9(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int):
            return n_qubits

        @staticmethod
        def get_control_indices(w: np.ndarray):
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Circuit19 ansatz.

            Length of flattened vector must be n_qubits*3-1
            because for >1 qubits there are three gates

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.Hadamard(wires=q)

            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    qml.CZ(wires=[n_qubits - q - 2, n_qubits - q - 1])

            for q in range(n_qubits):
                qml.RX(w[w_idx], wires=q)
                w_idx += 1

    class Circuit_1(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int):
            return n_qubits * 2

        @staticmethod
        def get_control_indices(w: np.ndarray):
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Circuit1 ansatz.

            Length of flattened vector must be n_qubits*2

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.RX(w[w_idx], wires=q)
                w_idx += 1
                qml.RZ(w[w_idx], wires=q)
                w_idx += 1

    class Strongly_Entangling(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int):
            if n_qubits > 1:
                return n_qubits * 6
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 2

        @staticmethod
        def get_control_indices(w: np.ndarray):
            return w[2::3]

        @staticmethod
        def build(w: np.ndarray, n_qubits: int) -> None:
            """
            Creates a StronglyEntanglingLayers ansatz.

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.Rot(w[w_idx], w[w_idx + 1], w[w_idx + 2], wires=q)
                w_idx += 3

            if n_qubits > 1:
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])

            for q in range(n_qubits):
                qml.Rot(w[w_idx], w[w_idx + 1], w[w_idx + 2], wires=q)
                w_idx += 3

            if n_qubits > 1:
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + n_qubits // 2) % n_qubits])

    class No_Entangling(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int):
            return n_qubits * 3

        @staticmethod
        def get_control_indices(w: np.ndarray):
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a circuit without entangling, but with U3 gates on all qubits

            Length of flattened vector must be n_qubits*3

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.Rot(w[w_idx], w[w_idx + 1], w[w_idx + 2], wires=q)
                w_idx += 3
