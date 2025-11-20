"""
Simple Quantum Computing Simulator
===================================
A basic implementation of quantum computing concepts including:
- Qubit representation using state vectors
- Quantum gates (Hadamard, Pauli-X/Y/Z, CNOT, etc.)
- Quantum circuits
- Measurement with probabilistic outcomes

This is an educational implementation to understand quantum computing fundamentals.
"""

import numpy as np
from typing import List, Tuple, Optional


class Qubit:
    """
    Represents a single qubit using a 2D state vector.
    |ψ⟩ = α|0⟩ + β|1⟩ where |α|² + |β|² = 1
    """

    def __init__(self, alpha: complex = 1.0, beta: complex = 0.0):
        """Initialize a qubit with given amplitudes."""
        self.state = np.array([alpha, beta], dtype=complex)
        self._normalize()

    def _normalize(self):
        """Normalize the state vector."""
        norm = np.sqrt(np.sum(np.abs(self.state) ** 2))
        if norm > 0:
            self.state = self.state / norm

    @classmethod
    def zero(cls) -> 'Qubit':
        """Create a qubit in |0⟩ state."""
        return cls(1.0, 0.0)

    @classmethod
    def one(cls) -> 'Qubit':
        """Create a qubit in |1⟩ state."""
        return cls(0.0, 1.0)

    @classmethod
    def plus(cls) -> 'Qubit':
        """Create a qubit in |+⟩ = (|0⟩ + |1⟩)/√2 state."""
        return cls(1.0 / np.sqrt(2), 1.0 / np.sqrt(2))

    @classmethod
    def minus(cls) -> 'Qubit':
        """Create a qubit in |-⟩ = (|0⟩ - |1⟩)/√2 state."""
        return cls(1.0 / np.sqrt(2), -1.0 / np.sqrt(2))

    def probabilities(self) -> Tuple[float, float]:
        """Return probabilities of measuring |0⟩ and |1⟩."""
        prob_0 = np.abs(self.state[0]) ** 2
        prob_1 = np.abs(self.state[1]) ** 2
        return (prob_0, prob_1)

    def measure(self) -> int:
        """Perform a measurement, collapsing the state."""
        prob_0, prob_1 = self.probabilities()
        result = np.random.choice([0, 1], p=[prob_0, prob_1])
        # Collapse the state
        if result == 0:
            self.state = np.array([1.0, 0.0], dtype=complex)
        else:
            self.state = np.array([0.0, 1.0], dtype=complex)
        return result

    def __repr__(self):
        return f"Qubit({self.state[0]:.4f}|0⟩ + {self.state[1]:.4f}|1⟩)"


class QuantumGates:
    """Collection of standard quantum gates."""

    # Pauli gates
    I = np.array([[1, 0], [0, 1]], dtype=complex)  # Identity
    X = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli-X (NOT)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)  # Pauli-Y
    Z = np.array([[1, 0], [0, -1]], dtype=complex)  # Pauli-Z

    # Hadamard gate
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    # Phase gates
    S = np.array([[1, 0], [0, 1j]], dtype=complex)  # S gate (π/2 phase)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # T gate (π/4 phase)

    # CNOT gate (2-qubit)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)

    # SWAP gate (2-qubit)
    SWAP = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)

    @staticmethod
    def Rx(theta: float) -> np.ndarray:
        """Rotation around X-axis by angle theta."""
        return np.array([
            [np.cos(theta/2), -1j * np.sin(theta/2)],
            [-1j * np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)

    @staticmethod
    def Ry(theta: float) -> np.ndarray:
        """Rotation around Y-axis by angle theta."""
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)

    @staticmethod
    def Rz(theta: float) -> np.ndarray:
        """Rotation around Z-axis by angle theta."""
        return np.array([
            [np.exp(-1j * theta/2), 0],
            [0, np.exp(1j * theta/2)]
        ], dtype=complex)


class QuantumCircuit:
    """
    A quantum circuit that can hold multiple qubits and apply gates.
    """

    def __init__(self, num_qubits: int):
        """Initialize a quantum circuit with given number of qubits."""
        self.num_qubits = num_qubits
        # Initialize all qubits to |0⟩
        self.state = np.zeros(2 ** num_qubits, dtype=complex)
        self.state[0] = 1.0  # |00...0⟩
        self.operations = []

    def apply_gate(self, gate: np.ndarray, target: int, control: Optional[int] = None):
        """
        Apply a gate to target qubit(s).

        Args:
            gate: The gate matrix to apply
            target: Target qubit index
            control: Control qubit index (for controlled gates)
        """
        if control is not None:
            # Controlled gate
            full_gate = self._controlled_gate(gate, control, target)
        else:
            # Single qubit gate
            full_gate = self._expand_gate(gate, target)

        self.state = full_gate @ self.state
        self.operations.append((gate, target, control))

    def _expand_gate(self, gate: np.ndarray, target: int) -> np.ndarray:
        """Expand a single-qubit gate to full circuit size."""
        result = np.array([[1]], dtype=complex)
        for i in range(self.num_qubits):
            if i == target:
                result = np.kron(result, gate)
            else:
                result = np.kron(result, QuantumGates.I)
        return result

    def _controlled_gate(self, gate: np.ndarray, control: int, target: int) -> np.ndarray:
        """Create a controlled version of a gate."""
        n = 2 ** self.num_qubits
        result = np.eye(n, dtype=complex)

        for i in range(n):
            # Check if control qubit is |1⟩
            if (i >> (self.num_qubits - 1 - control)) & 1:
                # Apply gate to target qubit
                target_bit = (i >> (self.num_qubits - 1 - target)) & 1
                for j in range(2):
                    if gate[j, target_bit] != 0:
                        # Calculate new state index
                        new_i = i
                        if j != target_bit:
                            new_i ^= (1 << (self.num_qubits - 1 - target))
                        result[new_i, i] = gate[j, target_bit]
                        if new_i != i:
                            result[i, i] = 0

        return result

    def h(self, target: int):
        """Apply Hadamard gate to target qubit."""
        self.apply_gate(QuantumGates.H, target)
        return self

    def x(self, target: int):
        """Apply Pauli-X gate to target qubit."""
        self.apply_gate(QuantumGates.X, target)
        return self

    def y(self, target: int):
        """Apply Pauli-Y gate to target qubit."""
        self.apply_gate(QuantumGates.Y, target)
        return self

    def z(self, target: int):
        """Apply Pauli-Z gate to target qubit."""
        self.apply_gate(QuantumGates.Z, target)
        return self

    def cnot(self, control: int, target: int):
        """Apply CNOT gate with control and target qubits."""
        self.apply_gate(QuantumGates.X, target, control)
        return self

    def measure(self, shots: int = 1000) -> dict:
        """
        Measure all qubits multiple times.

        Args:
            shots: Number of measurements to perform

        Returns:
            Dictionary with measurement outcomes and counts
        """
        probabilities = np.abs(self.state) ** 2
        outcomes = np.random.choice(
            len(self.state),
            size=shots,
            p=probabilities
        )

        results = {}
        for outcome in outcomes:
            # Convert to binary string
            key = format(outcome, f'0{self.num_qubits}b')
            results[key] = results.get(key, 0) + 1

        return dict(sorted(results.items()))

    def get_state_vector(self) -> np.ndarray:
        """Return the current state vector."""
        return self.state.copy()

    def get_probabilities(self) -> dict:
        """Return probabilities for each basis state."""
        probs = np.abs(self.state) ** 2
        return {
            format(i, f'0{self.num_qubits}b'): prob
            for i, prob in enumerate(probs) if prob > 1e-10
        }

    def __repr__(self):
        probs = self.get_probabilities()
        state_str = " + ".join([f"{np.sqrt(p):.4f}|{state}⟩" for state, p in probs.items()])
        return f"QuantumCircuit({self.num_qubits} qubits): {state_str}"


def demo_single_qubit():
    """Demonstrate single qubit operations."""
    print("=" * 50)
    print("Single Qubit Demonstration")
    print("=" * 50)

    # Create a qubit in |0⟩ state
    q = Qubit.zero()
    print(f"\nInitial state: {q}")
    print(f"Probabilities: P(0)={q.probabilities()[0]:.4f}, P(1)={q.probabilities()[1]:.4f}")

    # Apply Hadamard gate
    q.state = QuantumGates.H @ q.state
    print(f"\nAfter Hadamard: {q}")
    print(f"Probabilities: P(0)={q.probabilities()[0]:.4f}, P(1)={q.probabilities()[1]:.4f}")

    # Measure multiple times
    results = {0: 0, 1: 0}
    for _ in range(1000):
        q_temp = Qubit.plus()  # Create fresh superposition
        results[q_temp.measure()] += 1
    print(f"\nMeasurement results (1000 shots): {results}")


def demo_bell_state():
    """Demonstrate creating a Bell state (entanglement)."""
    print("\n" + "=" * 50)
    print("Bell State (Entanglement) Demonstration")
    print("=" * 50)

    # Create a 2-qubit circuit
    qc = QuantumCircuit(2)
    print(f"\nInitial state: {qc}")

    # Apply Hadamard to first qubit
    qc.h(0)
    print(f"After H on qubit 0: {qc}")

    # Apply CNOT with qubit 0 as control, qubit 1 as target
    qc.cnot(0, 1)
    print(f"After CNOT(0,1): {qc}")

    # Measure
    results = qc.measure(1000)
    print(f"\nMeasurement results (1000 shots): {results}")
    print("Note: We only get |00⟩ and |11⟩ - the qubits are entangled!")


def demo_quantum_teleportation():
    """Demonstrate quantum teleportation protocol."""
    print("\n" + "=" * 50)
    print("Quantum Teleportation Demonstration")
    print("=" * 50)

    # Create a 3-qubit circuit
    # Qubit 0: State to teleport
    # Qubit 1: Alice's half of entangled pair
    # Qubit 2: Bob's half of entangled pair

    qc = QuantumCircuit(3)

    # Prepare state to teleport (apply some gates to qubit 0)
    qc.h(0)  # Put qubit 0 in superposition
    qc.apply_gate(QuantumGates.T, 0)  # Apply T gate for interesting state

    original_state = qc.state.copy()
    print(f"\nState to teleport (qubit 0): Applied H and T gates")

    # Create entangled pair between qubits 1 and 2
    qc.h(1)
    qc.cnot(1, 2)
    print("Created Bell pair between qubits 1 and 2")

    # Alice's operations
    qc.cnot(0, 1)
    qc.h(0)
    print("Alice performed CNOT and H")

    # Measure qubits 0 and 1 (in real teleportation, these would determine corrections)
    results = qc.measure(1000)
    print(f"\nMeasurement results: {results}")
    print("In real teleportation, Bob would apply corrections based on Alice's measurements")


def demo_grover_iteration():
    """Demonstrate a single Grover iteration for 2 qubits."""
    print("\n" + "=" * 50)
    print("Grover's Algorithm (2-qubit) Demonstration")
    print("=" * 50)

    # Search for |11⟩
    qc = QuantumCircuit(2)

    # Initialize superposition
    qc.h(0).h(1)
    print(f"\nInitial superposition: {qc.get_probabilities()}")

    # Oracle: Mark |11⟩ (apply CZ)
    # CZ flips the phase of |11⟩
    qc.h(1)
    qc.cnot(0, 1)
    qc.h(1)
    print("Applied oracle (marking |11⟩)")

    # Diffusion operator
    qc.h(0).h(1)
    qc.x(0).x(1)
    qc.h(1)
    qc.cnot(0, 1)
    qc.h(1)
    qc.x(0).x(1)
    qc.h(0).h(1)
    print("Applied diffusion operator")

    print(f"\nFinal probabilities: {qc.get_probabilities()}")
    results = qc.measure(1000)
    print(f"Measurement results (1000 shots): {results}")
    print("Note: |11⟩ should have highest probability!")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#" + " Simple Quantum Computing Simulator ".center(58) + "#")
    print("#" + " " * 58 + "#")
    print("#" * 60)

    demo_single_qubit()
    demo_bell_state()
    demo_quantum_teleportation()
    demo_grover_iteration()

    print("\n" + "=" * 50)
    print("Demonstrations Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
