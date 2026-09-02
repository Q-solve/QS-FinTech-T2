"""
VQE solution for the same mean-variance NSE portfolio optimization
problem defined in portfolio_nse_classical.py.

The problem is identical to the classical and QAOA versions:

    minimize   q * x^T Sigma x - mu^T x
    subject to sum(x) == budget
    x in {0,1}^n

VQE converts the QUBO into an Ising Hamiltonian and uses a variational
quantum circuit (ansatz) together with a classical optimizer to search
for a low-energy solution.

The final VQE solution is compared against the exact classical
NumPyMinimumEigensolver reference.
"""

import numpy as np

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2

from qiskit.circuit.library import TwoLocal

from qiskit_optimization.algorithms import MinimumEigenOptimizer

from portfolio_nse_classical import (
    TICKERS,
    q,
    budget,
    portfolio,
    qp,
    summarize,
    solve_classical,
)


# ---------------------------------------------------------------------------
# 0. VQE configuration
# ---------------------------------------------------------------------------

SEED = 1234

MAXITER = 300

algorithm_globals.random_seed = SEED


# ---------------------------------------------------------------------------
# 1. Convert the QuadraticProgram to a QUBO
# ---------------------------------------------------------------------------

from qiskit_optimization.converters import QuadraticProgramToQubo

converter = QuadraticProgramToQubo()

qubo = converter.convert(qp)


# ---------------------------------------------------------------------------
# 2. Convert QUBO to Ising operator
# ---------------------------------------------------------------------------

ising_op, offset = qubo.to_ising()


# ---------------------------------------------------------------------------
# 3. Build the VQE ansatz
# ---------------------------------------------------------------------------

num_qubits = len(TICKERS)

ansatz = TwoLocal(
    num_qubits=num_qubits,
    rotation_blocks=["ry", "rz"],
    entanglement_blocks="cz",
    entanglement="full",
    reps=2,
    insert_barriers=False,
)


# ---------------------------------------------------------------------------
# 4. Classical optimizer used by VQE
# ---------------------------------------------------------------------------

optimizer = COBYLA(
    maxiter=MAXITER
)


# ---------------------------------------------------------------------------
# 5. Aer estimator
# ---------------------------------------------------------------------------

estimator = EstimatorV2()


# ---------------------------------------------------------------------------
# 6. Build VQE
# ---------------------------------------------------------------------------

vqe = VQE(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=optimizer,
)


# ---------------------------------------------------------------------------
# 7. Create MinimumEigenOptimizer wrapper
# ---------------------------------------------------------------------------

vqe_solver = MinimumEigenOptimizer(vqe)


# ---------------------------------------------------------------------------
# 8. Solve using VQE
# ---------------------------------------------------------------------------

def solve_vqe():
    """
    Solve the NSE portfolio optimization problem using VQE.

    Returns the MinimumEigenOptimizer result.
    """
    return vqe_solver.solve(qp)


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("VQE - NSE PORTFOLIO OPTIMIZATION")
    print("=" * 70)

    print(f"\nNSE tickers: {TICKERS}")
    print(f"Number of assets: {len(TICKERS)}")
    print(f"Budget: {budget}")
    print(f"Risk factor q: {q}")

    print("\nVQE configuration:")
    print(f"  Ansatz: TwoLocal")
    print(f"  Rotation blocks: RY + RZ")
    print(f"  Entanglement: CZ / full")
    print(f"  Ansatz reps: 2")
    print(f"  Optimizer: COBYLA")
    print(f"  Max iterations: {MAXITER}")
    print(f"  Seed: {SEED}")

    print("\nNumber of qubits:", num_qubits)

    print("\n" + "=" * 70)
    print("VQE RESULT")
    print("=" * 70)

    vqe_result = solve_vqe()

    vqe_selection = vqe_result.x
    vqe_value = vqe_result.fval

    vqe_chosen = summarize(
        vqe_selection,
        vqe_value,
        label="VQE"
    )

    print("\nVQE raw result:")
    print("Selection:", vqe_selection)
    print("Objective value:", vqe_value)

    # -----------------------------------------------------------------------
    # Classical exact reference
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLASSICAL REFERENCE")
    print("=" * 70)

    classical_result = solve_classical()

    classical_selection = classical_result.x
    classical_value = classical_result.fval

    classical_chosen = summarize(
        classical_selection,
        classical_value,
        label="Classical (NumPyMinimumEigensolver)"
    )

    # -----------------------------------------------------------------------
    # Compare VQE vs Classical
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("VQE vs CLASSICAL")
    print("=" * 70)

    print(
        f"Classical optimal portfolio : "
        f"{classical_chosen} "
        f"(value {classical_value:.6f})"
    )

    print(
        f"VQE portfolio                : "
        f"{vqe_chosen} "
        f"(value {vqe_value:.6f})"
    )

    match = set(vqe_chosen) == set(classical_chosen)

    gap = vqe_value - classical_value

    print(f"\nMatches classical optimum?   : {match}")
    print(f"Objective gap (VQE - exact) : {gap:.6f}")

    if classical_value != 0:
        relative_gap = (
            abs(vqe_value - classical_value)
            / abs(classical_value)
        ) * 100

        print(
            f"Relative objective gap      : "
            f"{relative_gap:.4f}%"
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
