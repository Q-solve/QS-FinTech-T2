"""
QAOA (Quantum Approximate Optimization Algorithm) solution for the same
mean-variance NSE portfolio optimization problem defined in
portfolio_nse_classical.py.

This file does NOT redefine the data loading, mu/sigma computation, or QUBO
construction - it imports the already-built `qp` (QuadraticProgram),
`portfolio`, `TICKERS`, `budget`, `q`, `print_result`, `summarize`, and
`solve_classical` straight from the classical module, so both approaches are
guaranteed to be solving the exact same problem instance.

QAOA works by:
  1. Converting the QUBO's budget constraint into a penalty term, then
     mapping the resulting unconstrained problem to an Ising Hamiltonian
     (handled internally by MinimumEigenOptimizer / QuadraticProgramToQubo).
  2. Preparing a parameterized quantum circuit (`reps` alternating layers of
     problem-Hamiltonian and mixer-Hamiltonian evolution) and using a
     classical optimizer (COBYLA) to tune those parameters so that
     measuring the circuit yields low-energy (= low objective value)
     bitstrings with high probability.
  3. Sampling the final circuit many times; the most frequent bitstring is
     reported as the QAOA solution.

Because QAOA is a heuristic sampled on a simulator, its result is compared
against the exact classical answer at the end.
"""

from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from portfolio_nse_classical import (
    TICKERS,
    q,
    budget,
    portfolio,
    qp,
    print_result,
    summarize,
    solve_classical,
)


# ---------------------------------------------------------------------------
# 0. QAOA configuration
# ---------------------------------------------------------------------------
SEED = 1234
REPS = 3            # number of QAOA layers (circuit depth knob)
MAXITER = 250        # COBYLA iteration budget

algorithm_globals.random_seed = SEED


# ---------------------------------------------------------------------------
# 1. Build the QAOA solver
# ---------------------------------------------------------------------------
# Aer's SamplerV2 needs circuits transpiled to its supported basis gates
# before it can execute them - a preset pass manager handles that.
backend = AerSimulator()
sampler = SamplerV2()
transpiler = generate_preset_pass_manager(optimization_level=1, backend=backend)

cobyla = COBYLA(maxiter=MAXITER)

qaoa_mes = QAOA(
    sampler=sampler,
    optimizer=cobyla,
    reps=REPS,
    transpiler=transpiler,
)
qaoa_solver = MinimumEigenOptimizer(qaoa_mes)


# ---------------------------------------------------------------------------
# 2. Solve with QAOA
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"NSE tickers (imported from classical module): {TICKERS}")
    print(f"Budget = {budget} of {len(TICKERS)}, risk factor q = {q}")
    print(f"QAOA config: reps={REPS}, optimizer=COBYLA(maxiter={MAXITER}), seed={SEED}")

    print("\n" + "=" * 60)
    print("QAOA RESULT")
    print("=" * 60)
    qaoa_result = qaoa_solver.solve(qp)
    qaoa_selection, qaoa_value = print_result(qaoa_result)
    qaoa_chosen = summarize(qaoa_selection, qaoa_value, label="QAOA")

    # -----------------------------------------------------------------
    # 3. Compare against the exact classical reference
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLASSICAL REFERENCE (for comparison)")
    print("=" * 60)
    classical_result = solve_classical()
    classical_selection, classical_value = print_result(classical_result)
    classical_chosen = summarize(
        classical_selection, classical_value, label="Classical (NumPyMinimumEigensolver)"
    )

    print("\n" + "=" * 60)
    print("QAOA vs CLASSICAL")
    print("=" * 60)
    print(f"Classical optimal portfolio : {classical_chosen}  (value {classical_value:.6f})")
    print(f"QAOA portfolio               : {qaoa_chosen}  (value {qaoa_value:.6f})")
    match = set(qaoa_chosen) == set(classical_chosen)
    gap = qaoa_value - classical_value
    print(f"Matches classical optimum?   : {match}")
    print(f"Objective gap (QAOA - exact) : {gap:.6f}")
