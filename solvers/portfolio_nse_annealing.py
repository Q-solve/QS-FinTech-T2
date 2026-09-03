"""
Quantum-annealing solution for the same mean-variance NSE portfolio
optimization problem defined in portfolio_nse_classical.py.

This file does NOT redefine the data loading, mu/sigma computation, or QUBO
construction - it imports the already-built `qp`, `TICKERS`, `budget`, `q`,
`solve_classical`, and `summarize` straight from the classical module, so
all three approaches (classical, QAOA, annealing) are guaranteed to be
solving the exact same problem instance.

SET UP FOR QBRAID
------------------
This script is written to run REAL quantum annealing on D-Wave hardware
when executed inside a qBraid environment, and to fall back to a local
classical simulation everywhere else (e.g. this sandbox, which has no
network path to D-Wave's cloud).

To run this on real D-Wave hardware on qBraid:
  1. Open this notebook/file in a qBraid environment that includes the
     Ocean SDK - e.g. the "D-Wave Ocean" or "qBraid-SDK" environment from
     the Environments panel (pip installs: dwave-ocean-sdk, dimod,
     dwave-system - already listed in requirements.txt alongside this file).
  2. In the qBraid Lab sidebar, open "Quantum Jobs" and toggle it ON for
     D-Wave. This is qBraid's proxy credentialing system: once enabled, all
     dwave.system calls in this notebook are transparently routed through
     qBraid's own D-Wave Leap access, so you do NOT need to run
     `dwave config create` or hold your own Leap API token.
  3. Just run the script / notebook cells top to bottom. `DWaveSampler()`
     will pick up a live QPU (e.g. Advantage_system) automatically; no
     other code changes are needed.

If Quantum Jobs is off, no Ocean SDK is installed, or there's no network
path to D-Wave (e.g. running this outside qBraid), the script automatically
falls back to `neal.SimulatedAnnealingSampler` - D-Wave's own open-source
classical reference implementation of annealing, using the identical
`dimod` BQM, so the rest of the pipeline and the printed comparison against
the classical solver work unchanged either way. Which path was used is
printed clearly at the top of the run.

How annealing (real or simulated) approaches the QUBO:
  1. The QUBO's budget constraint is folded into a penalty term (handled by
     Qiskit's QuadraticProgramToQubo converter), producing an unconstrained
     quadratic objective over binary variables x in {0,1}^n.
  2. That objective is expressed as an Ising-style Binary Quadratic Model:
     linear biases h_i on each variable, quadratic couplings J_ij between
     pairs of variables, plus a constant offset.
  3. On real D-Wave hardware, EmbeddingComposite first "minor-embeds" the
     problem graph onto the QPU's physical qubit topology (chains of
     physical qubits stand in for each logical variable), then the QPU
     anneals: every qubit starts in a high-energy superposition and the
     transverse field is gradually lowered, letting the system relax
     toward low-energy (= low-objective-value) configurations, potentially
     tunneling through energy barriers a purely thermal process would have
     to climb over. The simulated fallback mimics this with an effective
     cooling schedule instead of physical qubits.
  4. The process is repeated for many independent reads/samples; the
     lowest-energy sample found across all reads is reported as the answer.
"""

# ---------------------------------------------------------------------------
# Bootstrap: install requirements.txt if anything it lists isn't importable
# yet. This makes the file self-contained on a fresh qBraid environment -
# no separate "pip install -r requirements.txt" cell needed first. Skips
# straight through (no-op) once everything is already installed.
# ---------------------------------------------------------------------------
import importlib
import os
import subprocess
import sys

_REQUIREMENTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")

# requirements.txt package name -> the module name actually used to import it
_IMPORT_NAME_OVERRIDES = {
    "qiskit-finance": "qiskit_finance",
    "qiskit-algorithms": "qiskit_algorithms",
    "qiskit-aer": "qiskit_aer",
    "qiskit-optimization": "qiskit_optimization",
    "dwave-ocean-sdk": "dwave.cloud",
    "dwave-system": "dwave.system",
}


def _ensure_requirements_installed(requirements_path=_REQUIREMENTS_PATH):
    if not os.path.exists(requirements_path):
        return  # nothing to bootstrap from (e.g. file moved) - imports below will just fail loudly

    with open(requirements_path) as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    missing = []
    for pkg in packages:
        module_name = _IMPORT_NAME_OVERRIDES.get(pkg, pkg.replace("-", "_"))
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[setup] Installing missing packages from requirements.txt: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", requirements_path]
        )


_ensure_requirements_installed()


import numpy as np
import dimod
import neal
from dwave.system import DWaveSampler, EmbeddingComposite

from qiskit_optimization.converters import QuadraticProgramToQubo

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
# 0. Annealing configuration
# ---------------------------------------------------------------------------
NUM_READS = 1000     # independent annealing runs; best-of-all is reported
SEED = 1234          # only used by the simulated fallback


# ---------------------------------------------------------------------------
# 1. Convert the constrained QUBO (qp) into an unconstrained BQM
# ---------------------------------------------------------------------------
def build_bqm(quadratic_program):
    """Fold the budget constraint into a penalty and return a dimod BQM."""
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(quadratic_program)

    linear = qubo.objective.linear.to_dict()
    quadratic = qubo.objective.quadratic.to_dict()
    offset = qubo.objective.constant

    Q = {}
    for i, c in linear.items():
        Q[(i, i)] = Q.get((i, i), 0.0) + c
    for (i, j), c in quadratic.items():
        Q[(i, j)] = Q.get((i, j), 0.0) + c

    return dimod.BQM.from_qubo(Q, offset=offset), qubo


# ---------------------------------------------------------------------------
# 2. Get a sampler: real D-Wave QPU on qBraid if reachable, else simulate
# ---------------------------------------------------------------------------
def get_sampler():
    """Return (sampler, is_real_hardware, extra_sample_kwargs, backend_name)."""
    try:
        # This is the call that requires either a configured Leap API token
        # or qBraid's Quantum Jobs proxy to succeed. It fails fast (raises)
        # if neither is available, which is exactly what we want here.
        base_sampler = DWaveSampler()
        sampler = EmbeddingComposite(base_sampler)
        backend_name = base_sampler.properties.get("chip_id", "D-Wave QPU")
        # annealing_time (microseconds) is a real-hardware-only knob
        return sampler, True, {"annealing_time": 20}, backend_name

    except Exception as exc:  # noqa: BLE001 - deliberately broad: any
        # missing token or unreachable network should fall back to the
        # local simulator rather than crash the run.
        print(f"[info] Real D-Wave hardware unavailable ({exc.__class__.__name__}: {exc}).")
        print("[info] Falling back to neal.SimulatedAnnealingSampler (local, classical).")
        return neal.SimulatedAnnealingSampler(), False, {"seed": SEED}, "neal (simulated)"


# ---------------------------------------------------------------------------
# 3. Solve with (real or simulated) annealing
# ---------------------------------------------------------------------------
def solve_annealing(quadratic_program, num_reads=NUM_READS):
    """Run the D-Wave sampler (real if available, else simulated) on the
    BQM built from `qp`.

    Returns (selection, value, sampleset, is_real_hardware, backend_name).
    `selection` is a 0/1 numpy array in the same order as `TICKERS`.
    `value` is the ORIGINAL (unpenalized) portfolio objective evaluated on
    that selection - not the penalized QUBO energy.
    """
    bqm, qubo = build_bqm(quadratic_program)
    sampler, is_real_hardware, sample_kwargs, backend_name = get_sampler()

    sampleset = sampler.sample(bqm, num_reads=num_reads, **sample_kwargs)

    best = sampleset.first
    n = len(TICKERS)
    selection = np.array([best.sample[i] for i in range(n)])
    value = quadratic_program.objective.evaluate(selection)
    return selection, value, sampleset, is_real_hardware, backend_name


# ---------------------------------------------------------------------------
# 4. Run + compare against the exact classical reference
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"NSE tickers (imported from classical module): {TICKERS}")
    print(f"Budget = {budget} of {len(TICKERS)}, risk factor q = {q}")
    print(f"Annealing config: num_reads={NUM_READS}")

    print("\n" + "=" * 60)
    print("ANNEALING RESULT")
    print("=" * 60)
    (anneal_selection, anneal_value, sampleset,
     is_real_hardware, backend_name) = solve_annealing(qp)

    print(f"\nBackend used: {backend_name} "
          f"({'REAL D-Wave QPU' if is_real_hardware else 'local simulation'})")

    # Show the lowest-energy few samples found, deduplicated by bitstring
    print("\n------------- Lowest-energy samples found -------------")
    print("selection\t\t\tenergy\t\tnum_occurrences")
    print("---------------------------------------------------------")
    seen = set()
    shown = 0
    for datum in sampleset.data(sorted_by="energy"):
        bits = tuple(int(datum.sample[i]) for i in range(len(TICKERS)))
        if bits in seen:
            continue
        seen.add(bits)
        print(f"{list(bits)}\t{datum.energy:.6f}\t{datum.num_occurrences}")
        shown += 1
        if shown >= 10:
            break

    anneal_chosen = summarize(
        anneal_selection, anneal_value,
        label=f"Annealing ({backend_name})",
    )

    # -----------------------------------------------------------------
    # 5. Compare against the exact classical reference
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLASSICAL REFERENCE (for comparison)")
    print("=" * 60)
    classical_result = solve_classical()
    classical_selection = classical_result.x
    classical_value = classical_result.fval
    classical_chosen = summarize(
        classical_selection, classical_value, label="Classical (NumPyMinimumEigensolver)"
    )

    print("\n" + "=" * 60)
    print("ANNEALING vs CLASSICAL")
    print("=" * 60)
    print(f"Classical optimal portfolio : {classical_chosen}  (value {classical_value:.6f})")
    print(f"Annealing portfolio          : {anneal_chosen}  (value {anneal_value:.6f})")
    match = set(anneal_chosen) == set(classical_chosen)
    gap = anneal_value - classical_value
    print(f"Matches classical optimum?   : {match}")
    print(f"Objective gap (anneal - exact): {gap:.6f}")
