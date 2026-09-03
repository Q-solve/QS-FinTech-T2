"""
Classical reference solution for mean-variance portfolio optimization,
using REAL Nairobi Securities Exchange (NSE) daily price data in place of
Qiskit's RandomDataProvider, solved exactly with NumPyMinimumEigensolver.

Problem:
    minimize   q * x^T Sigma x - mu^T x
    subject to sum(x) == budget,  x in {0,1}^n

Data source: NSE_data_all_stocks_2025__1_.csv
    - 208 trading days, Jan 2025 - Oct 2025
    - 67 listed companies (ticker columns) + 8 index columns (^N10I, ^NASI, etc.)

This module is written to be BOTH:
  (a) runnable directly (`python3 portfolio_nse_classical.py`) to solve
      exactly and print results, and
  (b) importable from another script (e.g. the QAOA version), which reuses
      TICKERS / q / budget / mu / sigma / portfolio / qp / print_result /
      solve_classical without re-triggering the print-heavy __main__ block.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit.result import QuasiDistribution
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization.algorithms import MinimumEigenOptimizer


# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
CSV_PATH = "data/NSE_combination_data_all_stock_2013_2025.csv"

# 8 liquid, well-known NSE blue chips spanning telecom, banking, and
# consumer goods. Swap these for any of the 67 ticker columns in the CSV.
TICKERS = ["SCOM", "EQTY", "KCB", "EABL", "BAT", "COOP", "BAMB", "JUB","ABSA","BKG","ABSA","AMAC","CABL","CIC","CGEN","EGAD"]

q = 0.5                      # risk aversion factor
budget = len(TICKERS) // 2   # number of assets to select (4 of 8 here)
penalty = len(TICKERS)       # penalty coefficient for the budget constraint


# ---------------------------------------------------------------------------
# 1. Load real NSE price data and compute daily returns
# ---------------------------------------------------------------------------
prices = pd.read_csv(CSV_PATH, parse_dates=["Date"]).set_index("Date")
prices = prices[TICKERS].sort_index()

missing = prices.isna().sum()
if missing.any():
    prices = prices.ffill().bfill()

returns = prices.pct_change().dropna()

# mu: mean daily return per asset; sigma: covariance of daily returns
mu = returns.mean().values
sigma = returns.cov().values


# ---------------------------------------------------------------------------
# 2. Build the QUBO from real mu / sigma (shared by classical + QAOA)
# ---------------------------------------------------------------------------
portfolio = PortfolioOptimization(
    expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget
)
qp = portfolio.to_quadratic_program()


# ---------------------------------------------------------------------------
# 3. Utility: pretty-print a MinimumEigenOptimizer result distribution
#    (works for both NumPyMinimumEigensolver and QAOA/SamplingVQE results)
# ---------------------------------------------------------------------------
def print_result(result, top_n=10):
    selection = result.x
    value = result.fval
    print(f"\nOptimal: selection {selection}, value {value:.6f}\n")

    eigenstate = result.min_eigen_solver_result.eigenstate
    if isinstance(eigenstate, QuasiDistribution):
        # NumPyMinimumEigensolver: QuasiDistribution over integer states
        probabilities = eigenstate.binary_probabilities()
    elif isinstance(eigenstate, dict):
        # QAOA / SamplingVQE (qiskit_algorithms >= 0.3): dict of
        # bitstring -> probability straight from sampling
        probabilities = eigenstate
    else:
        # Statevector-like object: amplitudes -> probabilities
        probabilities = {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}

    print(f"----------------- Full result (top {top_n}) --------------")
    print("selection\tvalue\t\tprobability")
    print("-------------------------------------------------------")
    probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    for k, v in probabilities[:top_n]:
        x = np.array([int(i) for i in list(reversed(k))])
        val = portfolio.to_quadratic_program().objective.evaluate(x)
        print("%s\t%.6f\t%.4f" % (x, val, v))

    return selection, value


def summarize(selection, value, label):
    chosen = [TICKERS[i] for i, bit in enumerate(selection) if bit == 1]
    print("\n" + "=" * 60)
    print(f"SUMMARY - {label}")
    print("=" * 60)
    print(f"NSE tickers considered : {TICKERS}")
    print(f"Budget (k)              : {budget} of {len(TICKERS)}")
    print(f"Risk factor (q)         : {q}")
    print(f"Optimal portfolio       : {chosen}")
    print(f"Objective value         : {value:.6f}")
    return chosen


# ---------------------------------------------------------------------------
# 4. Solve exactly with NumPyMinimumEigensolver (classical reference)
# ---------------------------------------------------------------------------
def solve_classical():
    """Solve `qp` exactly and return the MinimumEigenOptimizer result."""
    exact_mes = NumPyMinimumEigensolver()
    exact_eigensolver = MinimumEigenOptimizer(exact_mes)
    return exact_eigensolver.solve(qp)


# ---------------------------------------------------------------------------
# 5. Only runs when executed directly, not on import
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Loaded {len(prices)} trading days ({prices.index.min().date()} to "
          f"{prices.index.max().date()}) for {len(TICKERS)} NSE tickers: {TICKERS}")
    print("\nAnnualized mean return (%) per asset (252 trading days):")
    for t, m in zip(TICKERS, mu):
        print(f"  {t:6s}  {m * 252 * 100:6.2f}%")

    # Save covariance heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sigma, interpolation="nearest", cmap="viridis")
    ax.set_xticks(range(len(TICKERS)))
    ax.set_yticks(range(len(TICKERS)))
    ax.set_xticklabels(TICKERS, rotation=45, ha="right")
    ax.set_yticklabels(TICKERS)
    ax.set_title("NSE daily-return covariance matrix (sigma)")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("/home/claude/nse_sigma_heatmap.png", dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("QUADRATIC PROGRAM (built from real NSE returns)")
    print("=" * 60)
    print(qp.prettyprint())

    result = solve_classical()

    print("\n" + "=" * 60)
    print("NUMPY CLASSICAL MINIMUM EIGENSOLVER RESULT")
    print("=" * 60)
    selection, value = print_result(result)
    summarize(selection, value, label="Classical (NumPyMinimumEigensolver)")
