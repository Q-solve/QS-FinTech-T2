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
CSV_PATH = "/mnt/user-data/uploads/NSE_data_all_stocks_2025__1_.csv"

# 8 liquid, well-known NSE blue chips spanning telecom, banking, and
# consumer goods. Swap these for any of the 67 ticker columns in the CSV.
TICKERS = ["SCOM", "EQTY", "KCB", "EABL", "BAT", "COOP", "BAMB", "JUB"]

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
    print("Warning - missing values found, forward-filling:")
    print(missing[missing > 0])
    prices = prices.ffill().bfill()

returns = prices.pct_change().dropna()

# mu: mean daily return per asset: sigma: covariance of daily returns
mu = returns.mean().values
sigma = returns.cov().values

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


# ---------------------------------------------------------------------------
# 2. Build the QUBO from real mu / sigma
# ---------------------------------------------------------------------------
portfolio = PortfolioOptimization(
    expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget
)
qp = portfolio.to_quadratic_program()

print("\n" + "=" * 60)
print("QUADRATIC PROGRAM (built from real NSE returns)")
print("=" * 60)
print(qp.prettyprint())


# ---------------------------------------------------------------------------
# 3. Utility: pretty-print the full result distribution
# ---------------------------------------------------------------------------
def print_result(result):
    selection = result.x
    value = result.fval
    print(f"\nOptimal: selection {selection}, value {value:.6f}\n")

    eigenstate = result.min_eigen_solver_result.eigenstate
    probabilities = (
        eigenstate.binary_probabilities()
        if isinstance(eigenstate, QuasiDistribution)
        else {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
    )

    print("----------------- Full result (top 10) --------------")
    print("selection\tvalue\t\tprobability")
    print("-------------------------------------------------------")
    probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    for k, v in probabilities[:10]:
        x = np.array([int(i) for i in list(reversed(k))])
        val = portfolio.to_quadratic_program().objective.evaluate(x)
        print("%s\t%.6f\t%.4f" % (x, val, v))

    return selection, value


# ---------------------------------------------------------------------------
# 4. Solve exactly with NumPyMinimumEigensolver (classical reference)
# ---------------------------------------------------------------------------
exact_mes = NumPyMinimumEigensolver()
exact_eigensolver = MinimumEigenOptimizer(exact_mes)

result = exact_eigensolver.solve(qp)

print("\n" + "=" * 60)
print("NUMPY CLASSICAL MINIMUM EIGENSOLVER RESULT")
print("=" * 60)
selection, value = print_result(result)

# ---------------------------------------------------------------------------
# 5. Human-readable summary
# ---------------------------------------------------------------------------
chosen = [TICKERS[i] for i, bit in enumerate(selection) if bit == 1]
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"NSE tickers considered : {TICKERS}")
print(f"Budget (k)              : {budget} of {len(TICKERS)}")
print(f"Risk factor (q)         : {q}")
print(f"Optimal portfolio       : {chosen}")
print(f"Objective value         : {value:.6f}")
