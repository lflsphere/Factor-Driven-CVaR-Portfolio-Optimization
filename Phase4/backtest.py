"""
=============================================================================
Phase 4.4 -- Rolling-Window Backtest
=============================================================================
Walk-forward backtest of the CVaR optimizer against standard risk-based
benchmarks. At each rebalance date t:

  1. Re-estimate factor model and tail distribution on data up to t
  2. Generate scenario returns (Phase 2 pipeline)
  3. Solve the Phase 3 optimization for new weights w_t
  4. Hold w_t until t+1, record realized portfolio return

Benchmarks
----------
  - 1/N         equal-weight
  - MV          minimum variance
  - DRP         diversified risk parity (4 style + market factors)
  - Min CVaR    Phase 3 R-U LP (no position cap)
  - Min CVaR + 40% position cap (Boudt 4.3.2)
  - MCC         minimum CVaR concentration (optional, differential evolution)

Reported metrics
----------------
  Annualised return / vol / Sharpe
  Maximum drawdown, Calmar ratio
  CVaR coverage ratio (fraction of months realized loss > predicted CVaR)
  Portfolio turnover
  Skewness, excess kurtosis (out-of-sample return distribution)

Inputs
------
  Phase1/data/Log_Returns.csv
  Phase4/data/Factor_Returns.csv
  Phase3 optimizer (callable)

Outputs
-------
  Phase4/data/backtest_results.csv      (per-date weights and realized returns)
  Phase4/data/backtest_summary.csv      (per-strategy summary statistics)

Owner: Hongjie (proposed)
=============================================================================
"""

# TODO: implement
#   1. Define rebalance schedule (monthly, 60-month rolling estimation window)
#   2. Wrap each benchmark + the Phase 3 optimizer behind a common interface
#   3. Loop over rebalance dates, store weights and realized OOS returns
#   4. Compute summary statistics matching Chambers/Lohre/Rother Table 1
