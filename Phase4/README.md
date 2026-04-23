# Phase 4 -- Risk Attribution, Backtest, Statistical Inference

## Split

- `factor_returns.py` + `backtest.py` -- Rene
- `euler_decomposition.py` -- Hongjie
- `coverage_tests.py` -- Gia

## Outputs

`Phase4/data/Factor_Returns.csv` -- monthly factor returns (Carry, Value, Momentum, Quality) constructed as quintile long-short portfolios from Phase 1 z-scores. 239 months, signal at t paired with return at t+1 (no look-ahead).

## Run

```bash
# from repo root, with Phase 1 outputs available (i.e. on merged main)
python Phase4/factor_returns.py
```

The script depends on `Phase1/data/Log_Returns.csv` and `Phase1/data/factors/{Carry,Value,MOM,Quality}_ZScore.csv`. On a standalone `phase4` branch these will not be present; run after merging or fetch the files from main.

```bash
python Phase4/backtest.py
```

Rolling 60-month window, monthly rebalance. Benchmarks: 1/N, min-variance, equal-risk-contribution (DRP proxy), min CVaR LP, min CVaR with 40% position cap. Writes per-date realized returns + ex-ante CVaR to `Phase4/data/backtest_results.csv` and summary table to `Phase4/data/backtest_summary.csv`.
