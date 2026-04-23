# Phase 4 -- Risk Attribution, Backtest, Statistical Inference

## Files

| Script | Owner (proposed) | Status |
| --- | --- | --- |
| `factor_returns.py` | Rene | Done -- builds long-short Carry / Value / Momentum / Quality factor return series from Phase 1 z-scores |
| `euler_decomposition.py` | Gia | Stub -- factor-level CVaR contributions via Euler theorem and tail averaging |
| `backtest.py` | Hongjie | Stub -- rolling-window walk-forward backtest vs 1/N, MV, DRP, Min CVaR, MCC |
| `coverage_tests.py` | Rene | Stub -- Kupiec / Christoffersen / Acerbi-Szekely + block bootstrap stability |

## Outputs

`Phase4/data/Factor_Returns.csv` -- monthly factor returns (Carry, Value, Momentum, Quality) constructed as quintile long-short portfolios from Phase 1 z-scores. 239 months, signal at t paired with return at t+1 (no look-ahead).

## Run

```bash
# from repo root, with Phase 1 outputs available (i.e. on merged main)
python Phase4/factor_returns.py
```

The script depends on `Phase1/data/Log_Returns.csv` and `Phase1/data/factors/{Carry,Value,MOM,Quality}_ZScore.csv`. On a standalone `phase4` branch these will not be present; run after merging or fetch the files from main.
