# Phase 4 — Risk Attribution, Backtest, and Statistical Inference

## Context

Phase 4 closes the loop on the project. Phase 1 built ETF style-factor
z-scores (Carry, Value, Momentum, Quality), Phase 2 implemented CVaR
estimators, and Phase 3 implemented the optimisers (Min-CVaR LP, MCC).
Phase 4 turns those building blocks into:

1. **Tradeable factor return series** — long-short quintile portfolios that
   serve as RHS regressors for risk decomposition.
2. **A walk-forward backtest** of the CVaR optimisers against standard
   risk-based benchmarks on a 27-ETF universe.
3. **Risk attribution and statistical inference** — Euler / Scaillet
   decomposition of portfolio CVaR onto contributing assets, plus
   coverage tests on the realised loss tail.

Outputs from this phase feed both the slide deck (5/2 presentation) and
the memo paper (due 5/13).

## Ownership

| Module                   | Owner   | Status         |
| ------------------------ | ------- | -------------- |
| `factor_returns.py`      | Rene    | Done           |
| `backtest.py`            | Rene    | Done           |
| `euler_decomposition.py` | Hongjie | Stub           |
| `coverage_tests.py`      | Gia     | Stub           |
| `scripts/make_figures.py`| Rene    | Done           |

## Algorithms

### 1. `factor_returns.py` — long-short quintile factor returns

- **Input**
  - `Phase1/data/Log_Returns.csv` — 27 ETFs × monthly log returns.
  - `Phase1/data/factors/{Carry,Value,MOM,Quality}_ZScore.csv` — cross-
    sectional z-scores per factor per month.
- **Processing**
  For each month `t` and each factor `f`:
  1. Rank the cross-section of ETFs by `z_{i,f,t}`.
  2. Long the top quintile (equal-weighted), short the bottom quintile
     (equal-weighted), held one month.
  3. Realise factor return at `t+1`:
     `f_{f, t+1} = mean(r_top, t+1) - mean(r_bot, t+1)`.
  4. Look-ahead protection: signals at `t` paired with returns at `t+1`
     via `shift(-1)`.
- **Output**
  `data/Factor_Returns.csv` — 239 monthly observations × 4 factors.
  Used as RHS regressors for the project's factor model:
  `r_{i,t} = α_i + β_C f_{C,t} + β_V f_{V,t} + β_M f_{M,t} + β_Q f_{Q,t} + ε_{i,t}`.

### 2. `backtest.py` — rolling-window walk-forward backtest

- **Input**
  - `Phase1/data/Log_Returns.csv` — asset return panel.
  - `Phase4/data/Factor_Returns.csv` — kept available for future
    factor-aware variants.
- **Processing**
  At each rebalance date `t`:
  1. Estimation window = trailing 60 months.
  2. Keep assets with ≥ 80% non-NaN coverage in the window.
  3. Solve each strategy on that sample.
  4. Hold weights `w_t` until `t+1`, record realised portfolio return.
  5. Store ex-ante in-sample CVaR for downstream coverage tests.

  Five strategies, all long-only, fully invested:
  - **EqualWeight** — 1/N across surviving assets.
  - **MinVar** — min-variance via SLSQP.
  - **DRP** — equal risk contribution, Spinu (2013) coordinate descent
    (proxy for Lohre diversified risk parity on assets).
  - **MinCVaR** — Rockafellar–Uryasev LP at β = 0.95.
  - **MinCVaR_Cap** — MinCVaR with a 40% per-name cap (Boudt 4.3.2).
- **Output**
  - `data/backtest_results.csv` — per-date, per-strategy realised return,
    ex-ante CVaR, and turnover.
  - `data/backtest_weights.csv` — stacked `Date × Strategy × Asset × Weight`.
  - `data/backtest_summary.csv` — annualised return / vol / Sharpe,
    max drawdown, Calmar, CVaR breach rate, turnover, skew, excess kurtosis.

### 3. `euler_decomposition.py` — CVaR risk attribution *(Hongjie, stub)*

- **Input** (planned) `backtest_weights.csv` + `Log_Returns.csv`.
- **Processing** (planned) Scaillet (2002) tail averaging:
  `MCVaR_i = E[r_i | r_p ≤ -VaR_p]`, then component CVaR
  `CCVaR_i = w_i · MCVaR_i`. Verifies Euler identity
  `Σ_i CCVaR_i = CVaR_p`.
- **Output** (planned) `data/cvar_attribution.csv`.

### 4. `coverage_tests.py` — statistical coverage of CVaR estimates *(Gia, stub)*

- **Input** (planned) `backtest_results.csv`.
- **Processing** (planned) Kupiec POF, Christoffersen independence,
  Acerbi–Szekely Z2; block bootstrap for finite-sample p-values.
- **Output** (planned) `data/coverage_tests.csv`.

### 5. `scripts/make_figures.py` — slide / appendix figures

- **Input** the three CSVs in `data/`.
- **Processing** matplotlib renders.
- **Output** three PNGs in `figures/` (see below).

## Figures (`figures/`)

| File                              | What it shows                                                                                              |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `cumulative_factor_returns.png`   | Cumulative log returns of the four long-short style factors (239 months).                                  |
| `strategy_equity_curves.png`      | Walk-forward growth of $1 per strategy across 205 OOS months, log scale.                                   |
| `strategy_summary_metrics.png`    | Per-strategy bars: annualised Sharpe, max drawdown, and CVaR breach rate vs the 5% nominal β = 0.95 line.  |

## Reproducing

```bash
# from repo root, with Phase 1 outputs available (i.e. on merged main)
python Phase4/factor_returns.py        # -> data/Factor_Returns.csv
python Phase4/backtest.py              # -> data/backtest_{results,weights,summary}.csv
python Phase4/scripts/make_figures.py  # -> figures/*.png
```

The `factor_returns.py` and `backtest.py` scripts depend on
`Phase1/data/Log_Returns.csv` and `Phase1/data/factors/*.csv`; on a
standalone `phase4` branch these are not present, so run after merging
or fetch the files from `main`.

## Configuration (set at top of `backtest.py`)

| Param          | Value | Source                                           |
| -------------- | ----- | ------------------------------------------------ |
| `WINDOW`       | 60    | rolling estimation window (months)               |
| `BETA`         | 0.95  | CVaR confidence level                            |
| `POS_CAP`      | 0.40  | Boudt 4.3.2 single-name cap for the capped variant |
| `MIN_COVERAGE` | 0.80  | minimum non-NaN fraction per asset per window    |

## Headline results (205 OOS months, β = 0.95)

| Strategy     | Ann Ret | Ann Vol | Sharpe | Max DD  | CVaR Breach |
| ------------ | ------: | ------: | -----: | ------: | ----------: |
| EqualWeight  |  10.98% |  12.06% |   0.91 | -19.62% |        2.93% |
| MinVar       |   1.75% |   1.39% |   1.26 |  -5.13% |        4.88% |
| DRP          |   4.27% |   4.34% |   0.98 | -10.87% |        4.88% |
| MinCVaR      |   1.83% |   1.57% |   1.16 |  -5.29% |        7.32% |
| MinCVaR_Cap  |   2.85% |   3.34% |   0.85 | -11.22% |        7.32% |

Numbers regenerate from `data/backtest_summary.csv` whenever the
backtest is re-run.
