"""
=============================================================================
Phase 4 -- CVaR / VaR Backtest Coverage Tests + Bootstrap Stability
=============================================================================

Two statistical-inference layers on top of the backtest:

(A) VaR / CVaR coverage tests
----------------------------------------------------------------------------
Test whether the realized exception rate matches the nominal level
(1 - beta) and whether exceptions are independent over time.

  Kupiec (1995) Proportion-of-Failures (POF):
      LR_POF = -2 ln[ (1-p)^(T-x) p^x / (1-pi)^(T-x) pi^x ]
      with p = 1 - beta, x = number of exceptions, pi = x/T
      ~ chi^2(1) under H0: pi = p

  Christoffersen (1998) Independence test:
      LR_ind = -2 ln[ L(pi)/L(pi_01, pi_11) ] ~ chi^2(1)

  Combined conditional coverage:
      LR_cc = LR_POF + LR_ind ~ chi^2(2)

  CVaR magnitude test (Acerbi & Szekely 2014, Test 2):
      Z_2 = (1/T) sum_t [ X_t * I(X_t < -VaR_t) / (T (1-beta) CVaR_t) ] + 1
      Approx zero under correct CVaR.

(B) Block-bootstrap stability (Chambers/Lohre/Rother 2018, Figure 4)
----------------------------------------------------------------------------
Generate B = 500 block-bootstrap simulations of asset returns (block length
~ 12 months to preserve serial dependence). For each, re-run the optimizer
and compute summary statistics. Report the resulting distribution of
volatility, max drawdown, Sharpe, Calmar, CVaR for each strategy.

Inputs
------
  Phase4/data/backtest_results.csv     (realized returns + ex-ante VaR/CVaR)

Outputs
-------
  Phase4/data/coverage_test_results.csv
  Phase4/data/bootstrap_stability.csv  (per-strategy summary distributions)

Owner: Rene
=============================================================================
"""

# TODO: implement
#   1. Kupiec POF test  -- exception count vs nominal
#   2. Christoffersen independence + conditional coverage
#   3. Acerbi-Szekely CVaR magnitude test (Test 2)
#   4. Stationary / circular block bootstrap (block length ~ 12)
#   5. Re-run each benchmark on B bootstrap paths, collect summary stats
#   6. Output boxplot-ready dataframe of per-strategy distributions
