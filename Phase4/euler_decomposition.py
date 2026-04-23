"""
=============================================================================
Phase 4.1 -- Euler Decomposition of Portfolio CVaR into Factor Contributions
=============================================================================
Given a fitted factor model

  r_i = alpha_i + B_i' f + eps_i              (n_assets x n_factors loadings B)

and portfolio weights w, the portfolio return decomposes as

  r_p = w' r = (B'w)' f + w' eps
             = z' f      + w' eps              with z = B'w

By Euler's theorem on positively homogeneous risk measures, CVaR satisfies

  CVaR(r_p) = sum_k z_k * dCVaR / dz_k
            + sum_i w_i * dCVaR / dw_{i, idio}

The first sum gives the **factor contribution** to portfolio CVaR; the
second the residual idiosyncratic contribution. Empirically each marginal
contribution is

  dCVaR / dz_k = -E[ f_k | r_p <= -VaR(r_p) ]      (Scaillet 2002)

which is estimated from the simulated scenario set as the average factor
return across the (1 - beta) tail scenarios.

Inputs
------
  Phase4/data/Factor_Returns.csv             (factor returns f_t)
  Phase1/data/Log_Returns.csv                (asset returns r_{i,t})
  Phase2/data/scenario_returns_matrix.csv    (S x n simulated asset returns)
  Phase3 weights (optimized portfolio w)

Outputs
-------
  Phase4/data/factor_cvar_contributions.csv   (factor-level CVaR contributions)
  Phase4/data/asset_cvar_contributions.csv    (asset-level CVaR contributions)

Owner: Hongjie Zhang (proposed)
=============================================================================
"""

# TODO: implement
#   1. Fit per-asset OLS regressions r_i ~ f to get B (n_assets x 4) and residuals
#   2. Load scenario_returns_matrix and portfolio weights w
#   3. Identify tail scenarios where w' r_k <= -VaR
#   4. Compute factor marginal CVaR contributions via tail averaging
#   5. Aggregate to z_k * MC_k for total factor contribution
#   6. Output stacked-bar style attribution (Carry/Value/Mom/Quality + idio)
