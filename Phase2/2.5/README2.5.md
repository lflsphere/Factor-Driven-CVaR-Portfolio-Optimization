# Phase 2.5 — CVaR Calculation Module

Implements the CVaR computation module described in §2.5 of the Implementation
Plan. Provides three methods for computing Conditional Value-at-Risk and a
cross-validation suite that confirms they agree.

## Files

- `cvar.py` — the module. Importable from Phase 3 as `from cvar import ...`.
- `test_cvar.py` — cross-validation suite (synthetic + regression).
- `demo_cvar.py` — end-to-end demo on the real §2.3 scenario matrix.
- `README.md` — this file.

## Methods implemented

| Method | Function | When to use |
|---|---|---|
| (a) Empirical / sample CVaR | `cvar_empirical(losses, beta)` | Fast, default for historical scenarios. |
| (b) Parametric closed-form | `cvar_gaussian(mu, sigma, beta)`, `cvar_student_t(mu, sigma, nu, beta)` | Sanity check, or when fitting a parametric marginal to factor returns. |
| (c) Rockafellar–Uryasev auxiliary | `cvar_rockafellar_uryasev(losses, beta)` | Canonical definition — matches the LP in §3.1 exactly. **Use this as the primary estimator.** |

Portfolio wrapper: `portfolio_cvar(R, w, beta, method=...)`.

## Convention

**Losses are positive.** Given a return matrix `R` (S × N) and weight vector
`w` (N,), portfolio losses are `L = -R @ w`. All functions return CVaR as a
positive number.

## How to run

```bash
# from Phase2/2.5/
pip install -r requirements.txt
python test_cvar.py   # runs the cross-validation suite
python demo_cvar.py   # loads ../2.3/scenario_returns_matrix.csv and reports equal-weight CVaR
```

Expected output of `test_cvar.py`: all four sections (Gaussian, Student-t,
Invariants, Regression) print PASS.

## Inputs

`demo_cvar.py` reads `../2.3/scenario_returns_matrix.csv` produced by Sijing's
§2.3. Shape is expected to be (S scenarios × N ETFs) of de-volatilized and
rescaled log returns.

## Quickstart API

```python
import numpy as np
import pandas as pd
from cvar import portfolio_cvar, cvar_rockafellar_uryasev, portfolio_losses

R = pd.read_csv("../2.3/scenario_returns_matrix.csv",
                index_col=0, parse_dates=True).values
w = np.full(R.shape[1], 1.0 / R.shape[1])      # equal weight

# One-shot:
cvar = portfolio_cvar(R, w, beta=0.95, method="ru")

# Or with VaR exposed:
L = portfolio_losses(R, w)
cvar, var = cvar_rockafellar_uryasev(L, beta=0.95)
```

## Cross-validation results

On S = 200,000 synthetic samples:

- Empirical (a) and RU (c) match to machine precision when `(1-β)·S` is an
  integer.
- Empirical (a) and the Gaussian / Student-t closed-form (b) match within
  Monte Carlo error (~4σ tolerance) in `test_cvar.py`.
- Per-asset CVaR on the real §2.3 matrix matches Sijing's
  `cvar_forecast_summary.csv` exactly (after sign-flipping to loss
  convention).

## Findings to flag for the team

1. **Tail fragility at high β.** With the current scenario matrix
   (S = 204 monthly scenarios), the empirical tail contains only
   ~11 observations at β = 0.95, 6 at β = 0.975, and **3 at β = 0.99**.
   The 99% number is essentially noise. Options: more scenarios via §2.2
   (optional Monte Carlo) or EVT tail modelling via §2.4 (optional GPD).

2. **(a) vs (c) on finite samples.** They disagree when `(1-β)·S` is
   non-integer (e.g. 0.05 · 204 = 10.2). This is expected — (c) is the
   canonical CVaR definition with a fractional weight on the boundary
   observation; (a) over-includes it with full weight and biases up
   slightly. **Report the (c) RU value as the headline number.**

3. **Data frequency.** §2.3's output is monthly (S = 204 ≈ 17 years
   end-of-month). The Implementation Plan's §2.3 describes "3-year long
   window / 60-day short window" which assumes daily data. Either the
   write-up needs to reflect the monthly cadence, or §2.3 should be re-run
   on daily log returns. Worth checking with Sijing.

## Handoff to §3.1 (Elif)

The LP in §3.1 is the Rockafellar–Uryasev formulation embedded in an
optimizer over `w`. For a fixed `w`, it should return exactly the same
number as `cvar_rockafellar_uryasev` here. Sanity-check value for
equal-weight, β = 0.95 on the current scenario matrix:

```
α* (VaR)          = 4.63%
CVaR (method c)   = 6.24%   ← 3.1's LP should match this at w = 1/N
CVaR (method a)   = 6.13%   ← for comparison (non-integer-tail bias)
```

Input contract Elif will consume:
- `R` : (S × N) ndarray of scenario returns — `scenario_returns_matrix.csv`
- `β` : scalar, default 0.95
- `w` : (N,) decision variable in the LP, with `w >= 0`, `sum(w) == 1`
  (long-only v1 per Louis's spec)

## Dependencies

- numpy, pandas — required
- scipy — required for methods (b); (a) and (c) run on numpy alone

## References

- Rockafellar & Uryasev (2000), "Optimization of Conditional Value-at-Risk", J. Risk.
- McNeil, Frey, Embrechts, *Quantitative Risk Management*, §2.2 (closed-form CVaR for location-scale families).
- Boudt et al. (2013) — the `PortfolioAnalytics` CVaR reference we're using for v1 scope.
