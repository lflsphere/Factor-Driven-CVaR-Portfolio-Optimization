"""
test_cvar.py — Cross-validation harness for §2.5.

Implementation Plan §2.5 says:
    "Cross-validate all three methods to ensure consistency."

This script:

  (1) Synthetic Gaussian losses → empirical (a), parametric (b), and
      Rockafellar–Uryasev (c) must all agree on E[L | L≥VaR] within
      Monte Carlo error.
  (2) Synthetic Student-t losses → same agreement, with the parametric
      method using the closed-form ν > 1 formula.
  (3) Algebraic invariants:
        - CVaR(aL + b) = a·CVaR(L) + b for a > 0
        - Degenerate case: CVaR of a constant = the constant
  (4) Regression check: per-asset empirical CVaR computed from
      scenario_returns_matrix.csv reproduces the teammate's
      cvar_forecast_summary.csv (sign-flipped, since they were in
      return space and we work in loss space).

Run:  python test_cvar.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Make the local cvar.py importable regardless of cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cvar import (
    cvar_empirical,
    cvar_gaussian,
    cvar_student_t,
    cvar_rockafellar_uryasev,
    portfolio_losses,
    portfolio_cvar,
    per_asset_cvar_table,
    effective_tail_size,
)


def mc_se(sigma_tail: float, S: int, beta: float) -> float:
    """Rough one-sigma Monte Carlo error for empirical CVaR."""
    n_tail = max(int(np.ceil((1.0 - beta) * S)), 1)
    return sigma_tail / np.sqrt(n_tail)


def report(name, ok, *vals):
    flag = "OK   " if ok else "FAIL "
    extras = "  ".join(f"{v:.6f}" if isinstance(v, float) else str(v)
                       for v in vals)
    print(f"  [{flag}] {name:<55s} {extras}")
    return ok


# ---------------------------------------------------------------------------
# (1) Synthetic Gaussian
# ---------------------------------------------------------------------------
def test_gaussian():
    print("\n(1) Gaussian losses, mu=0, sigma=1, S=200_000")
    rng = np.random.default_rng(42)
    mu, sigma = 0.0, 1.0
    S = 200_000
    L = rng.normal(mu, sigma, size=S)

    passed = []
    for beta in (0.95, 0.975, 0.99):
        emp   = cvar_empirical(L, beta)
        ru, _ = cvar_rockafellar_uryasev(L, beta)
        passed.append(report(f"β={beta}: RU vs empirical (must match exactly)",
                             abs(ru - emp) < 1e-9, ru, emp))
        if HAS_SCIPY:
            analytic = cvar_gaussian(mu, sigma, beta)
            tail_sd  = sigma
            tol      = 4.0 * mc_se(tail_sd, S, beta)
            passed.append(report(
                f"β={beta}: empirical vs Gaussian closed-form (tol={tol:.4f})",
                abs(emp - analytic) < tol, emp, analytic))
        else:
            print(f"    [SKIP] β={beta}: parametric Gaussian needs scipy")
    return all(passed)


# ---------------------------------------------------------------------------
# (2) Synthetic Student-t
# ---------------------------------------------------------------------------
def test_student_t():
    print("\n(2) Student-t losses, ν=5, mu=0, sigma=1, S=200_000")
    rng = np.random.default_rng(7)
    nu, mu, sigma = 5.0, 0.0, 1.0
    S = 200_000
    L = mu + sigma * rng.standard_t(nu, size=S)

    passed = []
    for beta in (0.95, 0.975, 0.99):
        emp   = cvar_empirical(L, beta)
        ru, _ = cvar_rockafellar_uryasev(L, beta)
        passed.append(report(f"β={beta}: RU vs empirical (must match exactly)",
                             abs(ru - emp) < 1e-9, ru, emp))
        if HAS_SCIPY:
            analytic = cvar_student_t(mu, sigma, nu, beta)
            tail_sd  = sigma * np.sqrt(nu / (nu - 2))
            tol      = 5.0 * mc_se(tail_sd, S, beta)
            passed.append(report(
                f"β={beta}: empirical vs Student-t closed-form (tol={tol:.4f})",
                abs(emp - analytic) < tol, emp, analytic))
        else:
            print(f"    [SKIP] β={beta}: parametric Student-t needs scipy")
    return all(passed)


# ---------------------------------------------------------------------------
# (3) Algebraic invariants
# ---------------------------------------------------------------------------
def test_invariants():
    print("\n(3) Algebraic invariants")
    rng = np.random.default_rng(0)
    L = rng.normal(0, 1, size=10_000)
    a, b = 2.5, 0.3
    beta = 0.95

    base = cvar_empirical(L, beta)
    scaled = cvar_empirical(a * L + b, beta)
    p1 = report("CVaR(aL + b) == a·CVaR(L) + b",
                abs(scaled - (a * base + b)) < 1e-10,
                scaled, a * base + b)

    const = cvar_empirical(np.full(1000, 7.0), beta)
    p2 = report("CVaR(constant 7) == 7", abs(const - 7.0) < 1e-12, const)

    return p1 and p2


# ---------------------------------------------------------------------------
# (4) Regression check against teammate's 2.3 output
# ---------------------------------------------------------------------------
def test_regression_against_teammate():
    print("\n(4) Per-asset CVaR vs teammate's cvar_forecast_summary.csv")
    # Look for the upload paths first, then a local copy.
    candidate_dirs = [
        "/sessions/keen-dazzling-goodall/mnt/uploads",
        HERE,
    ]
    scen_path = stat_path = None
    for d in candidate_dirs:
        s = os.path.join(d, "scenario_returns_matrix.csv")
        c = os.path.join(d, "cvar_forecast_summary.csv")
        if os.path.exists(s) and os.path.exists(c):
            scen_path, stat_path = s, c
            break
    if scen_path is None:
        print("  [SKIP ] scenario_returns_matrix.csv not found; skipping regression test")
        return True

    R_df = pd.read_csv(scen_path, index_col=0, parse_dates=True)
    teammate = pd.read_csv(stat_path, index_col=0)
    R = R_df.values

  
    ours = per_asset_cvar_table(R, betas=(0.95, 0.975, 0.99))

    passed = True
    for b, col in [(0.95, "CVaR_95.0%"), (0.975, "CVaR_97.5%"), (0.99, "CVaR_99.0%")]:
        theirs_loss = -teammate[col].values  # flip sign to losses
        diff = np.max(np.abs(ours[b] - theirs_loss))
  
        rel = diff / np.maximum(np.abs(theirs_loss).max(), 1e-9)
        ok = rel < 0.10  # within 10% relative on the worst asset
        passed &= ok
        report(f"β={b}: max |our - teammate'| = {diff:.5f}  (rel {rel:.2%})", ok)
    return passed


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("§2.5 CVaR module — cross-validation suite")
    print("=" * 72)

    results = {
        "Gaussian":      test_gaussian(),
        "Student-t":     test_student_t(),
        "Invariants":    test_invariants(),
        "Regression":    test_regression_against_teammate(),
    }

    print("\n" + "=" * 72)
    for name, ok in results.items():
        print(f"  {name:<15s}  {'PASS' if ok else 'FAIL'}")
    print("=" * 72)

    sys.exit(0 if all(results.values()) else 1)
