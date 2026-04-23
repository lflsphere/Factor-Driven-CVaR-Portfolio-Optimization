"""
demo_cvar.py — End-to-end demo of §2.5 on the real scenario matrix.

Loads scenario_returns_matrix.csv from §2.3, computes equal-weight portfolio
CVaR at 95 / 97.5 / 99 % via all three methods (a, b, c), and reports the
effective tail-sample size at each β so the fragility is visible.

Equal-weight is just a placeholder — Phase 3 will solve for optimal w.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cvar import (
    portfolio_losses,
    cvar_empirical,
    cvar_gaussian,
    cvar_student_t,
    cvar_rockafellar_uryasev,
    effective_tail_size,
)
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def find_scenarios():
    for d in ["/sessions/keen-dazzling-goodall/mnt/uploads", HERE]:
        p = os.path.join(d, "scenario_returns_matrix.csv")
        if os.path.exists(p):
            return p
    raise FileNotFoundError("scenario_returns_matrix.csv not found")


def main():
    scen_path = find_scenarios()
    R_df = pd.read_csv(scen_path, index_col=0, parse_dates=True)
    R = R_df.values
    S, N = R.shape
    print(f"Loaded {scen_path}")
    print(f"Scenario matrix: S={S} scenarios × N={N} assets")
    print(f"Assets: {list(R_df.columns)}\n")

    # Equal-weight portfolio.
    w = np.full(N, 1.0 / N)
    L = portfolio_losses(R, w)

    print(f"Equal-weight portfolio losses: mean={L.mean():+.4%}  "
          f"std={L.std(ddof=1):.4%}")
    if HAS_SCIPY:
        nu_hat, mu_hat, sigma_hat = stats.t.fit(L)
        print(f"Student-t MLE on portfolio losses: ν={nu_hat:.2f}  "
              f"μ={mu_hat:+.4%}  σ={sigma_hat:.4%}\n")
    else:
        print("(scipy not installed → parametric Gaussian/Student-t columns "
              "will be marked n/a in the table)\n")

    rows = []
    for beta in (0.95, 0.975, 0.99):
        n_tail = effective_tail_size(S, beta)
        emp = cvar_empirical(L, beta)
        ru, alpha_star = cvar_rockafellar_uryasev(L, beta)
        gauss = (f"{cvar_gaussian(L.mean(), L.std(ddof=1), beta):.4%}"
                 if HAS_SCIPY else "n/a")
        tdist = (f"{cvar_student_t(mu_hat, sigma_hat, nu_hat, beta):.4%}"
                 if HAS_SCIPY else "n/a")
        rows.append({
            "β":            beta,
            "tail size":    n_tail,
            "VaR (α*)":     f"{alpha_star:.4%}",
            "(a) empirical":f"{emp:.4%}",
            "(c) RU":       f"{ru:.4%}",
            "(b) Gaussian": gauss,
            "(b) Student-t":tdist,
        })

    print("Equal-weight portfolio CVaR (positive = loss):")
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    # Fragility flag.
    print("Note: with S=204 historical scenarios,")
    for beta in (0.95, 0.975, 0.99):
        n = effective_tail_size(S, beta)
        print(f"  β={beta:.3f} → only {n} obs in the empirical tail "
              f"({'OK' if n >= 10 else 'FRAGILE — consider EVT (§2.4) or more scenarios via §2.2'})")

    print("""
Why (a) and (c) differ on this data:
  When (1−β)·S is not an integer (e.g. 0.05·204 = 10.2), the textbook
  CVaR definition (= what RU returns) gives a fractional weight to the
  VaR-boundary observation. The simple "mean of top ⌈(1−β)·S⌉" formula
  in (a) over-includes that boundary point with full weight, biasing the
  estimate slightly upward. (c) is the canonical estimator; (a) and (c)
  agree exactly when (1−β)·S is integer (verified on synthetic data with
  S=200,000 in test_cvar.py).
  → Treat the (c) RU column as the primary CVaR. (a) is a quick check.""")


if __name__ == "__main__":
    main()
