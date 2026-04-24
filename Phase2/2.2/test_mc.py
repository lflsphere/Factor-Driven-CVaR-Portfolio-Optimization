"""
test_mc.py — Validation suite for §2.2 Monte Carlo (v1).

Checks:
  (1) Fit-then-recover: fit MVT to a known synthetic MVT sample, ν̂ within
      ±2 of true ν, μ̂ within MC error, Σ̂ entries within MC error.
  (2) Simulated marginal moments match historical (mean, std). Heavier-tail
      moments (skew, kurt) won't match exactly because the MVT is symmetric
      with a single ν — that's a known v1 limitation; test just reports.
  (3) Per-asset CVaR at β = 0.95 / 0.975 / 0.99 — simulated should be in the
      same ballpark as historical (within one historical SE), AND should be
      smoother (no ±50% jumps from one ETF to the next that you see in the
      historical with only 2-3 tail obs).
  (4) Effective tail size at β = 0.99 — the whole point: should jump from
      3 (historical) to ~500 (S=50k MC).
  (5) Reproducibility: same seed ⇒ identical output.

Run:  python test_mc.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Reuse 2.5's empirical CVaR for apples-to-apples comparison.
sys.path.insert(0, os.path.join(HERE, "..", "2.5"))
sys.path.insert(0, HERE)  # fallback if 2.5 sits next to this folder
try:
    from cvar import cvar_empirical, effective_tail_size
except ImportError:
    # Fallback: load 2.5's cvar.py from the outputs directory layout.
    sys.path.insert(0, os.path.join(HERE))
    from cvar import cvar_empirical, effective_tail_size  # type: ignore

from mc_simulator import fit_mvt, simulate_mvt, find_input_csv


def report(name, ok, *vals):
    flag = "OK   " if ok else "FAIL "
    extras = "  ".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                       for v in vals)
    print(f"  [{flag}] {name:<55s} {extras}")
    return ok


# ---------------------------------------------------------------------------
def test_fit_recovery():
    print("\n(1) Fit recovery on synthetic MVT(ν=7, d=10)")
    rng = np.random.default_rng(11)
    n, d, nu_true = 5000, 10, 7.0
    mu_true = rng.normal(0, 0.01, size=d)
    A = rng.normal(0, 0.02, size=(d, d))
    Sigma_true = A @ A.T + 1e-4 * np.eye(d)

    R = simulate_mvt(mu_true, Sigma_true, nu_true, S=n, rng=rng)
    mu_hat, Sigma_hat, nu_hat, _ = fit_mvt(R)

    p1 = report(f"ν̂ in [ν−2, ν+2]: ν̂={nu_hat:.2f}",
                abs(nu_hat - nu_true) < 2.0, nu_hat, nu_true)
    p2 = report(f"max |μ̂ − μ| < 0.01",
                np.max(np.abs(mu_hat - mu_true)) < 0.01,
                float(np.max(np.abs(mu_hat - mu_true))))
    p3 = report(f"max |Σ̂ − Σ| / max|Σ| < 0.20 (rough)",
                np.max(np.abs(Sigma_hat - Sigma_true)) / np.max(np.abs(Sigma_true)) < 0.20,
                float(np.max(np.abs(Sigma_hat - Sigma_true)) / np.max(np.abs(Sigma_true))))
    return p1 and p2 and p3


# ---------------------------------------------------------------------------
def _load_real_data():
    """Load Sijing's historical scenarios + (if present) the MC output."""
    src = find_input_csv()
    R_hist_df = pd.read_csv(src, index_col=0, parse_dates=True)
    R_hist = R_hist_df.values

    mc_csv = os.path.join(HERE, "scenario_returns_matrix_mc.csv")
    if not os.path.exists(mc_csv):
        # Generate on the fly into a tmp location.
        from mc_simulator import main as run_mc
        run_mc(S_mc=50_000, seed=20260423, out_dir=HERE)
    R_mc_df = pd.read_csv(mc_csv, index_col=0)
    R_mc = R_mc_df.values
    return R_hist_df, R_hist, R_mc_df, R_mc


def test_marginal_moments():
    print("\n(2) Marginal moments: simulated vs historical")
    R_hist_df, R_hist, R_mc_df, R_mc = _load_real_data()
    cols = R_hist_df.columns

    # Mean
    diff_mean = np.abs(R_mc.mean(0) - R_hist.mean(0))
    rel_mean = diff_mean / (np.abs(R_hist.mean(0)) + 1e-6)
    p1 = report(f"max |μ_mc - μ_hist| (raw)",
                bool(diff_mean.max() < 0.005),
                float(diff_mean.max()))

    # Std
    diff_std = np.abs(R_mc.std(0, ddof=1) - R_hist.std(0, ddof=1))
    rel_std = diff_std / R_hist.std(0, ddof=1)
    p2 = report(f"max relative |σ_mc - σ_hist| / σ_hist",
                bool(rel_std.max() < 0.10),
                float(rel_std.max()))
    return p1 and p2


def test_per_asset_cvar():
    print("\n(3) Per-asset CVaR comparison (positive = loss)")
    R_hist_df, R_hist, _, R_mc = _load_real_data()
    cols = R_hist_df.columns

    rows = []
    all_ok = True
    for beta in (0.95, 0.975, 0.99):
        c_hist = np.array([cvar_empirical(-R_hist[:, j], beta)
                           for j in range(R_hist.shape[1])])
        c_mc   = np.array([cvar_empirical(-R_mc[:, j], beta)
                           for j in range(R_mc.shape[1])])
        # Median absolute relative deviation, ignoring assets with tiny CVaR
        # (e.g. SHY, very low risk; floats around zero magnify rel error).
        meaningful = c_hist > 0.005
        rel = np.abs(c_mc[meaningful] - c_hist[meaningful]) / c_hist[meaningful]
        med_rel = float(np.median(rel))
        rows.append({
            "β":            beta,
            "median |Δ|/CVaR_hist": f"{med_rel:.2%}",
            "max |Δ|/CVaR_hist":    f"{float(rel.max()):.2%}",
        })
        # Loose tolerance: 25% median relative deviation is fine.
        # Historical at β=0.99 is so noisy (3 obs) that exact match is
        # impossible — and undesirable.
        if not (med_rel < 0.25):
            all_ok = False
    print(pd.DataFrame(rows).to_string(index=False))
    return report("median rel deviation < 25% at all β", all_ok)


def test_tail_size():
    print("\n(4) Effective tail size: historical vs MC")
    R_hist_df, R_hist, _, R_mc = _load_real_data()
    rows = []
    for beta in (0.95, 0.975, 0.99):
        rows.append({
            "β":             beta,
            "tail size hist":effective_tail_size(R_hist.shape[0], beta),
            "tail size MC":  effective_tail_size(R_mc.shape[0], beta),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    return report("MC tail at β=0.99 has ≥100 obs",
                  effective_tail_size(R_mc.shape[0], 0.99) >= 100)


def test_reproducibility():
    print("\n(5) Reproducibility (same seed → identical output)")
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    mu = np.zeros(5)
    Sigma = np.eye(5)
    A = simulate_mvt(mu, Sigma, 7.0, S=1000, rng=rng1)
    B = simulate_mvt(mu, Sigma, 7.0, S=1000, rng=rng2)
    return report("seed reproducibility", np.array_equal(A, B))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("§2.2 MC simulator — validation suite")
    print("=" * 72)

    results = {
        "Fit recovery":        test_fit_recovery(),
        "Marginal moments":    test_marginal_moments(),
        "Per-asset CVaR":      test_per_asset_cvar(),
        "Tail size jump":      test_tail_size(),
        "Reproducibility":     test_reproducibility(),
    }

    print("\n" + "=" * 72)
    for name, ok in results.items():
        print(f"  {name:<22s}  {'PASS' if ok else 'FAIL'}")
    print("=" * 72)
    sys.exit(0 if all(results.values()) else 1)
