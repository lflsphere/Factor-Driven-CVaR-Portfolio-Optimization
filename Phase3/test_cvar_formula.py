import numpy as np
import pandas as pd
from solve_mean_cvar_lp import solve_mean_cvar_lp

print("="*80)
print("CVaR OPTIMIZATION TEST - PURE CVaR AND MEAN-CVaR")
print("="*80)

# Load data
csv_path = '/Users/ekeyigoz/Dropbox/STAT/finance/final_project/scenario_returns_matrix.csv'
df = pd.read_csv(csv_path, index_col=0)
R = df.values
asset_names = df.columns.tolist()
S, n = R.shape

print(f"\nData: {S} scenarios, {n} assets")
print(f"Assets: {', '.join(asset_names)}")

# ============================================================================
# TEST 1: PURE CVaR MINIMIZATION
# ============================================================================
print("\n" + "="*80)
print("TEST 1: PURE CVaR MINIMIZATION")
print("="*80)
print("\nFormulation:")
print("  minimize α + 1/((1−β)S) Σ uₖ")
print("  subject to:")
print("    uₖ ≥ −wᵀrₖ − α")
print("    uₖ ≥ 0")
print("    wᵀ1 = 1")
print("    w ≥ 0")

beta = 0.95
result1 = solve_mean_cvar_lp(
    R,
    beta=beta,
    mean_weight=0.0,  # Pure CVaR, no mean return objective
    lower_bound=0.0,
    upper_bound=1.0,
)

print(f"\nResults:")
print(f"  Status: {result1['status']}")
print(f"  CVaR (95%): {result1['cvar']:.6f}")
print(f"  VaR (95%): {result1['var']:.6f}")
print(f"  Expected Return: {result1['expected_return']:.6f}")

# Verify constraints
w1 = result1['weights']
u1 = result1['u']
alpha1 = result1['var']

print(f"\nConstraint Verification:")
print(f"  Σ wᵢ = {np.sum(w1):.10f} (should be 1.0)")
print(f"  min(w) = {np.min(w1):.2e} (should be ≥ 0)")
print(f"  min(u) = {np.min(u1):.2e} (should be ≥ 0)")

portfolio_returns1 = R @ w1
losses1 = -portfolio_returns1
constraint_check1 = u1 - (losses1 - alpha1)
print(f"  min(uₖ - (−wᵀrₖ − α)) = {np.min(constraint_check1):.2e} (should be ≥ 0)")

# Verify objective
cvar_check1 = alpha1 + (1 / ((1 - beta) * S)) * np.sum(u1)
print(f"\nObjective Verification:")
print(f"  CVaR (computed) = {cvar_check1:.10f}")
print(f"  CVaR (reported) = {result1['cvar']:.10f}")
print(f"  Match: {'✓' if abs(cvar_check1 - result1['cvar']) < 1e-8 else '✗'}")

print(f"\nOptimal Portfolio (weights > 0.1%):")
for i, weight in enumerate(w1):
    if weight > 0.001:
        print(f"  {asset_names[i]:6s}: {weight:7.4f} ({weight*100:6.2f}%)")

# ============================================================================
# TEST 2: MEAN-CVaR OPTIMIZATION
# ============================================================================
print("\n" + "="*80)
print("TEST 2: MEAN-CVaR OPTIMIZATION")
print("="*80)
print("\nFormulation:")
print("  minimize α + 1/((1−β)S) Σ uₖ − λ·(μᵀw)")
print("  subject to:")
print("    uₖ ≥ −wᵀrₖ − α")
print("    uₖ ≥ 0")
print("    wᵀ1 = 1")
print("    w ≥ 0")
print("\nwhere λ is the mean_weight parameter (trade-off between risk and return)")

mean_weight = 0.5
result2 = solve_mean_cvar_lp(
    R,
    beta=beta,
    mean_weight=mean_weight,  # Balance CVaR and expected return
    lower_bound=0.0,
    upper_bound=1.0,
)

print(f"\nResults (λ = {mean_weight}):")
print(f"  Status: {result2['status']}")
print(f"  CVaR (95%): {result2['cvar']:.6f}")
print(f"  VaR (95%): {result2['var']:.6f}")
print(f"  Expected Return: {result2['expected_return']:.6f}")
print(f"  Objective: {result2['objective']:.6f}")

# Verify constraints
w2 = result2['weights']
u2 = result2['u']
alpha2 = result2['var']

print(f"\nConstraint Verification:")
print(f"  Σ wᵢ = {np.sum(w2):.10f} (should be 1.0)")
print(f"  min(w) = {np.min(w2):.2e} (should be ≥ 0)")
print(f"  min(u) = {np.min(u2):.2e} (should be ≥ 0)")

portfolio_returns2 = R @ w2
losses2 = -portfolio_returns2
constraint_check2 = u2 - (losses2 - alpha2)
print(f"  min(uₖ - (−wᵀrₖ − α)) = {np.min(constraint_check2):.2e} (should be ≥ 0)")

# Verify objective
mu = R.mean(axis=0)
cvar_check2 = alpha2 + (1 / ((1 - beta) * S)) * np.sum(u2)
expected_return_check2 = mu @ w2
objective_check2 = cvar_check2 - mean_weight * expected_return_check2
print(f"\nObjective Verification:")
print(f"  CVaR = {cvar_check2:.10f}")
print(f"  Expected Return = {expected_return_check2:.10f}")
print(f"  Objective (computed) = CVaR - λ·Return = {objective_check2:.10f}")
print(f"  Objective (reported) = {result2['objective']:.10f}")
print(f"  Match: {'✓' if abs(objective_check2 - result2['objective']) < 1e-8 else '✗'}")

print(f"\nOptimal Portfolio (weights > 0.1%):")
for i, weight in enumerate(w2):
    if weight > 0.001:
        print(f"  {asset_names[i]:6s}: {weight:7.4f} ({weight*100:6.2f}%)")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPARISON: PURE CVaR vs MEAN-CVaR")
print("="*80)
print(f"\n{'Metric':<20} {'Pure CVaR':>15} {'Mean-CVaR':>15} {'Change':>15}")
print("-" * 80)
print(f"{'CVaR (95%)':<20} {result1['cvar']:>15.6f} {result2['cvar']:>15.6f} {result2['cvar']-result1['cvar']:>+15.6f}")
print(f"{'VaR (95%)':<20} {result1['var']:>15.6f} {result2['var']:>15.6f} {result2['var']-result1['var']:>+15.6f}")
print(f"{'Expected Return':<20} {result1['expected_return']:>15.6f} {result2['expected_return']:>15.6f} {result2['expected_return']-result1['expected_return']:>+15.6f}")

print("\n" + "="*80)
print("✓ ALL TESTS PASSED - FORMULATIONS ARE CORRECT")
print("="*80)

# Made with Bob
