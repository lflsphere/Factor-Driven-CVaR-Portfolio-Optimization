import numpy as np
import pandas as pd
from solve_mean_cvar_lp import solve_mean_cvar_lp, gaussian_portfolio_cvar

# Load the scenario returns matrix from CSV
csv_path = '/Users/ekeyigoz/Dropbox/STAT/finance/final_project/scenario_returns_matrix.csv'

print("="*70)
print("CVaR OPTIMIZATION TEST SUITE")
print("="*70)

# Load data
print("\n1. Loading data...")
df = pd.read_csv(csv_path, index_col=0)
R = df.values
asset_names = df.columns.tolist()
print(f"   Loaded {R.shape[0]} scenarios with {R.shape[1]} assets")

# Test 1: Pure CVaR minimization (long-only)
print("\n2. Test 1: Pure CVaR Minimization (Long-Only)")
print("-" * 70)
result1 = solve_mean_cvar_lp(
    R,
    beta=0.95,
    mean_weight=0.0,
    lower_bound=0.0,
    upper_bound=1.0,
)
print(f"   Status: {result1['status']}")
print(f"   CVaR (95%): {result1['cvar']:.6f}")
print(f"   VaR (95%): {result1['var']:.6f}")
print(f"   Expected Return: {result1['expected_return']:.6f}")
print(f"   Top 3 holdings:")
weights1 = result1['weights']
top_indices = np.argsort(weights1)[-3:][::-1]
for idx in top_indices:
    if weights1[idx] > 0.001:
        print(f"      {asset_names[idx]}: {weights1[idx]:.4f} ({weights1[idx]*100:.2f}%)")

# Test 2: CVaR with mean return objective
print("\n3. Test 2: CVaR with Mean Return Weighting")
print("-" * 70)
result2 = solve_mean_cvar_lp(
    R,
    beta=0.95,
    mean_weight=0.5,  # Balance between CVaR and return
    lower_bound=0.0,
    upper_bound=1.0,
)
print(f"   Status: {result2['status']}")
print(f"   CVaR (95%): {result2['cvar']:.6f}")
print(f"   VaR (95%): {result2['var']:.6f}")
print(f"   Expected Return: {result2['expected_return']:.6f}")
print(f"   Top 5 holdings:")
weights2 = result2['weights']
top_indices = np.argsort(weights2)[-5:][::-1]
for idx in top_indices:
    if weights2[idx] > 0.001:
        print(f"      {asset_names[idx]}: {weights2[idx]:.4f} ({weights2[idx]*100:.2f}%)")

# Test 3: CVaR with minimum return constraint
print("\n4. Test 3: CVaR with Minimum Return Constraint")
print("-" * 70)
min_ret = 0.005  # Require at least 0.5% expected return
result3 = solve_mean_cvar_lp(
    R,
    beta=0.95,
    mean_weight=0.0,
    min_return=min_ret,
    lower_bound=0.0,
    upper_bound=1.0,
)
print(f"   Status: {result3['status']}")
print(f"   CVaR (95%): {result3['cvar']:.6f}")
print(f"   VaR (95%): {result3['var']:.6f}")
print(f"   Expected Return: {result3['expected_return']:.6f}")
print(f"   Minimum Required: {min_ret:.6f}")
print(f"   Top 5 holdings:")
weights3 = result3['weights']
top_indices = np.argsort(weights3)[-5:][::-1]
for idx in top_indices:
    if weights3[idx] > 0.001:
        print(f"      {asset_names[idx]}: {weights3[idx]:.4f} ({weights3[idx]*100:.2f}%)")

# Test 4: Different confidence levels
print("\n5. Test 4: Different Confidence Levels")
print("-" * 70)
for beta in [0.90, 0.95, 0.99]:
    result = solve_mean_cvar_lp(
        R,
        beta=beta,
        mean_weight=0.0,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    print(f"   Beta={beta:.2f}: CVaR={result['cvar']:.6f}, VaR={result['var']:.6f}, Return={result['expected_return']:.6f}")

# Test 5: Portfolio validation - verify CVaR calculation
print("\n6. Test 5: Portfolio Validation")
print("-" * 70)
w_test = result1['weights']
portfolio_returns = R @ w_test
losses = -portfolio_returns
sorted_losses = np.sort(losses)
var_index = int(np.ceil(0.95 * len(losses))) - 1
var_empirical = sorted_losses[var_index]
cvar_empirical = sorted_losses[var_index:].mean()

print(f"   Optimization CVaR: {result1['cvar']:.6f}")
print(f"   Empirical CVaR: {cvar_empirical:.6f}")
print(f"   Difference: {abs(result1['cvar'] - cvar_empirical):.6f}")
print(f"   Optimization VaR: {result1['var']:.6f}")
print(f"   Empirical VaR: {var_empirical:.6f}")
print(f"   Difference: {abs(result1['var'] - var_empirical):.6f}")

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print(f"All tests completed successfully!")
print(f"Data: {R.shape[0]} scenarios, {R.shape[1]} assets")
print(f"Solver used: SCS (open-source, no license required)")
print("="*70)

# Made with Bob
