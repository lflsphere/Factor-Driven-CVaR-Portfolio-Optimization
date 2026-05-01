import numpy as np
import pandas as pd
from solve_mean_cvar_lp import solve_mean_cvar_lp

# Load the scenario returns matrix from CSV
csv_path = '/Users/ekeyigoz/Dropbox/STAT/finance/final_project/scenario_returns_matrix.csv'

print(f"Loading returns data from {csv_path}...")
# Read CSV with header row, skip the Date column (first column)
df = pd.read_csv(csv_path, index_col=0)
R = df.values
asset_names = df.columns.tolist()

print(f"Loaded {R.shape[0]} scenarios with {R.shape[1]} assets")
print(f"Return matrix shape: {R.shape}")
print(f"Mean returns: {R.mean(axis=0)}")
print(f"Std returns: {R.std(axis=0)}")

# Solve the CVaR optimization problem
print("\nSolving CVaR optimization...")
result = solve_mean_cvar_lp(
    R,
    beta=0.95,
    mean_weight=0.0,
    lower_bound=0.0,
    upper_bound=1.0,
)

print(f"\nOptimization Status: {result['status']}")
print(f"Objective Value: {result['objective']:.6f}")
print(f"CVaR (95%): {result['cvar']:.6f}")
print(f"VaR (95%): {result['var']:.6f}")
print(f"Expected Return: {result['expected_return']:.6f}")
print(f"\nOptimal Weights:")
for i, w in enumerate(result['weights']):
    if w > 0.001:  # Only show non-zero weights
        print(f"  {asset_names[i]}: {w:.4f} ({w*100:.2f}%)")

# Made with Bob
