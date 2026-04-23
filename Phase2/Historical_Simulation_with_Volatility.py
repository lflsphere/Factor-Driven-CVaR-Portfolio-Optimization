import pandas as pd
import numpy as np
import os

# 1. Setup Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "Phase1", "data", "Adj_Close.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Data not found: {file_path}")

# 2. Data Processing
df = pd.read_csv(file_path, index_col=0, parse_dates=True)
returns = np.log(df / df.shift(1)).dropna() # Absolute Log Returns

# 3. Dual-Window Volatility (Implementation Plan 2.3)
LONG_WINDOW = 36   # For de-volatilization (3-year)
SHORT_WINDOW = 12  # For forward-looking rescaling

long_vol = returns.rolling(window=LONG_WINDOW).std().dropna()
short_vol = returns.rolling(window=SHORT_WINDOW).std().dropna()

# Align indices
common_idx = long_vol.index.intersection(short_vol.index)
z_scores = returns.loc[common_idx] / long_vol.loc[common_idx]

# 4. Generate S x n Scenario Matrix
current_vol = short_vol.iloc[-1] 
rescaled_scenarios = z_scores * current_vol

# 5. Multi-level CVaR Module (Implementation Plan 2.5)
def get_cvar(series, beta):
    alpha = 1 - beta
    return series[series <= series.quantile(alpha)].mean()

stats = pd.DataFrame(index=rescaled_scenarios.columns)
for b in [0.95, 0.975, 0.99]:
    stats[f'CVaR_{b*100}%'] = rescaled_scenarios.apply(get_cvar, beta=b)

# 6. Export Deliverables
rescaled_scenarios.to_csv(os.path.join(current_dir, "scenario_returns_matrix.csv"))
stats.to_csv(os.path.join(current_dir, "cvar_forecast_summary.csv"))

print(f"Matrix Shape (S x n): {rescaled_scenarios.shape}")
print("\nCVaR Estimates:\n", stats.head())