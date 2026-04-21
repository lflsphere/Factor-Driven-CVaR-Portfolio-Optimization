import pandas as pd
import numpy as np
import os

# 1. Setup Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "Phase1", "data", "Adj_Close.csv")
output_dir = current_dir

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Data missing at: {os.path.abspath(file_path)}")

# 2. Load Data (Monthly frequency)
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

# 3. Calculate Monthly Log Returns
monthly_returns = np.log(df / df.shift(1)).dropna()

# 4. Calculate Excess Returns over SPY
excess_returns = monthly_returns.sub(monthly_returns['SPY'], axis=0)

# 5. Compute Rolling Volatility (60-month window)
window_size = 60 

if len(excess_returns) < window_size:
    window_size = len(excess_returns) // 2
    print(f"Warning: Data too short, adjusting window to {window_size} months.")

rolling_vol = excess_returns.rolling(window=window_size).std().dropna()
excess_returns_aligned = excess_returns.loc[rolling_vol.index]

# 6. Volatility Normalization (De-volatilization)
normalized_returns = excess_returns_aligned / rolling_vol

# 7. Re-scale by Current Volatility
current_vol = rolling_vol.iloc[-1]
rescaled_scenarios = normalized_returns * current_vol

# 8. CVaR Calculation (95% Confidence Level)
def compute_cvar(series, alpha=0.05):
    var_threshold = series.quantile(alpha)
    return series[series <= var_threshold].mean()

cvar_results = rescaled_scenarios.apply(compute_cvar)

# 9. Output and Export
print(f"Calculated CVaR based on {window_size}-month rolling window:")
print(cvar_results)

output_path = os.path.join(output_dir, "normalized_cvar_results.csv")
cvar_results.to_csv(output_path)
print(f"Results saved to: {output_path}")