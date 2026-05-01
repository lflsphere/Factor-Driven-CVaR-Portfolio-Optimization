import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# 1. Configuration & Paths (Robust Version)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "Phase1", "data", "Adj_Close.csv")

# Constants for Windows (as per task requirements)
LONG_WINDOW = 36   
SHORT_WINDOW = 12  
BETA_LEVELS = [0.95, 0.975, 0.99] # Multi-level CVaR

# ---------------------------------------------------------
# 2. Core Quant Functions
# ---------------------------------------------------------
def get_cvar(series, beta=0.95):
    alpha = 1 - beta
    var_threshold = series.quantile(alpha)
    return series[series <= var_threshold].mean()

def calculate_rolling_cvar(returns_series, window, beta=0.95):
    return returns_series.rolling(window=window).apply(lambda x: get_cvar(x, beta))

# ---------------------------------------------------------
# 3. Main Execution Workflow
# ---------------------------------------------------------
def execute_phase_2_3():
    print(f"Loading data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found. Check if the path exists.")

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    raw_returns = np.log(df / df.shift(1)).dropna()

    # Step A: Volatility Normalization
    long_vol = raw_returns.rolling(window=LONG_WINDOW).std()
    short_vol = raw_returns.rolling(window=SHORT_WINDOW).std()
    valid_idx = long_vol.dropna().index.intersection(short_vol.dropna().index)
    
    z_scores = raw_returns.loc[valid_idx] / long_vol.loc[valid_idx]

    # Step B: Re-scaling to Current Regime
    current_vol_regime = short_vol.iloc[-1]
    rescaled_matrix = z_scores * current_vol_regime

    # Step C: Generate CVaR Forecast Summary (Deliverable 2.5)
    print("Generating CVaR Forecast Summary...")
    summary_stats = pd.DataFrame(index=rescaled_matrix.columns)
    for b in BETA_LEVELS:
        summary_stats[f'CVaR_{int(b*100)}%'] = rescaled_matrix.apply(get_cvar, beta=b)

    # ---------------------------------------------------------
    # 4. Export Deliverables
    # ---------------------------------------------------------
    rescaled_matrix.to_csv(os.path.join(BASE_DIR, "scenario_returns_matrix.csv"))
    summary_stats.to_csv(os.path.join(BASE_DIR, "cvar_forecast_summary.csv"))
    print("Success: CSV deliverables saved.")

    # ---------------------------------------------------------
    # 5. Visualization & Validation
    # ---------------------------------------------------------
    target_asset = raw_returns.columns[0]
    
    # Plot 1: Returns Normalization Impact
    plt.figure(figsize=(10, 5))
    plt.plot(raw_returns[target_asset].tail(60), label='Raw Returns', color='gray', alpha=0.4)
    plt.plot(rescaled_matrix[target_asset].tail(60), label='Scaled Returns', color='blue', linewidth=1.5)
    plt.title(f"Normalization Impact: {target_asset}")
    plt.legend()
    plt.savefig(os.path.join(BASE_DIR, "volatility_correction_plot.png"), dpi=300)

    # Plot 2: CVaR Time-Series Comparison
    pro_cvar = calculate_rolling_cvar(raw_returns[target_asset], window=LONG_WINDOW, beta=0.95)
    norm_cvar = (z_scores[target_asset].rolling(window=LONG_WINDOW).apply(lambda x: get_cvar(x, 0.95)) 
                 * short_vol[target_asset].loc[valid_idx])

    plt.figure(figsize=(12, 6))
    plt.plot(pro_cvar.tail(500), label='Procyclical (Raw)', color='red', linestyle='--')
    plt.plot(norm_cvar.tail(500), label='Normalized (Phase 2.3)', color='blue')
    plt.title("CVaR Validation: Procyclical vs Normalized")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig(os.path.join(BASE_DIR, "cvar_validation_comparison.png"), dpi=300)
    
    print("Success: All plots generated.")

if __name__ == "__main__":
    execute_phase_2_3()