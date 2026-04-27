import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set working directory to the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

def generate_deliverables():
    # Path configuration
    matrix_file = "scenario_returns_matrix.csv"
    raw_data_file = os.path.join("..", "Phase1", "data", "Adj_Close.csv")

    if not os.path.exists(matrix_file):
        print(f"Error: {matrix_file} not found in {BASE_DIR}")
        return

    # 1. Load data
    processed = pd.read_csv(matrix_file, index_col=0, parse_dates=True)
    
    # Try to locate raw data for comparison
    if os.path.exists(raw_data_file):
        raw_df = pd.read_csv(raw_data_file, index_col=0, parse_dates=True)
        raw_ret = np.log(raw_df / raw_df.shift(1)).dropna()
        
        # 2. Plotting (Using the first available asset)
        asset = processed.columns[0]
        plt.figure(figsize=(10, 5))
        
        # Plot last 60 months for clarity
        plt.plot(raw_ret[asset].tail(60), label='Raw Log Returns', color='gray', alpha=0.4)
        plt.plot(processed[asset].tail(60), label='Volatility-Adjusted (Phase 2.3)', color='blue', linewidth=1.5)
        
        plt.title(f"Risk Normalization Impact: {asset}")
        plt.xlabel("Date")
        plt.ylabel("Returns")
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        plot_name = "volatility_correction_plot.png"
        plt.savefig(plot_name, dpi=300)
        print(f"Success: Plot saved as {os.path.join(BASE_DIR, plot_name)}")
    else:
        print(f"Error: Raw data not found at {raw_data_file}")

if __name__ == "__main__":
    generate_deliverables()