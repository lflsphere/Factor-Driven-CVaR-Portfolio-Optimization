import os
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend to ensure figure is saved

import pandas as pd
import matplotlib.pyplot as plt

# Print current working directory
print("Current working directory:", os.getcwd())

# Load backtest results
df = pd.read_csv("/Users/lxy/Desktop/backtest_results.csv")

# Preview raw data
print(df.head())
print(df.columns)

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Pivot from long format to wide format
df_pivot = df.pivot(index="Date", columns="Strategy", values="Realized")

# Preview pivoted data
print(df_pivot.head())

# Compute cumulative returns
cum = (1 + df_pivot).cumprod()

# Plot cumulative returns
plt.figure(figsize=(10, 6))
for col in cum.columns:
    plt.plot(cum.index, cum[col], label=col)

plt.legend()
plt.title("Cumulative Returns Comparison")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.tight_layout()

# Save figure
output1 = "/Users/lxy/Desktop/cumulative_returns.png"
output2 = "cumulative_returns.png"

plt.savefig(output1, dpi=300)
plt.savefig(output2, dpi=300)

# Confirm saved paths
print("Saved to desktop:", os.path.exists(output1), output1)
print("Saved to current folder:", os.path.exists(output2), os.path.abspath(output2))
# Compute drawdown
running_max = cum.cummax()
drawdown = cum / running_max - 1

# Plot drawdown
plt.figure(figsize=(10, 6))
for col in drawdown.columns:
    plt.plot(drawdown.index, drawdown[col], label=col)

plt.legend()
plt.title("Drawdown Comparison")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.tight_layout()

# Save drawdown figure
output3 = "/Users/lxy/Desktop/drawdown_comparison.png"
output4 = "drawdown_comparison.png"

plt.savefig(output3, dpi=300)
plt.savefig(output4, dpi=300)

print("Saved drawdown to desktop:", os.path.exists(output3), output3)
print("Saved drawdown to current folder:", os.path.exists(output4), os.path.abspath(output4))

# Load summary table
summary_df = pd.read_csv("/Users/lxy/Desktop/backtest_summary.csv")
print(summary_df)

print(summary_df.columns)

summary_selected = summary_df[
    ["Strategy", "Ann_Return", "Ann_Vol", "Sharpe", "Max_Drawdown"]
].copy()

summary_selected = summary_selected.round(4)

print(summary_selected)

summary_selected.to_csv("/Users/lxy/Desktop/summary_selected.csv", index=False)

summary_selected.to_csv("/Users/lxy/Desktop/summary_table.csv", index=False)