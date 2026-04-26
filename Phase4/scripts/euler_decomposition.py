import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------
# File paths
# ----------------------------
BASE_DIR = "/Users/lxy/Desktop/5261 组/euler"

SCENARIO_FILE = os.path.join(BASE_DIR, "scenario_returns_matrix.csv")
WEIGHTS_FILE = os.path.join(BASE_DIR, "backtest_weights.csv")

OUTPUT_FILE = os.path.join(BASE_DIR, "asset_cvar_contributions_all_strategies.csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------
# Settings
# ----------------------------
BETA = 0.95

# ----------------------------
# Load data
# ----------------------------
scenario_df = pd.read_csv(SCENARIO_FILE)
weights_df = pd.read_csv(WEIGHTS_FILE)

scenario_df["Date"] = pd.to_datetime(scenario_df["Date"])
weights_df["Date"] = pd.to_datetime(weights_df["Date"])

scenario_returns = scenario_df.set_index("Date")

strategies = weights_df["Strategy"].unique()

all_results = []

print("Strategies found:", strategies)

# ----------------------------
# Euler decomposition function
# ----------------------------
for strategy in strategies:
    print(f"\nProcessing strategy: {strategy}")

    strategy_weights = weights_df[weights_df["Strategy"] == strategy].copy()

    weights_wide = strategy_weights.pivot_table(
        index="Date",
        columns="Asset",
        values="Weight",
        aggfunc="sum"
    )

    # Align assets between scenario returns and weights
    common_assets = scenario_returns.columns.intersection(weights_wide.columns)

    scenario_sub = scenario_returns[common_assets]
    weights_sub = weights_wide[common_assets]

    # Weights at time t are applied to returns at t+1
    weights_aligned = weights_sub.shift(1)

    # Align dates
    common_dates = scenario_sub.index.intersection(weights_aligned.index)

    scenario_aligned = scenario_sub.loc[common_dates]
    weights_aligned = weights_aligned.loc[common_dates].fillna(0)

    # Portfolio returns and losses
    portfolio_returns = (scenario_aligned * weights_aligned).sum(axis=1)
    portfolio_losses = -portfolio_returns

    # Tail scenarios
    var_threshold = portfolio_losses.quantile(BETA)
    tail_mask = portfolio_losses >= var_threshold

    tail_returns = scenario_aligned.loc[tail_mask]
    tail_weights = weights_aligned.loc[tail_mask]
    tail_losses = portfolio_losses.loc[tail_mask]

    cvar_estimate = tail_losses.mean()

    # Asset-level Euler contribution
    asset_contributions = -(tail_weights * tail_returns).mean(axis=0)

    total_contribution = asset_contributions.sum()

    if total_contribution != 0:
        contribution_share = asset_contributions / total_contribution
    else:
        contribution_share = asset_contributions * 0

    result = pd.DataFrame({
        "Strategy": strategy,
        "Asset": asset_contributions.index,
        "CVaR_Contribution": asset_contributions.values,
        "Contribution_Share": contribution_share.values,
        "VaR_Threshold": var_threshold,
        "CVaR_Estimate": cvar_estimate,
        "N_Tail_Scenarios": tail_mask.sum(),
        "N_Common_Assets": len(common_assets)
    })

    result = result.sort_values("CVaR_Contribution", ascending=False)
    all_results.append(result)

    print("Tail scenarios:", tail_mask.sum())
    print("CVaR estimate:", cvar_estimate)
    print("Top 5 contributors:")
    print(result[["Asset", "CVaR_Contribution", "Contribution_Share"]].head())

    # ----------------------------
    # Plot top 10 contributors
    # ----------------------------
    top10 = result.head(10).copy()

    plt.figure(figsize=(10, 6))
    plt.bar(top10["Asset"], top10["Contribution_Share"])
    plt.title(f"Top 10 Asset CVaR Contributions - {strategy}")
    plt.xlabel("Asset")
    plt.ylabel("Contribution Share")
    plt.xticks(rotation=45)
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, f"top10_cvar_contribution_{strategy}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

# ----------------------------
# Save combined results
# ----------------------------
final_result = pd.concat(all_results, ignore_index=True)
final_result.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved all strategy Euler decomposition results to: {OUTPUT_FILE}")
print(f"Saved figures to: {FIG_DIR}")

# ----------------------------
# Summary chart: top contributors by strategy
# ----------------------------
top_assets_by_strategy = final_result.groupby("Strategy").head(5)

plt.figure(figsize=(12, 7))

for strategy in strategies:
    temp = top_assets_by_strategy[top_assets_by_strategy["Strategy"] == strategy]
    plt.plot(
        temp["Asset"],
        temp["Contribution_Share"],
        marker="o",
        label=strategy
    )

plt.title("Top Asset CVaR Contribution Shares by Strategy")
plt.xlabel("Asset")
plt.ylabel("Contribution Share")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

summary_fig_path = os.path.join(FIG_DIR, "top_asset_cvar_contributions_by_strategy.png")
plt.savefig(summary_fig_path, dpi=300)
plt.close()

print(f"Saved summary figure to: {summary_fig_path}")
