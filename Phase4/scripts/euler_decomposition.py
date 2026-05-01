"""
=============================================================================
Phase 4 -- Asset-level Euler / Scaillet CVaR decomposition
=============================================================================
For each backtested strategy, compute the asset-level CVaR contributions
(Scaillet 2002 tail averaging):

    CVaR_p   = E[ -r_p | r_p <= -VaR_p ]
    MCVaR_i  = E[ -r_i  | r_p <= -VaR_p ]      (marginal contribution)
    CCVaR_i  = w_i * MCVaR_i                   (component contribution)

where the conditioning event is the worst (1 - beta) tail of the realised
portfolio loss series. Outputs a per-asset contribution table and a top-10
contributor bar chart per strategy, plus a comparison line chart.

Inputs
------
    Phase1/data/Log_Returns.csv          -- Date x Asset return panel
    Phase4/data/backtest_weights.csv     -- stacked Date x Strategy x Asset

Outputs
-------
    Phase4/data/euler_asset_cvar_contributions_all_strategies.csv
    Phase4/figures/cvar_contribution_{strategy_lower}_top10.png
    Phase4/figures/cvar_contribution_summary.png

Author: Shirley (XueyiLiu11). Repo-relative paths so it runs from any clone.

Run from anywhere:
    python Phase4/scripts/euler_decomposition.py
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent.parent
SCENARIO   = ROOT / "Phase1" / "data" / "Log_Returns.csv"
WEIGHTS    = ROOT / "Phase4" / "data" / "backtest_weights.csv"
DATA_DIR   = ROOT / "Phase4" / "data"
FIG_DIR    = ROOT / "Phase4" / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "euler_asset_cvar_contributions_all_strategies.csv"

BETA = 0.95


def fig_slug(strategy: str) -> str:
    return strategy.lower()


def main():
    print("[Phase 4] Euler decomposition")
    print(f"  scenarios : {SCENARIO}")
    print(f"  weights   : {WEIGHTS}")
    print(f"  data dir  : {DATA_DIR}")
    print(f"  fig dir   : {FIG_DIR}\n")

    scenario_df = pd.read_csv(SCENARIO)
    weights_df  = pd.read_csv(WEIGHTS)

    scenario_df["Date"] = pd.to_datetime(scenario_df["Date"])
    weights_df["Date"]  = pd.to_datetime(weights_df["Date"])

    scenario_returns = scenario_df.set_index("Date")
    strategies = weights_df["Strategy"].unique()

    print("Strategies found:", strategies)
    all_results = []

    for strategy in strategies:
        print(f"\nProcessing strategy: {strategy}")
        strategy_weights = weights_df[weights_df["Strategy"] == strategy].copy()

        weights_wide = strategy_weights.pivot_table(
            index="Date", columns="Asset", values="Weight", aggfunc="sum",
        )

        common_assets = scenario_returns.columns.intersection(weights_wide.columns)
        scenario_sub = scenario_returns[common_assets]
        weights_sub  = weights_wide[common_assets]

        # Weights at t are applied to returns at t+1
        weights_aligned = weights_sub.shift(1)

        common_dates = scenario_sub.index.intersection(weights_aligned.index)
        scenario_aligned = scenario_sub.loc[common_dates]
        weights_aligned  = weights_aligned.loc[common_dates].fillna(0)

        portfolio_returns = (scenario_aligned * weights_aligned).sum(axis=1)
        portfolio_losses  = -portfolio_returns

        var_threshold = portfolio_losses.quantile(BETA)
        tail_mask     = portfolio_losses >= var_threshold

        tail_returns = scenario_aligned.loc[tail_mask]
        tail_weights = weights_aligned.loc[tail_mask]
        tail_losses  = portfolio_losses.loc[tail_mask]
        cvar_estimate = tail_losses.mean()

        asset_contributions = -(tail_weights * tail_returns).mean(axis=0)
        total_contribution  = asset_contributions.sum()
        if total_contribution != 0:
            contribution_share = asset_contributions / total_contribution
        else:
            contribution_share = asset_contributions * 0

        result = pd.DataFrame({
            "Strategy":            strategy,
            "Asset":               asset_contributions.index,
            "CVaR_Contribution":   asset_contributions.values,
            "Contribution_Share":  contribution_share.values,
            "VaR_Threshold":       var_threshold,
            "CVaR_Estimate":       cvar_estimate,
            "N_Tail_Scenarios":    tail_mask.sum(),
            "N_Common_Assets":     len(common_assets),
        }).sort_values("CVaR_Contribution", ascending=False)
        all_results.append(result)

        print("Tail scenarios:", tail_mask.sum())
        print("CVaR estimate:", cvar_estimate)
        print("Top 5 contributors:")
        print(result[["Asset", "CVaR_Contribution", "Contribution_Share"]].head())

        # Top-10 bar chart
        top10 = result.head(10).copy()
        plt.figure(figsize=(10, 6))
        plt.bar(top10["Asset"], top10["Contribution_Share"])
        plt.title(f"Top 10 Asset CVaR Contributions - {strategy}")
        plt.xlabel("Asset")
        plt.ylabel("Contribution Share")
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig_path = FIG_DIR / f"cvar_contribution_{fig_slug(strategy)}_top10.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"  wrote {fig_path}")

    # Combined results
    final_result = pd.concat(all_results, ignore_index=True)
    final_result.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved all strategy Euler decomposition results to: {OUTPUT_FILE}")

    # Summary chart
    top_assets_by_strategy = final_result.groupby("Strategy").head(5)
    plt.figure(figsize=(12, 7))
    for strategy in strategies:
        temp = top_assets_by_strategy[top_assets_by_strategy["Strategy"] == strategy]
        plt.plot(temp["Asset"], temp["Contribution_Share"],
                 marker="o", label=strategy)
    plt.title("Top Asset CVaR Contribution Shares by Strategy")
    plt.xlabel("Asset")
    plt.ylabel("Contribution Share")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    summary_fig_path = FIG_DIR / "cvar_contribution_summary.png"
    plt.savefig(summary_fig_path, dpi=300)
    plt.close()
    print(f"Saved summary figure to: {summary_fig_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
