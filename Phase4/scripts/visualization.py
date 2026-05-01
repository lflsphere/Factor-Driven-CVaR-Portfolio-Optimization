"""
=============================================================================
Phase 4 -- Visualization (cumulative returns, drawdowns, summary table)
=============================================================================
Reads the per-date backtest realised returns and the per-strategy summary
written by `backtest.py`, and produces:

    figures/cumulative_returns.png    -- (1+r) compounded growth per strategy
    figures/drawdown_comparison.png   -- drawdown lines per strategy
    data/summary_table.csv            -- {Ann_Return, Ann_Vol, Sharpe,
                                          Max_Drawdown} extract for the deck

Author: Shirley (XueyiLiu11). Repo-relative paths so it runs from any clone.

Run from anywhere:
    python Phase4/scripts/visualization.py
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR  = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("[Phase 4] Visualization")
    print(f"  data dir : {DATA_DIR}")
    print(f"  fig dir  : {FIG_DIR}\n")

    # Per-date realised returns -> wide format
    df = pd.read_csv(DATA_DIR / "backtest_results.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df_pivot = df.pivot(index="Date", columns="Strategy", values="Realized")

    # Cumulative returns
    cum = (1 + df_pivot).cumprod()
    plt.figure(figsize=(10, 6))
    for col in cum.columns:
        plt.plot(cum.index, cum[col], label=col)
    plt.legend()
    plt.title("Cumulative Returns Comparison")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.tight_layout()
    out_cum = FIG_DIR / "cumulative_returns.png"
    plt.savefig(out_cum, dpi=300)
    plt.close()
    print(f"  wrote {out_cum}")

    # Drawdowns
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    plt.figure(figsize=(10, 6))
    for col in drawdown.columns:
        plt.plot(drawdown.index, drawdown[col], label=col)
    plt.legend()
    plt.title("Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    out_dd = FIG_DIR / "drawdown_comparison.png"
    plt.savefig(out_dd, dpi=300)
    plt.close()
    print(f"  wrote {out_dd}")

    # Summary subset table
    summary_df = pd.read_csv(DATA_DIR / "backtest_summary.csv")
    summary_selected = summary_df[
        ["Strategy", "Ann_Return", "Ann_Vol", "Sharpe", "Max_Drawdown"]
    ].copy().round(4)
    out_sum = DATA_DIR / "summary_table.csv"
    summary_selected.to_csv(out_sum, index=False)
    print(f"  wrote {out_sum}")
    print("\nDone.")


if __name__ == "__main__":
    main()
