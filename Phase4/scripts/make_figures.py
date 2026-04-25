"""
=============================================================================
Phase 4 -- Figure generation for slide deck + memo paper appendix
=============================================================================
Reads the CSVs already written by factor_returns.py and backtest.py, and
writes three PNGs to Phase4/figures/:

  1. cumulative_factor_returns.png
       Cumulative log returns of the four long-short style factor portfolios
       (Carry, Value, Momentum, Quality), 239 months.
  2. strategy_equity_curves.png
       Walk-forward growth of $1 for the five backtested strategies
       (EqualWeight, MinVar, DRP, MinCVaR, MinCVaR_Cap), 205 OOS months.
  3. strategy_summary_metrics.png
       Per-strategy bars for Annualised Sharpe, Max Drawdown,
       and CVaR breach rate (vs the 5% nominal beta=0.95 line).

Run from anywhere:
    python Phase4/scripts/make_figures.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR  = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_ORDER = ["EqualWeight", "MinVar", "DRP", "MinCVaR", "MinCVaR_Cap"]
STRATEGY_COLORS = {
    "EqualWeight":  "#7f7f7f",
    "MinVar":       "#1f77b4",
    "DRP":          "#2ca02c",
    "MinCVaR":      "#d62728",
    "MinCVaR_Cap":  "#9467bd",
}


def plot_factor_returns():
    path = DATA_DIR / "Factor_Returns.csv"
    df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
    cum = df.fillna(0.0).cumsum()

    fig, ax = plt.subplots(figsize=(9, 5))
    for col in cum.columns:
        ax.plot(cum.index, cum[col], label=col, linewidth=1.6)
    ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_title("Cumulative Long-Short Factor Returns (Quintile Portfolios)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative log return")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "cumulative_factor_returns.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_equity_curves():
    path = DATA_DIR / "backtest_results.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values(["Strategy", "Date"])

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in STRATEGY_ORDER:
        g = df[df["Strategy"] == name]
        if g.empty:
            continue
        equity = np.exp(g["Realized"].cumsum().values)
        ax.plot(g["Date"].values, equity,
                label=name, linewidth=1.6,
                color=STRATEGY_COLORS.get(name))
    ax.set_yscale("log")
    ax.set_title("Walk-Forward Strategy Equity Curves (growth of $1, log scale)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative growth")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = FIG_DIR / "strategy_equity_curves.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_summary_metrics():
    path = DATA_DIR / "backtest_summary.csv"
    s = pd.read_csv(path).set_index("Strategy")
    s = s.reindex(STRATEGY_ORDER)
    nominal = float(s["Nominal_Breach_Rate"].iloc[0])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    colors = [STRATEGY_COLORS[n] for n in s.index]

    axes[0].bar(s.index, s["Sharpe"], color=colors)
    axes[0].set_title("Annualised Sharpe")
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(s.index, s["Max_Drawdown"], color=colors)
    axes[1].set_title("Max Drawdown")
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(s.index, s["CVaR_Breach_Rate"], color=colors)
    axes[2].axhline(nominal, color="black", linestyle="--", linewidth=1.0,
                    label=f"Nominal {nominal:.0%}")
    axes[2].set_title("CVaR Breach Rate (β=0.95)")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].legend(frameon=False, loc="best")

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Phase 4 Backtest Summary (205 OOS months)", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "strategy_summary_metrics.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print("[Phase 4] Generating figures")
    print(f"  data dir : {DATA_DIR}")
    print(f"  fig dir  : {FIG_DIR}\n")
    plot_factor_returns()
    plot_equity_curves()
    plot_summary_metrics()
    print("\nDone.")


if __name__ == "__main__":
    main()
