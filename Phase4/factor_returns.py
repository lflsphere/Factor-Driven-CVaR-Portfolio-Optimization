"""
=============================================================================
Phase 4 -- Factor Return Series Construction
Carry | Value | Momentum | Quality
=============================================================================
Builds tradeable long-short factor portfolio return series from the
cross-sectional z-scores produced in Phase 1.

For each month t and each factor f:
  1. Rank ETFs by z-score z_{i,f,t}
  2. Long the top quintile (equal-weighted)
  3. Short the bottom quintile (equal-weighted)
  4. Hold one month, realising return at t+1

  f_{f, t+1} = (1/n_top) sum_{i in top} r_{i, t+1}
             - (1/n_bot) sum_{i in bot} r_{i, t+1}

Look-ahead protection: signal at t is paired with return at t+1 via shift(-1).

These series are the RHS regressors for the Phase 4 factor model:
  r_{i,t} = alpha_i + beta_C f_{C,t} + beta_V f_{V,t}
                   + beta_M f_{M,t} + beta_Q f_{Q,t} + eps_{i,t}

Output -> Phase4/data/Factor_Returns.csv

Inputs (produced by Phase 1):
  Phase1/data/Log_Returns.csv
  Phase1/data/factors/{Carry,Value,MOM,Quality}_ZScore.csv

References
----------
  Fama & French (1993, 2015)
  Asness, Moskowitz & Pedersen (JF 2013)
  Jegadeesh & Titman (JF 1993)
  Novy-Marx (JFE 2013)
  Koijen, Moskowitz, Pedersen & Vrugt (JFE 2018)
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
FACTOR_DIR = ROOT / "Phase1" / "data" / "factors"
RETURNS    = ROOT / "Phase1" / "data" / "Log_Returns.csv"
OUT_DIR    = ROOT / "Phase4" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FACTOR_FILES = {
    "Carry":    "Carry_ZScore.csv",
    "Value":    "Value_ZScore.csv",
    "Momentum": "MOM_ZScore.csv",
    "Quality":  "Quality_ZScore.csv",
}

QUINTILE = 0.20   # top / bottom 20% per factor


def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. This script depends on Phase 1 outputs; "
            f"run on the merged main branch or after merging phase1 into phase4."
        )
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df.sort_index()


def factor_return_series(z: pd.DataFrame,
                         ret: pd.DataFrame,
                         q: float = QUINTILE) -> pd.Series:
    """
    Long top quintile, short bottom quintile, equal-weighted within each leg.
    Signal at t is paired with return at t+1 (shifted) to avoid look-ahead.
    """
    common = z.columns.intersection(ret.columns)
    z   = z[common]
    ret = ret[common].shift(-1)

    out = pd.Series(index=z.index, dtype=float)
    for t, row in z.iterrows():
        valid = row.dropna()
        if len(valid) < 5 or t not in ret.index:
            continue
        n_leg  = max(int(round(q * len(valid))), 1)
        ranked = valid.sort_values()
        bot    = ranked.iloc[:n_leg].index
        top    = ranked.iloc[-n_leg:].index
        r_next = ret.loc[t]
        long_leg  = r_next[top].mean()
        short_leg = r_next[bot].mean()
        if pd.notna(long_leg) and pd.notna(short_leg):
            out.loc[t] = long_leg - short_leg
    return out.dropna()


def main():
    print("[Phase 4] Building factor return series")
    print(f"  factor dir : {FACTOR_DIR}")
    print(f"  returns    : {RETURNS}")
    print(f"  output dir : {OUT_DIR}\n")

    ret = load_panel(RETURNS)
    print(f"  log returns : {ret.shape[0]} months x {ret.shape[1]} ETFs\n")

    factor_returns = {}
    for name, fname in FACTOR_FILES.items():
        z = load_panel(FACTOR_DIR / fname)
        f = factor_return_series(z, ret)
        factor_returns[name] = f
        ann_mean   = f.mean() * 12
        ann_vol    = f.std()  * np.sqrt(12)
        ann_sharpe = ann_mean / ann_vol if ann_vol > 0 else np.nan
        print(f"  {name:9s} : {len(f):>4d} obs   "
              f"ann mean = {ann_mean:+.2%}   "
              f"ann vol = {ann_vol:.2%}   "
              f"sharpe = {ann_sharpe:+.2f}")

    out = pd.concat(factor_returns, axis=1)
    out.index.name = "Date"
    out_path = OUT_DIR / "Factor_Returns.csv"
    out.to_csv(out_path)
    print(f"\n  Saved {out.shape[0]} months x {out.shape[1]} factors -> {out_path}")

    print("\n  Pairwise correlations:")
    print(out.corr().round(3))


if __name__ == "__main__":
    main()
