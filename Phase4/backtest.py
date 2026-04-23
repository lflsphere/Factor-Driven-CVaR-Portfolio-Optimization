"""
=============================================================================
Phase 4.4 -- Rolling-Window Backtest
=============================================================================
Walk-forward backtest of CVaR-based portfolios against standard risk-based
benchmarks. At each rebalance date t:

  1. Estimation window = trailing 60 months of asset returns
  2. Keep only assets with >= 80% non-NaN coverage in the window
  3. Solve each strategy's optimizer on that sample
  4. Hold weights w_t until t+1, record realized portfolio return
  5. Store ex-ante CVaR (in-sample) for later coverage tests

Strategies
----------
  1/N         equal-weighted across available assets
  MinVar      minimum variance, long-only, fully invested
  DRP         equal risk contribution (proxy for Lohre diversified risk parity)
  MinCVaR     Rockafellar-Uryasev LP, long-only, fully invested
  MinCVaR_Cap MinCVaR + 40% per-asset position cap (Boudt 4.3.2)

Reported metrics (per strategy)
-------------------------------
  Annualised return / vol / Sharpe
  Max drawdown, Calmar ratio
  CVaR coverage ratio (fraction of months with realized loss above ex-ante CVaR)
  Turnover, skewness, excess kurtosis

Inputs
------
  Phase1/data/Log_Returns.csv
  Phase4/data/Factor_Returns.csv     (kept for future factor-aware variants)

Outputs
-------
  Phase4/data/backtest_results.csv    (per-date, per-strategy realized + CVaR)
  Phase4/data/backtest_weights.csv    (stacked: Date, Strategy, Asset, Weight)
  Phase4/data/backtest_summary.csv    (per-strategy summary)
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog, minimize

ROOT        = Path(__file__).resolve().parent.parent
LOG_RETURNS = ROOT / "Phase1" / "data" / "Log_Returns.csv"
FACTOR_RETS = ROOT / "Phase4" / "data" / "Factor_Returns.csv"
OUT_DIR     = ROOT / "Phase4" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW        = 60        # rolling estimation window (months)
BETA          = 0.95      # CVaR confidence level
POS_CAP       = 0.40      # Boudt 4.3.2 single-name cap
MIN_COVERAGE  = 0.80      # asset must have >= 80% non-NaN in window
ANN_FACTOR    = 12


def load_returns(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run after merging phase1, or fetch the file from main."
        )
    df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
    return df


def cvar_empirical(losses: np.ndarray, beta: float) -> float:
    losses = np.asarray(losses, dtype=float).ravel()
    k = int(np.ceil((1.0 - beta) * len(losses)))
    if k <= 0:
        return float(np.max(losses))
    return float(np.sort(losses)[-k:].mean())


def portfolio_cvar(R: np.ndarray, w: np.ndarray, beta: float = BETA) -> float:
    return cvar_empirical(-(R @ w), beta)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def w_equal(R: np.ndarray) -> np.ndarray:
    n = R.shape[1]
    return np.ones(n) / n


def w_min_variance(R: np.ndarray) -> np.ndarray:
    n = R.shape[1]
    cov = np.cov(R.T)
    cov += 1e-8 * np.eye(n)
    x0 = np.ones(n) / n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(0.0, 1.0)] * n
    res = minimize(lambda w: w @ cov @ w,
                   x0, method="SLSQP",
                   bounds=bnds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9})
    return res.x if res.success else x0


def w_equal_risk(R: np.ndarray,
                 max_iter: int = 500,
                 tol: float = 1e-8) -> np.ndarray:
    """
    Equal risk contribution (ERC / proxy for Lohre DRP on assets), solved via
    Spinu (2013) cyclical coordinate descent:
        update w_i to root of   sigma_{i,i} w_i^2 + (Sigma w)_{-i} w_i - c/n = 0
    where c = w' Sigma w. Renormalise each pass.
    """
    n = R.shape[1]
    cov = np.cov(R.T) + 1e-8 * np.eye(n)
    inv_vol = 1.0 / np.sqrt(np.diag(cov))
    w = inv_vol / inv_vol.sum()

    for _ in range(max_iter):
        w_old = w.copy()
        c = w @ cov @ w
        for i in range(n):
            sigii = cov[i, i]
            off   = cov[i, :] @ w - sigii * w[i]
            disc  = off * off + 4.0 * sigii * (c / n)
            w[i]  = (-off + np.sqrt(disc)) / (2.0 * sigii)
        w = w / w.sum()
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w


def w_min_cvar(R: np.ndarray,
               beta: float = BETA,
               pos_cap: float | None = None) -> np.ndarray:
    """
    Rockafellar-Uryasev LP with long-only, fully-invested weights.
    Decision variables: x = [w_1..w_n, alpha, z_1..z_S].
        min  alpha + (1 / ((1 - beta) S)) sum_s z_s
        s.t. z_s >= -(R_s w) - alpha
             z_s >= 0
             sum w = 1
             0 <= w_i <= pos_cap
    """
    S, n = R.shape
    k = 1.0 / ((1.0 - beta) * S)

    c = np.concatenate([np.zeros(n), [1.0], np.full(S, k)])

    # -(R w) - alpha - z <= 0
    A_ub = np.zeros((S, n + 1 + S))
    A_ub[:, :n] = -R
    A_ub[:, n]  = -1.0
    A_ub[:, n + 1:] = -np.eye(S)
    b_ub = np.zeros(S)

    A_eq = np.zeros((1, n + 1 + S))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    w_upper = pos_cap if pos_cap is not None else 1.0
    bounds = ([(0.0, w_upper)] * n
              + [(None, None)]
              + [(0.0, None)] * S)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return np.ones(n) / n
    w = res.x[:n]
    w = np.maximum(w, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(n) / n


def w_min_cvar_capped(R: np.ndarray) -> np.ndarray:
    return w_min_cvar(R, beta=BETA, pos_cap=POS_CAP)


STRATEGIES = {
    "EqualWeight":  w_equal,
    "MinVar":       w_min_variance,
    "DRP":          w_equal_risk,
    "MinCVaR":      w_min_cvar,
    "MinCVaR_Cap":  w_min_cvar_capped,
}


# ---------------------------------------------------------------------------
# Backtest loop
# ---------------------------------------------------------------------------

def run_backtest(returns: pd.DataFrame,
                 window: int = WINDOW,
                 beta: float = BETA):
    dates = returns.index
    realized_rows = []
    weights_rows  = []

    prev_weights = {name: None for name in STRATEGIES}

    for t_idx in range(window, len(dates) - 1):
        train = returns.iloc[t_idx - window:t_idx]
        valid = train.columns[train.notna().mean() >= MIN_COVERAGE]
        if len(valid) < 5:
            continue
        train_clean = train[valid].fillna(0.0)
        R = train_clean.values
        next_row = returns.iloc[t_idx + 1][valid]
        if next_row.isna().any():
            next_row = next_row.fillna(0.0)
        next_ret = next_row.values
        next_date = dates[t_idx + 1]

        for name, strat_fn in STRATEGIES.items():
            w = strat_fn(R)
            realized    = float(next_ret @ w)
            cvar_ante   = portfolio_cvar(R, w, beta)
            prev = prev_weights[name]
            if prev is None:
                turnover = np.nan
            else:
                aligned_prev = prev.reindex(valid).fillna(0.0).values
                turnover = 0.5 * np.sum(np.abs(w - aligned_prev))
            realized_rows.append({
                "Date":          next_date,
                "Strategy":      name,
                "Realized":      realized,
                "CVaR_Ex_Ante":  cvar_ante,
                "Turnover":      turnover,
            })
            for asset, wi in zip(valid, w):
                weights_rows.append({
                    "Date":     next_date,
                    "Strategy": name,
                    "Asset":    asset,
                    "Weight":   float(wi),
                })
            prev_weights[name] = pd.Series(w, index=valid)

    return (pd.DataFrame(realized_rows),
            pd.DataFrame(weights_rows))


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def max_drawdown(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    dd    = (equity - peaks) / peaks
    return float(dd.min())


def summarise(df: pd.DataFrame, beta: float = BETA) -> pd.DataFrame:
    out = []
    for name, g in df.groupby("Strategy", sort=False):
        r = g["Realized"].values
        mu_ann  = r.mean() * ANN_FACTOR
        vol_ann = r.std(ddof=1) * np.sqrt(ANN_FACTOR)
        sharpe  = mu_ann / vol_ann if vol_ann > 0 else np.nan
        equity  = np.exp(np.cumsum(r))
        mdd     = max_drawdown(equity)
        calmar  = mu_ann / abs(mdd) if mdd < 0 else np.nan
        losses  = -r
        tail_mask = losses > g["CVaR_Ex_Ante"].values
        coverage  = tail_mask.mean()
        skew = pd.Series(r).skew()
        kurt = pd.Series(r).kurt()
        turnover = g["Turnover"].mean(skipna=True)
        out.append({
            "Strategy":     name,
            "Ann_Return":   mu_ann,
            "Ann_Vol":      vol_ann,
            "Sharpe":       sharpe,
            "Max_Drawdown": mdd,
            "Calmar":       calmar,
            "CVaR_Breach_Rate": coverage,
            "Nominal_Breach_Rate": 1.0 - beta,
            "Turnover":     turnover,
            "Skew":         skew,
            "Excess_Kurt":  kurt,
            "N_Months":     len(g),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[Phase 4] Rolling-window backtest")
    print(f"  returns : {LOG_RETURNS}")
    print(f"  window  : {WINDOW} months   beta = {BETA}   cap = {POS_CAP}\n")

    returns = load_returns(LOG_RETURNS)
    returns = returns.iloc[1:]   # drop the all-NaN first row
    print(f"  loaded  : {returns.shape[0]} months x {returns.shape[1]} ETFs\n")

    results, weights = run_backtest(returns)
    summary = summarise(results)

    res_path = OUT_DIR / "backtest_results.csv"
    wt_path  = OUT_DIR / "backtest_weights.csv"
    sum_path = OUT_DIR / "backtest_summary.csv"
    results.to_csv(res_path, index=False)
    weights.to_csv(wt_path,  index=False)
    summary.to_csv(sum_path, index=False)

    print("  Summary")
    print("  " + "-" * 70)
    with pd.option_context("display.float_format", "{:+.4f}".format,
                           "display.width", 120):
        print(summary.to_string(index=False))
    print()
    print(f"  results -> {res_path}")
    print(f"  weights -> {wt_path}")
    print(f"  summary -> {sum_path}")


if __name__ == "__main__":
    main()
