"""
etf_beta_weights.py
────────────────────
Per-ETF composite-factor weights derived from rolling OLS factor betas
of excess returns vs SPY benchmark.

Framework
─────────
For each ETF i, the rolling OLS already estimated in compute_benchmark_attribution()
produces a time series of factor exposures:

    r_{i,t}^{excess} = α_i + β_{C,i,t}·Carry_z + β_{V,i,t}·Value_z
                           + β_{M,i,t}·Mom_z   + β_{Q,i,t}·Quality_z + ε_{i,t}

The beta vector [β_C, β_V, β_M, β_Q]_i describes how much of ETF i's
active return (vs SPY) is driven by each style factor.

Weighting rationale
────────────────────
If a factor's beta is large and positive for ETF i, that factor is a dominant
driver of its active performance — it deserves a higher weight in the composite
score for that ETF.  Conversely, a near-zero or negative beta means the factor
adds little or even adverse signal.

Three transformations are available:

  1. abs_beta  (default)
     w_{i,f} ∝ |β_{i,f}|
     Magnitude of exposure, regardless of sign.  Appropriate when the composite
     is used as a ranking signal (not a directional bet): a large negative beta
     is still informative about the ETF's style profile.

  2. pos_beta
     w_{i,f} ∝ max(β_{i,f}, 0)
     Only reward factors where the ETF has a positive (long) loading.
     Appropriate when you want the composite to tilt toward factors the ETF
     is already exposed to in the intended direction.

  3. beta_t_stat
     w_{i,f} ∝ max(|t_{i,f}|, 0)  where t = β / se(β)
     Weight by statistical significance rather than raw magnitude.
     Discounts noisy betas estimated from few valid observations.
     Requires running the regression with standard errors (uses statsmodels
     OLS if available, else falls back to OLS residual-based se).

Beta-weight reconciliation with EFA × IC/IR
────────────────────────────────────────────
Optional: blend beta-weights with the EFA × IC/IR weights from
etf_factor_weights.py for a three-source reconciliation:

    w_{i,f} = γ · w^β_{i,f}  +  (1-γ) · w^{EFA×IC/IR}_{i,f}

    γ = 0   → pure EFA × IC/IR
    γ = 0.5 → equal blend (default)
    γ = 1   → pure beta

Academic references
────────────────────
  • Fama & MacBeth (1973) — two-pass cross-sectional regression linking
    betas to realized returns
  • Carhart (1997) — four-factor model OLS loadings as style attribution
  • Grinold & Kahn (1999) — betas as active risk drivers in multi-factor
    attribution
  • Barra / MSCI Fundamental Factor Model — factor exposures (betas) as
    the primary descriptor for composite risk attribution
  • Brinson, Hood & Beebower (1986) — attribution framework: factor betas
    explain >90% of return variation
  • Shu & Mulvey (arXiv 2024) — dynamic factor allocation via rolling
    exposure signals
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

warnings.filterwarnings("ignore")

FACTORS      = ["carry", "value", "momentum", "quality"]
BETA_COLS    = ["beta_Carry", "beta_Value", "beta_Momentum", "beta_Quality"]
FACTOR_MAP   = dict(zip(FACTORS, BETA_COLS))   # factor name → rolling_loadings col
MIN_OBS_BETA = 24   # minimum months of valid betas before trusting the estimate


# ─────────────────────────────────────────────────────────────────────────────
# 1.  EXTRACT LATEST / EWMA BETAS FROM ROLLING ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_beta_panel(
    rolling_loadings: dict,
    ewma_halflife: int = 36,
    method: str = "ewma",          # "latest" | "ewma" | "full_mean"
) -> pd.DataFrame:
    """
    Collapse the per-ticker rolling beta time series into a single
    (N_tickers × 4) summary matrix.

    Parameters
    ----------
    rolling_loadings : dict  {ticker: DataFrame(Date × params)}
                       Output of compute_benchmark_attribution()
    ewma_halflife    : int   Half-life in months for EWMA averaging
    method           : str
        "latest"    — most recent rolling-window beta (point estimate)
        "ewma"      — exponentially weighted mean of the full history
                      (up-weights recent periods, moderates noise)
        "full_mean" — simple equally-weighted historical mean beta

    Returns
    -------
    pd.DataFrame  (N_tickers × 4)  columns: carry, value, momentum, quality
    """
    records = {}
    for ticker, df in rolling_loadings.items():
        if df is None or len(df) < MIN_OBS_BETA:
            records[ticker] = pd.Series(np.nan, index=FACTORS)
            continue

        row = {}
        for factor, beta_col in FACTOR_MAP.items():
            if beta_col not in df.columns:
                row[factor] = np.nan
                continue

            series = df[beta_col].dropna()
            if len(series) == 0:
                row[factor] = np.nan
                continue

            if method == "latest":
                row[factor] = series.iloc[-1]

            elif method == "ewma":
                # EWMA: weight t proportional to 0.5^((T-t)/halflife)
                T     = len(series)
                idx   = np.arange(T)
                wts   = 0.5 ** ((T - 1 - idx) / ewma_halflife)
                wts  /= wts.sum()
                row[factor] = float(np.dot(wts, series.values))

            else:  # full_mean
                row[factor] = float(series.mean())

        records[ticker] = pd.Series(row)

    beta_df = pd.DataFrame(records).T.reindex(columns=FACTORS)
    return beta_df


def extract_beta_se_panel(
    rolling_loadings: dict,
    log_ret: pd.DataFrame,
    factor_panels: dict,
    benchmark: str = "SPY",
    window: int = 60,
) -> pd.DataFrame:
    """
    Re-run the last window of rolling OLS for each ticker and return the
    standard errors of the four factor betas, used for t-stat weighting.

    This is lightweight: only the most recent `window` observations are used.

    Returns
    -------
    pd.DataFrame  (N_tickers × 4)  beta standard errors
    """
    se_records = {}

    # Identify common time index
    panels = [factor_panels[f] for f in FACTORS if f in factor_panels]
    if not panels:
        return pd.DataFrame()

    common_idx = log_ret.index
    for p in panels:
        common_idx = common_idx.intersection(p.index)

    common_idx = common_idx[-window:]   # most recent window

    bmk_ret = log_ret[benchmark].reindex(common_idx).fillna(0) \
              if benchmark in log_ret.columns else pd.Series(0.0, index=common_idx)

    for ticker in rolling_loadings.keys():
        if ticker not in log_ret.columns:
            se_records[ticker] = pd.Series(np.nan, index=FACTORS)
            continue

        excess = (log_ret[ticker] - bmk_ret).reindex(common_idx).values.astype(float)

        Xf = np.column_stack([
            factor_panels[f][ticker].reindex(common_idx).values.astype(float)
            if f in factor_panels and ticker in factor_panels[f].columns
            else np.full(len(common_idx), np.nan)
            for f in FACTORS
        ])

        valid = ~(np.isnan(excess) | np.any(np.isnan(Xf), axis=1))
        if valid.sum() < 10:
            se_records[ticker] = pd.Series(np.nan, index=FACTORS)
            continue

        X_aug = np.column_stack([np.ones(valid.sum()), Xf[valid]])
        y     = excess[valid]

        try:
            betas, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            residuals = y - X_aug @ betas
            n, k = X_aug.shape
            s2   = np.dot(residuals, residuals) / (n - k)
            XTX_inv = np.linalg.pinv(X_aug.T @ X_aug)
            se_full  = np.sqrt(np.maximum(s2 * np.diag(XTX_inv), 0))
            se_betas = se_full[1:]   # drop intercept se
        except Exception:
            se_records[ticker] = pd.Series(np.nan, index=FACTORS)
            continue

        se_records[ticker] = pd.Series(se_betas, index=FACTORS)

    return pd.DataFrame(se_records).T.reindex(columns=FACTORS)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  WEIGHT TRANSFORMATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_rows(df: pd.DataFrame, floor: float = 0.0) -> pd.DataFrame:
    """Row-normalise so each row sums to 1.  Floor values below `floor`."""
    df = df.clip(lower=floor)
    row_sums = df.sum(axis=1)
    zero_mask = row_sums < 1e-8
    df.loc[zero_mask] = 0.25          # equal fallback
    return df.div(df.sum(axis=1), axis=0)


def beta_weights_abs(beta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Weight ∝ |β_{i,f}|  — absolute magnitude.

    Rationale: a large negative beta (e.g. Value very negative for a growth
    ETF like IWF) still signals strong factor sensitivity.  The composite
    should reflect this sensitivity even though its direction opposes the
    typical positive-score interpretation.  Sign inversion is handled
    downstream by the z-score orientation convention.
    """
    return _normalise_rows(beta_df.abs())


def beta_weights_positive(beta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Weight ∝ max(β_{i,f}, 0)  — positive exposures only.

    Rationale: if an ETF has a negative beta to Momentum, amplifying
    the Momentum z-score in its composite would reward *low momentum*,
    which contradicts the factor's intended direction.  Setting weight=0
    for negative betas prevents this directional flip.
    """
    return _normalise_rows(beta_df, floor=0.0)


def beta_weights_tstat(
    beta_df: pd.DataFrame,
    se_df:   pd.DataFrame,
    dof:     int = 58,
) -> pd.DataFrame:
    """
    Weight ∝ max(|t_{i,f}|, 0)  where  t_{i,f} = β_{i,f} / se(β_{i,f}).

    Rationale: a beta estimated from 60 months has uncertainty.  Two ETFs
    may have similar betas but very different confidence levels — a highly
    significant beta (high |t|) is more informative for weighting than a
    noisy estimate.  This approach is consistent with Fama-MacBeth (1973)
    standard practice of using t-statistics to assess factor reliability.

    Parameters
    ----------
    beta_df : (N × 4) OLS betas
    se_df   : (N × 4) OLS standard errors of betas
    dof     : degrees of freedom for t-distribution (window - 5)
    """
    t_stats = (beta_df / se_df.replace(0, np.nan)).abs()
    t_stats = t_stats.fillna(0.0)
    return _normalise_rows(t_stats)


def beta_weights_r2_contribution(
    rolling_loadings: dict,
    beta_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Weight ∝ variance-explained contribution of each factor to excess returns.

    For ticker i:
        VarContrib_{i,f} = β_{i,f}² × Var(z_f)

    Since the z-scores are cross-sectionally standardised (unit variance by
    construction), Var(z_f) ≈ 1 for all f, so:
        VarContrib_{i,f} ≈ β_{i,f}²

    This is the Pratt decomposition of R²:  each factor's weight is
    proportional to its squared beta, i.e. its contribution to the fitted
    variance.  Closely related to the MSCI Barra concept of "factor R²
    contribution".

    Reference: Pratt (1987) "Dividing the Indivisible" — dominance analysis.
    """
    return _normalise_rows(beta_df ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def compute_beta_weights(
    out: dict,
    transformation: str  = "abs",    # "abs" | "positive" | "tstat" | "r2"
    beta_method:    str  = "ewma",   # "latest" | "ewma" | "full_mean"
    ewma_halflife:  int  = 36,
    verbose:        bool = True,
) -> pd.DataFrame:
    """
    Derive per-ETF factor weights from the rolling OLS beta attribution
    already stored in the pipeline output dict.

    Parameters
    ----------
    out             : pipeline output from run_pipeline() (must contain
                      "rolling_betas" and "log_ret" keys)
    transformation  : how to convert betas into non-negative weights
        "abs"       — |β| normalised                    (default)
        "positive"  — max(β, 0) normalised
        "tstat"     — |t-stat| normalised (needs se recomputation)
        "r2"        — β² normalised  (variance-explained contribution)
    beta_method     : how to summarise the rolling beta history
        "latest"    — most recent rolling estimate
        "ewma"      — EWMA-smoothed mean (recommended)
        "full_mean" — simple historical mean
    ewma_halflife   : months for EWMA smoothing of beta history
    verbose         : print diagnostic table

    Returns
    -------
    pd.DataFrame  (N_tickers × 4)  normalised per-ETF factor weights,
                  columns: carry, value, momentum, quality
    """
    rolling_betas = out.get("rolling_betas", {})
    if not rolling_betas:
        raise ValueError(
            "out['rolling_betas'] is empty — run compute_benchmark_attribution() first."
        )

    # ── Step 1: collapse rolling betas to (N × 4) matrix ─────────────────
    beta_df = extract_beta_panel(rolling_betas, ewma_halflife, beta_method)

    # ── Step 2: apply selected transformation ────────────────────────────
    if transformation == "abs":
        weights = beta_weights_abs(beta_df)

    elif transformation == "positive":
        weights = beta_weights_positive(beta_df)

    elif transformation == "r2":
        weights = beta_weights_r2_contribution(rolling_betas, beta_df)

    elif transformation == "tstat":
        factor_panels = {
            f: out.get(f"{f}_z", pd.DataFrame()) for f in FACTORS
        }
        log_ret  = out.get("log_ret", pd.DataFrame())
        se_df    = extract_beta_se_panel(rolling_betas, log_ret, factor_panels)
        weights  = beta_weights_tstat(beta_df, se_df)

    else:
        raise ValueError(
            f"Unknown transformation '{transformation}'. "
            "Choose: 'abs', 'positive', 'tstat', 'r2'"
        )

    # ── Step 3: handle tickers with all-NaN betas ─────────────────────────
    all_nan_rows = weights.isna().all(axis=1)
    weights.loc[all_nan_rows] = 0.25

    if verbose:
        print(f"\n  Beta-derived weights  "
              f"(transformation={transformation}, beta={beta_method}, "
              f"halflife={ewma_halflife}M)")
        print("─" * 60)
        print(f"  {'Ticker':<8}  {'Carry':>7}  {'Value':>7}  "
              f"{'Momentum':>9}  {'Quality':>8}  {'Dominant factor'}")
        print("─" * 60)
        for ticker, row in weights.iterrows():
            dominant = row.idxmax()
            print(f"  {ticker:<8}  {row['carry']:7.3f}  {row['value']:7.3f}  "
                  f"{row['momentum']:9.3f}  {row['quality']:8.3f}  {dominant}")
        print("─" * 60)
        print(f"  Mean:    {weights.mean().to_dict()}")

    return weights.round(4)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  RECONCILIATION WITH EFA × IC/IR WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def reconcile_beta_with_efa_icir(
    beta_weights:       pd.DataFrame,
    efa_icir_weights:   pd.DataFrame,
    gamma:              float = 0.5,
) -> pd.DataFrame:
    """
    Convex blend of beta-based and EFA × IC/IR based weights.

        w_{i,f} = γ · w^β_{i,f}  +  (1-γ) · w^{EFA×ICIR}_{i,f}

    Parameters
    ----------
    beta_weights      : (N × 4) from compute_beta_weights()
    efa_icir_weights  : (N × 4) from etf_factor_weights.compute_adaptive_weights()
    gamma             : blend weight on beta (0 = pure EFA/ICIR, 1 = pure beta)

    Returns
    -------
    pd.DataFrame  (N × 4) blended, row-normalised weights
    """
    common_idx = beta_weights.index.intersection(efa_icir_weights.index)
    b  = beta_weights.reindex(common_idx)
    e  = efa_icir_weights.reindex(common_idx)
    blended = gamma * b + (1 - gamma) * e
    blended = blended.clip(lower=0.0)
    # row-normalise
    row_sums = blended.sum(axis=1)
    zero_mask = row_sums < 1e-8
    blended.loc[zero_mask] = 0.25
    return blended.div(blended.sum(axis=1), axis=0).round(4)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DIAGNOSTIC UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def beta_weight_diagnostics(out: dict, weights: pd.DataFrame) -> pd.DataFrame:
    """
    For each ETF, show: raw β, weight, and the latest R² of the rolling OLS.

    Returns a DataFrame with columns:
        beta_carry, beta_value, beta_momentum, beta_quality,
        w_carry, w_value, w_momentum, w_quality,
        alpha, last_r2 (approx, from residual variance)
    """
    rolling_betas  = out.get("rolling_betas", {})
    latest_loading = out.get("latest_loadings", pd.DataFrame())

    rows = {}
    for ticker, df in rolling_betas.items():
        if df is None or len(df) == 0:
            continue
        last = df.iloc[-1]
        r = {
            "beta_carry":    last.get("beta_Carry",    np.nan),
            "beta_value":    last.get("beta_Value",    np.nan),
            "beta_momentum": last.get("beta_Momentum", np.nan),
            "beta_quality":  last.get("beta_Quality",  np.nan),
            "alpha":         last.get("alpha",         np.nan),
        }
        if ticker in weights.index:
            for f in FACTORS:
                r[f"w_{f}"] = weights.loc[ticker, f]
        rows[ticker] = r

    diag = pd.DataFrame(rows).T
    return diag.round(4)


def asset_class_beta_profile(
    beta_df: pd.DataFrame,
    asset_class: dict,
) -> pd.DataFrame:
    """
    Average beta magnitude per asset class — reveals which factors
    structurally drive each ETF type.
    """
    ac = pd.Series(asset_class).reindex(beta_df.index)
    result = (beta_df.abs()
              .assign(_ac=ac)
              .groupby("_ac")[FACTORS]
              .mean()
              .round(3))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DROP-IN COMPOSITE USING BETA WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def compute_beta_composite(
    out:     dict,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute composite score using per-ETF beta-derived weights.

        Score_{i,t} = Σ_f  w_{i,f} · z_{f,i,t}

    Identical API to compute_adaptive_composite() in etf_factor_weights.py.
    """
    composite = pd.DataFrame()
    for f in FACTORS:
        z_panel = out.get(f"{f}_z", pd.DataFrame())
        if z_panel.empty:
            continue
        w_f     = weights[f]
        weighted = z_panel.multiply(w_f, axis="columns")
        composite = weighted if composite.empty else composite.add(weighted, fill_value=0.0)
    return composite


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — demonstration with synthetic data
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/workspace")

    print("Building synthetic rolling_betas for demonstration …\n")

    rng     = np.random.default_rng(0)
    T, N    = 200, 8
    dates   = pd.date_range("2006-04-30", periods=T, freq="ME")
    tickers = ["SPY","IWD","IWF","TLT","GLD","XLK","XLE","XLP"]
    non_bmk = tickers[1:]

    FACTORS_CAPS = ["Carry", "Value", "Momentum", "Quality"]

    # Simulate rolling betas with realistic asset-class patterns
    true_betas = {
        "IWD": [0.05, 0.40, 0.20, 0.15],   # value tilt
        "IWF": [0.05, -0.30, 0.35, 0.20],  # growth: negative value, positive momentum
        "TLT": [0.30, 0.10, -0.10, 0.05],  # carry-driven, negative momentum
        "GLD": [0.00, 0.00, 0.25, 0.05],   # carry NaN, momentum dominant
        "XLK": [0.05, -0.10, 0.45, 0.25],  # tech: momentum + quality
        "XLE": [0.10, 0.20, 0.15, 0.05],   # energy: value + carry
        "XLP": [0.20, 0.10, 0.05, 0.20],   # staples: carry + quality
    }

    rolling_betas_mock = {}
    for ticker in non_bmk:
        tb = true_betas[ticker]
        rows = []
        for i in range(60, T):
            noisy = tb + rng.normal(0, 0.05, 4)
            rows.append({
                "Date":           dates[i],
                "alpha":          rng.normal(0, 0.002),
                "beta_Carry":     noisy[0],
                "beta_Value":     noisy[1],
                "beta_Momentum":  noisy[2],
                "beta_Quality":   noisy[3],
            })
        rolling_betas_mock[ticker] = (
            pd.DataFrame(rows).set_index("Date")
        )

    # z-scored factor panels
    factor_panels = {
        f: pd.DataFrame(rng.standard_normal((T, N)),
                        index=dates, columns=tickers)
        for f in FACTORS
    }
    log_ret = pd.DataFrame(rng.standard_normal((T, N)) * 0.04,
                           index=dates, columns=tickers)

    out_mock = {
        "rolling_betas":   rolling_betas_mock,
        "latest_loadings": pd.DataFrame(
            {t: pd.Series(rolling_betas_mock[t].iloc[-1]) for t in non_bmk}
        ).T,
        "log_ret": log_ret,
        **{f"{f}_z": factor_panels[f] for f in FACTORS},
    }

    print("=== Transformation: abs (|β| normalised) ===")
    w_abs = compute_beta_weights(out_mock, transformation="abs",
                                 beta_method="ewma", verbose=True)

    print("\n=== Transformation: r2 (β² normalised) ===")
    w_r2  = compute_beta_weights(out_mock, transformation="r2",
                                 beta_method="ewma", verbose=False)
    print(w_r2.to_string())
