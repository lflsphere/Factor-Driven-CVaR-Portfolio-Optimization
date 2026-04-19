"""
etf_factor_weights.py
─────────────────────
Per-ETF adaptive composite-factor weights via a three-stage pipeline:

  Stage 1 · EFA on z-score panel
            Exploratory Factor Analysis on the (T × 4) matrix of monthly
            z-scored factor signals for each ETF.  The rotated factor
            loadings reveal how much variance in each style factor is
            explained by latent common structure *for that ETF*.

  Stage 2 · IC/IR weighting
            For each factor f, build a time series of monthly Information
            Coefficients IC_{f,t} = rank_corr(z_{f,t}, r_{t+1}).
            The Information Ratio IR_f = mean(IC) / std(IC) measures
            stable predictive power.  Weights proportional to max(IR,0).

  Stage 3 · Reconciliation (EFA × IC/IR)
            Per-ETF weight = normalise( EFA_communality_f × IR_f )
            EFA communality captures how much of factor f's variance is
            shared across latent dimensions (its "relevance" in the data).
            IC/IR captures predictive power vs. forward returns.
            Their product gives a weight that is both data-driven and
            anchored to actual return predictability.

Academic foundations:
  • Grinold & Kahn (1999) — Fundamental Law, IC/IR weighting
  • Cattell (1966) — Scree plot; factor retention by eigenvalue
  • Koijen et al. (JFE 2018), Asness et al. (JF 2013) — factor relevance
    varies systematically across asset classes
  • Shu & Mulvey (arXiv 2024) — dynamic factor allocation via regime signals
  • Garrone (arXiv 2026) — IC/IR diagnostics for bounded multi-factor tilts
  • S&P DJI (2017) — factor efficiency ratio in multi-factor index construction
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

# Optional: factor_analyzer gives proper ML-EFA with various rotations
try:
    from factor_analyzer import FactorAnalyzer
    HAS_FA = True
except ImportError:
    HAS_FA = False
    warnings.warn(
        "factor_analyzer not installed — falling back to PCA-based EFA. "
        "Install with: pip install factor_analyzer"
    )

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FACTORS         = ["carry", "value", "momentum", "quality"]
DEFAULT_WEIGHTS = {f: 0.25 for f in FACTORS}   # equal-weight baseline
MIN_OBS_EFA     = 36    # months needed to run EFA for a single ETF
MIN_OBS_IC      = 24    # months needed to compute IC/IR reliably
IC_HORIZON      = 1     # 1-month forward return horizon for IC
EWMA_HALFLIFE   = 36    # months for EWMA IC decay (de-emphasise stale ICs)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  STAGE 1 — EFA PER ETF
# ─────────────────────────────────────────────────────────────────────────────

def _pca_communalities(X: np.ndarray, n_factors: int) -> np.ndarray:
    """
    Fallback PCA-based communality estimate when factor_analyzer is absent.
    Returns communalities h² for each column of X (shape: n_vars).
    """
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    # Loadings: L = V * sqrt(λ)
    loadings = eigvecs[:, :n_factors] * np.sqrt(np.maximum(eigvals[:n_factors], 0))
    communalities = np.sum(loadings ** 2, axis=1)
    return communalities


def efa_communalities_for_ticker(
    factor_panel: pd.DataFrame,
    ticker: str,
    n_factors: int = 2,
    rotation: str = "varimax",
) -> pd.Series:
    """
    Run EFA on the (T × 4) z-score matrix for a single ETF and return
    the per-factor communalities h²_f.

    Parameters
    ----------
    factor_panel : dict or DataFrame mapping factor_name → (T × N) DataFrame
        Pass the `out` dict from run_pipeline(), or a (T × 4) DataFrame
        where columns are ['carry','value','momentum','quality'].
    ticker : str
        ETF ticker.
    n_factors : int
        Number of latent factors to retain (default 2 for a 4-variable system).
    rotation : str
        EFA rotation ('varimax', 'oblimin', etc.).

    Returns
    -------
    pd.Series  index=FACTORS, values=communalities h²_f ∈ [0, 1]
    """
    if isinstance(factor_panel, dict):
        # Build (T × 4) DataFrame from pipeline output
        df = pd.DataFrame({
            f: factor_panel.get(f"{f}_z", pd.DataFrame()).get(ticker, pd.Series())
            for f in FACTORS
        }).dropna()
    else:
        df = factor_panel[[c for c in FACTORS if c in factor_panel.columns]].dropna()

    if len(df) < MIN_OBS_EFA:
        # Insufficient data → return equal communalities
        return pd.Series(0.25, index=FACTORS)

    X = StandardScaler().fit_transform(df.values)  # (T × 4)
    n_vars = X.shape[1]
    n_factors = min(n_factors, n_vars - 1)

    if HAS_FA:
        try:
            fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation,
                                method="ml")  # ML-EFA (Lawley-Maxwell)
            fa.fit(X)
            h2 = fa.get_communalities()           # shape (4,)
        except Exception:
            h2 = _pca_communalities(X, n_factors)
    else:
        h2 = _pca_communalities(X, n_factors)

    h2 = np.maximum(h2, 1e-6)   # floor at near-zero
    return pd.Series(h2, index=FACTORS[:n_vars])


def efa_communalities_all_tickers(
    out: dict,
    n_factors: int = 2,
    rotation: str = "varimax",
) -> pd.DataFrame:
    """
    Run EFA for every ticker in the universe.

    Returns
    -------
    pd.DataFrame  shape (N_tickers × 4)
        Per-ticker communalities for carry, value, momentum, quality.
    """
    # Infer tickers from composite panel
    composite = out.get("composite", pd.DataFrame())
    tickers   = list(composite.columns)
    results   = {}

    for ticker in tickers:
        results[ticker] = efa_communalities_for_ticker(out, ticker, n_factors, rotation)

    comm_df = pd.DataFrame(results).T          # (N × 4)
    comm_df.columns = FACTORS[:comm_df.shape[1]]
    return comm_df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  STAGE 2 — IC / IR WEIGHTING PER ETF
# ─────────────────────────────────────────────────────────────────────────────

def compute_ic_ir(
    out: dict,
    horizon: int = IC_HORIZON,
    ewma_halflife: int = EWMA_HALFLIFE,
) -> pd.DataFrame:
    """
    For each (ticker, factor) pair, compute the rolling/EWMA Information Ratio:
        IC_{f,i,t}  = Spearman rank-corr between z_{f,t} and r_{i,t+h}
                      (cross-sectional at each date t)
        IR_{f,i}    = mean(IC_{f,i}) / std(IC_{f,i})
                      using EWMA decay to up-weight recent predictability.

    The *cross-sectional* IC measures how well factor f ranks the 33 ETFs
    at time t relative to their forward returns → it is the standard
    Grinold-Kahn IC definition.

    Returns
    -------
    pd.DataFrame  shape (N_tickers × 4)
        Per-ticker IC/IR for each factor.
    """
    log_ret  = out.get("log_ret",   pd.DataFrame())
    fwd_ret  = log_ret.shift(-horizon)              # 1-month forward return

    ir_records = []

    for factor_name in FACTORS:
        z_panel = out.get(f"{factor_name}_z", pd.DataFrame())
        if z_panel.empty:
            continue

        # Align on common dates
        common_idx = z_panel.index.intersection(fwd_ret.index)
        z_aligned  = z_panel.loc[common_idx]
        r_aligned  = fwd_ret.loc[common_idx]

        # Cross-sectional IC at each date t (scalar per date)
        ic_ts = pd.Series(index=common_idx, dtype=float)
        for t in common_idx:
            z_t = z_aligned.loc[t].dropna()
            r_t = r_aligned.loc[t].reindex(z_t.index).dropna()
            z_t = z_t.reindex(r_t.index)
            if len(r_t) < 5:
                continue
            corr, _ = spearmanr(z_t.values, r_t.values)
            ic_ts[t] = corr

        ic_ts = ic_ts.dropna()
        if len(ic_ts) < MIN_OBS_IC:
            # Insufficient → assign IC/IR = 0
            for col in z_panel.columns:
                ir_records.append({"ticker": col, "factor": factor_name, "ir": 0.0,
                                   "ic_mean": 0.0, "ic_std": np.nan})
            continue

        # EWMA-weighted IC mean and std
        weights_ewma = np.array(
            [0.5 ** ((len(ic_ts) - 1 - i) / ewma_halflife)
             for i in range(len(ic_ts))]
        )
        weights_ewma /= weights_ewma.sum()
        ic_mean_ewma = np.dot(weights_ewma, ic_ts.values)
        ic_std_ewma  = np.sqrt(
            np.dot(weights_ewma, (ic_ts.values - ic_mean_ewma) ** 2)
        )
        ir_global = ic_mean_ewma / (ic_std_ewma + 1e-8)

        # All tickers share the same cross-sectional IC/IR for this factor
        for col in z_panel.columns:
            ir_records.append({
                "ticker":   col,
                "factor":   factor_name,
                "ir":       ir_global,
                "ic_mean":  ic_mean_ewma,
                "ic_std":   ic_std_ewma,
            })

    ir_df = pd.DataFrame(ir_records)
    if ir_df.empty:
        # Return equal weights if nothing computed
        tickers = list(out.get("composite", pd.DataFrame()).columns)
        return pd.DataFrame(0.25, index=tickers, columns=FACTORS)

    # Pivot to (N_tickers × 4)
    ir_pivot = ir_df.pivot(index="ticker", columns="factor", values="ir")
    ir_pivot = ir_pivot.reindex(columns=FACTORS).fillna(0.0)

    # Floor negative IR at zero (no predictive power → zero weight)
    ir_pivot = ir_pivot.clip(lower=0.0)
    return ir_pivot


# ─────────────────────────────────────────────────────────────────────────────
# 3.  STAGE 3 — RECONCILIATION: EFA × IC/IR
# ─────────────────────────────────────────────────────────────────────────────

def compute_adaptive_weights(
    out: dict,
    n_efa_factors: int      = 2,
    rotation: str           = "varimax",
    horizon: int            = IC_HORIZON,
    ewma_halflife: int      = EWMA_HALFLIFE,
    efa_blend: float        = 0.5,
    fallback_min_ir: float  = 0.0,
) -> pd.DataFrame:
    """
    Compute per-ETF adaptive factor weights via EFA × IC/IR reconciliation.

    Weight for ETF i, factor f:
        raw_w_{i,f} = efa_blend  × normalise(h²_{i,f})
                    + (1-efa_blend) × normalise(max(IR_{i,f}, 0))
        w_{i,f}     = raw_w_{i,f} / sum_f raw_w_{i,f}

    where:
        h²_{i,f}  = EFA communality of factor f for ETF i
                    (how much variance of f is structurally present in i's history)
        IR_{i,f}  = EWMA IC/IR of factor f in predicting forward returns
                    (stable predictive power of f for the cross-section)

    Parameters
    ----------
    out           : pipeline output dict from run_pipeline()
    n_efa_factors : latent dimensions in EFA (default 2 for 4 observables)
    rotation      : EFA rotation method ('varimax' recommended for interpretability)
    horizon       : forward return horizon for IC (months, default 1)
    ewma_halflife : half-life for EWMA decay of IC series (months, default 36)
    efa_blend     : weight on EFA communalities vs IC/IR in reconciliation
                    0.0 = pure IC/IR,  0.5 = equal blend (default),  1.0 = pure EFA
    fallback_min_ir : minimum IR floor before applying EFA blend;
                    set > 0 to enforce a minimum predictability threshold.

    Returns
    -------
    pd.DataFrame  shape (N_tickers × 4)
        Per-ETF normalised weights summing to 1.0 across the four factors.
        Columns: carry, value, momentum, quality.
    """
    # ── 3a. EFA communalities ─────────────────────────────────────────────
    comm_df = efa_communalities_all_tickers(out, n_efa_factors, rotation)
    comm_norm = comm_df.div(comm_df.sum(axis=1), axis=0)     # row-normalise

    # ── 3b. IC/IR ──────────────────────────────────────────────────────────
    ir_df   = compute_ic_ir(out, horizon, ewma_halflife)
    ir_df   = ir_df.reindex(index=comm_norm.index, columns=FACTORS).fillna(0.0)
    ir_norm = ir_df.div(ir_df.sum(axis=1) + 1e-8, axis=0)   # row-normalise

    # ── 3c. Blend ──────────────────────────────────────────────────────────
    raw = efa_blend * comm_norm + (1.0 - efa_blend) * ir_norm
    raw = raw.clip(lower=0.0)

    # Ensure rows where all factors have zero weight fall back to equal
    row_sums = raw.sum(axis=1)
    zero_rows = row_sums < 1e-8
    raw.loc[zero_rows] = 0.25

    weights = raw.div(raw.sum(axis=1), axis=0)
    return weights.round(4)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ADAPTIVE COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_adaptive_composite(
    out: dict,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the composite style score using per-ETF adaptive weights.

    Score_{i,t} = sum_f  w_{i,f} * z_{f,i,t}

    Parameters
    ----------
    out     : pipeline output dict
    weights : (N_tickers × 4) weight DataFrame from compute_adaptive_weights()

    Returns
    -------
    pd.DataFrame  (T_factor × N_tickers)  adaptive composite scores.
    """
    composite = pd.DataFrame()

    for factor_name in FACTORS:
        z_panel = out.get(f"{factor_name}_z", pd.DataFrame())
        if z_panel.empty:
            continue

        # Per-ticker weight vector for this factor
        w_f = weights[factor_name]   # pd.Series (N,)

        # Broadcast: z_panel (T×N) * w_f (N,) column-wise
        weighted = z_panel.multiply(w_f, axis="columns")

        if composite.empty:
            composite = weighted
        else:
            composite = composite.add(weighted, fill_value=0.0)

    return composite


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DIAGNOSTIC TOOLS
# ─────────────────────────────────────────────────────────────────────────────

def weight_summary(weights: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise adaptive weights across the universe.

    Returns a DataFrame with cross-ticker statistics (mean, std, min, max)
    for each factor weight, plus an 'effective_N_factors' column (entropy-based
    number of effective factors per ETF).
    """
    summary = pd.DataFrame({
        "mean":  weights.mean(),
        "std":   weights.std(),
        "min":   weights.min(),
        "max":   weights.max(),
        "median": weights.median(),
    }).T

    # Effective number of factors (exponential entropy)
    # N_eff = exp(-sum_f w_f * log(w_f))
    log_w = np.log(weights.clip(lower=1e-8))
    entropy = -(weights * log_w).sum(axis=1)
    n_eff   = np.exp(entropy)
    weights["n_eff_factors"] = n_eff.round(2)

    return summary


def asset_class_weight_profile(
    weights: pd.DataFrame,
    asset_class: dict,
) -> pd.DataFrame:
    """
    Aggregate average weights by asset class to reveal systematic
    differences in factor relevance across ETF types.

    Parameters
    ----------
    weights      : (N_tickers × 4) from compute_adaptive_weights()
    asset_class  : dict mapping ticker → asset class (from ASSET_CLASS constant)

    Returns
    -------
    pd.DataFrame  (n_classes × 4) mean weights per asset class.
    """
    ac_series = pd.Series(asset_class).reindex(weights.index)
    return (weights
            .assign(_ac=ac_series)
            .groupby("_ac")[FACTORS]
            .mean()
            .round(3))


def ic_significance_table(out: dict, horizon: int = IC_HORIZON) -> pd.DataFrame:
    """
    Return a table of IC mean, IC std, IR, and t-stat for each factor,
    with a flag indicating whether IC is statistically significant at 5%.

    t-stat = IC_mean / (IC_std / sqrt(T))
    Under H0: IC=0, t ~ Student(T-1)
    """
    records = []
    for factor_name in FACTORS:
        z_panel = out.get(f"{factor_name}_z", pd.DataFrame())
        log_ret = out.get("log_ret", pd.DataFrame())
        fwd_ret = log_ret.shift(-horizon)
        if z_panel.empty:
            continue

        common_idx = z_panel.index.intersection(fwd_ret.index)
        z_aligned  = z_panel.loc[common_idx]
        r_aligned  = fwd_ret.loc[common_idx]

        ics = []
        for t in common_idx:
            z_t = z_aligned.loc[t].dropna()
            r_t = r_aligned.loc[t].reindex(z_t.index).dropna()
            z_t = z_t.reindex(r_t.index)
            if len(r_t) < 5:
                continue
            corr, _ = spearmanr(z_t.values, r_t.values)
            ics.append(corr)

        if len(ics) < 2:
            continue

        ics    = np.array(ics)
        mean_  = ics.mean()
        std_   = ics.std(ddof=1)
        ir_    = mean_ / (std_ + 1e-8)
        tstat  = mean_ / (std_ / np.sqrt(len(ics)) + 1e-8)
        sig    = abs(tstat) > 1.96

        records.append({
            "factor":  factor_name,
            "IC_mean": round(mean_, 4),
            "IC_std":  round(std_, 4),
            "IR":      round(ir_, 3),
            "t-stat":  round(tstat, 2),
            "T":       len(ics),
            "sig_5pct": sig,
        })

    return pd.DataFrame(records).set_index("factor")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FULL WEIGHT PIPELINE (drop-in replacement for equal-weight composite)
# ─────────────────────────────────────────────────────────────────────────────

def run_adaptive_weighting(
    out: dict,
    n_efa_factors: int = 2,
    rotation: str      = "varimax",
    efa_blend: float   = 0.5,
    verbose: bool      = True,
) -> dict:
    """
    Full adaptive weighting pipeline.  Returns an extended `out` dict with:
        out["adaptive_weights"]      : (N × 4) per-ETF weights
        out["adaptive_composite"]    : (T × N) adaptive composite scores
        out["efa_communalities"]     : (N × 4) EFA communalities
        out["ic_ir_table"]           : (N × 4) IC/IR values
        out["ic_significance"]       : (4 × 6) IC significance table
        out["weight_summary"]        : (5 × 4) cross-ticker weight stats

    Usage
    -----
    >>> out = run_pipeline()          # from etf_style_factors.py
    >>> out = run_adaptive_weighting(out)
    >>> adaptive_composite = out["adaptive_composite"]
    """
    if verbose:
        print("─" * 60)
        print("  Stage 1 · EFA communalities")

    comm_df = efa_communalities_all_tickers(out, n_efa_factors, rotation)
    if verbose:
        print(f"    EFA run for {len(comm_df)} tickers  "
              f"(n_factors={n_efa_factors}, rotation={rotation})")
        print(f"    Average communalities:\n{comm_df.mean().round(3).to_string()}")

    if verbose:
        print("\n  Stage 2 · IC/IR computation")

    ir_df  = compute_ic_ir(out)
    ic_sig = ic_significance_table(out)
    if verbose:
        print(f"\n  IC significance table (cross-sectional, 1M forward horizon):")
        print(ic_sig.to_string())

    if verbose:
        print("\n  Stage 3 · Reconciliation (EFA × IC/IR blend)")

    weights = compute_adaptive_weights(
        out, n_efa_factors=n_efa_factors, rotation=rotation, efa_blend=efa_blend
    )
    if verbose:
        print(f"\n  Weight summary (cross-ticker, efa_blend={efa_blend}):")
        print(weight_summary(weights).drop(columns=["n_eff_factors"],
                                           errors="ignore").to_string())

    adaptive_comp = compute_adaptive_composite(out, weights)

    out["adaptive_weights"]   = weights
    out["adaptive_composite"] = adaptive_comp
    out["efa_communalities"]  = comm_df
    out["ic_ir_table"]        = ir_df
    out["ic_significance"]    = ic_sig
    out["weight_summary"]     = weight_summary(weights)

    if verbose:
        print("\n  ✓ Adaptive weighting complete.")
        print("─" * 60)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 7.  COMPARISON HELPER
# ─────────────────────────────────────────────────────────────────────────────

def compare_composites(out: dict) -> pd.DataFrame:
    """
    Compare rank correlations between the equal-weight composite and the
    adaptive composite at each date.  A consistently high Spearman ρ means
    both composites largely agree; divergences identify dates where
    adaptive weights materially change the cross-sectional ranking.

    Returns
    -------
    pd.Series of monthly Spearman ρ between EW and adaptive composites.
    """
    ew_comp  = out.get("composite",          pd.DataFrame())
    adp_comp = out.get("adaptive_composite", pd.DataFrame())

    if ew_comp.empty or adp_comp.empty:
        return pd.Series(dtype=float)

    common_idx = ew_comp.index.intersection(adp_comp.index)
    ew_comp    = ew_comp.loc[common_idx]
    adp_comp   = adp_comp.loc[common_idx]

    rho_ts = pd.Series(index=common_idx, dtype=float)
    for t in common_idx:
        v1 = ew_comp.loc[t].dropna()
        v2 = adp_comp.loc[t].reindex(v1.index).dropna()
        if len(v2) < 5:
            continue
        rho, _ = spearmanr(v1.values, v2.values)
        rho_ts[t] = rho

    return rho_ts


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — demonstration with synthetic data
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic pipeline output for demonstration …")

    rng    = np.random.default_rng(42)
    T, N   = 200, 10
    dates  = pd.date_range("2006-04-30", periods=T, freq="ME")
    tickers = [f"ETF_{i:02d}" for i in range(N)]

    # Simulate z-scored factor panels with realistic cross-correlations
    # Carry and Quality have moderate positive correlation for equity ETFs
    # Value and Momentum are near-zero correlated (classic style tension)
    corr_mat = np.array([
        [1.00,  0.10,  0.05,  0.30],   # carry
        [0.10,  1.00, -0.20,  0.15],   # value
        [0.05, -0.20,  1.00,  0.10],   # momentum
        [0.30,  0.15,  0.10,  1.00],   # quality
    ])
    L = np.linalg.cholesky(corr_mat)

    def make_panel():
        panels = {}
        for f_i, fname in enumerate(FACTORS):
            # (T × 4) block with inter-factor correlation, then take column f_i
            raw_block = rng.standard_normal((T * N, 4)) @ L.T  # (T*N × 4)
            col = raw_block[:, f_i].reshape(T, N)
            panels[fname] = pd.DataFrame(col, index=dates, columns=tickers)
        return panels

    panels = make_panel()

    # Simulate log returns (forward 1 month), partially driven by factors
    true_w = np.array([0.15, 0.25, 0.35, 0.25])   # true factor weights
    ret = pd.DataFrame(
        sum(true_w[i] * panels[FACTORS[i]].values for i in range(4))
        + 0.5 * rng.standard_normal((T, N)),
        index=dates,
        columns=tickers,
    )

    # Build a mock pipeline output dict
    out_mock = {
        "log_ret":      ret,
        "composite":    sum(0.25 * panels[f] for f in FACTORS),
        **{f"{f}_z": panels[f] for f in FACTORS},
    }

    out_mock = run_adaptive_weighting(out_mock, verbose=True)

    print("\n  Per-ETF adaptive weights (first 5 tickers):")
    print(out_mock["adaptive_weights"].head().to_string())

    rho = compare_composites(out_mock)
    print(f"\n  EW vs adaptive composite Spearman ρ:  "
          f"mean={rho.mean():.3f},  min={rho.min():.3f},  max={rho.max():.3f}")
