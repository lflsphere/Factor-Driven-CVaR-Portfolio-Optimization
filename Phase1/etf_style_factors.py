"""
=============================================================================
US ETF Universe — Style Factor Computation
Carry | Value | Momentum | Quality
=============================================================================
Quantitative Research | Investment Management
April 2026

Universe
--------
33 US-listed ETFs with ≥ 20 years of continuous history (inception ≤ Apr 2006),
spanning multiple market regimes:
  • Dot-com bust (2000–2002)
  • Pre-GFC expansion (2003–2007)
  • Global Financial Crisis (2007–2009)
  • Post-GFC recovery (2010–2019)
  • COVID crash & recovery (2020)
  • Rate-hike cycle / inflation (2021–2023)
  • Post-hiking normalisation (2024–2026)

Benchmark : SPY — SPDR S&P 500 ETF Trust (inception Jan 1993)

─────────────────────────────────────────────────────────────────────────────
Factor Framework  (Koijen et al. 2018; Asness, Moskowitz & Pedersen 2013;
                   Novy-Marx 2013; Fama & French 1993, 2015)
─────────────────────────────────────────────────────────────────────────────

  CARRY    Koijen, Moskowitz, Pedersen & Vrugt (JFE 2018): expected return
           assuming price does not change.
           Signal = Trailing 12M Dividend Yield − Risk-Free Rate (GS1/12)
           Fallback for commodity ETFs (GLD, IAU, USO): NaN
                    (no spot yield; production use requires futures term structure)

  VALUE    Fama & French (1992); Asness, Moskowitz & Pedersen (JF 2013)
           Equity/REIT : negative 36M log return (long-run reversal as value proxy)
           Fixed income: trailing 12M div-yield deviation from 60M rolling mean
           Commodity   : negative 36M log return (mean reversion in real price)
           Higher score = "cheaper" relative to cross-section

  MOMENTUM Jegadeesh & Titman (JF 1993); Moskowitz, Ooi & Pedersen (JF 2012)
           Cross-Sectional: 12-1 month log return (skip-1-month convention)
           Time-Series   : sign of 12-1 month return (±1 signal)
           Both computed for all asset classes

  QUALITY  Novy-Marx (JFE 2013); Asness, Frazzini & Pedersen (JF 2019)
           Dynamic proxy: 36-month rolling annualised Sharpe ratio
           High Sharpe = consistent excess return per unit risk = "quality"
           For equity ETFs with rich data, blend with static ROE/ROA signals

─────────────────────────────────────────────────────────────────────────────
Computation Convention
─────────────────────────────────────────────────────────────────────────────
  • Monthly log returns:    r_{i,t} = ln(P_{i,t} / P_{i,t-1})
  • All factors cross-sectionally z-scored (μ=0, σ=1) with ±3σ winsorisation
  • Common factor start:    2006-04-01 (all 33 ETFs have data)
  • Prices: Yahoo Finance   auto_adjust=False → 'Adj Close' column
  • Dividends: yfinance     Ticker.dividends (raw cash per share)
  • Risk-free rate: FRED    GS1 (1-Year CMT) / 100 / 12 → monthly decimal

─────────────────────────────────────────────────────────────────────────────
Dependencies
─────────────────────────────────────────────────────────────────────────────
    pip install yfinance pandas numpy requests openpyxl scipy
=============================================================================
"""

import warnings
import io
import requests
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

START_DATE   = "2004-01-01"   # warm-up buffer for rolling windows
END_DATE     = "2026-03-31"   # last date for factor computation
FACTOR_START = "2006-04-01"   # first date all 33 ETFs have continuous history
BENCHMARK    = "SPY"

# ── ETF Universe: ≥ 20 years of continuous data (inception ≤ Apr-2006) ────────
# Verified inception dates via yfinance (see universe_inception_dates below)
UNIVERSE = {
    # ── Broad Market Equity ─────────────────────────────────────────────────
    "SPY":  "S&P 500 — SPDR (BENCHMARK, inception 1993-01-29)",
    "IVV":  "S&P 500 — iShares Core (inception 2000-05-19)",
    "VTI":  "Total US Market — Vanguard (inception 2001-06-15)",
    "ITOT": "Total US Market — iShares Core (inception 2004-01-23)",
    # ── Equity Style ────────────────────────────────────────────────────────
    "IWB":  "Russell 1000 — iShares Large Blend (2000-05-19)",
    "IWD":  "Russell 1000 Value — iShares (2000-05-26)",
    "IWF":  "Russell 1000 Growth — iShares (2000-05-26)",
    "IWM":  "Russell 2000 — iShares Small Blend (2000-05-26)",
    "IWN":  "Russell 2000 Value — iShares (2000-07-28)",
    "IWO":  "Russell 2000 Growth — iShares (2000-07-28)",
    "IJH":  "S&P 400 Mid-Cap — iShares Core (2000-05-26)",
    "IJJ":  "S&P 400 Mid-Cap Value — iShares (2000-07-28)",
    # ── GICS Sectors (all 9 available with 20yr history) ────────────────────
    "XLK":  "Information Technology — SPDR (1998-12-22)",
    "XLF":  "Financials — SPDR (1998-12-22)",
    "XLV":  "Health Care — SPDR (1998-12-22)",
    "XLE":  "Energy — SPDR (1998-12-22)",
    "XLI":  "Industrials — SPDR (1998-12-22)",
    "XLY":  "Consumer Discretionary — SPDR (1998-12-22)",
    "XLP":  "Consumer Staples — SPDR (1998-12-22)",
    "XLU":  "Utilities — SPDR (1998-12-22)",
    "XLB":  "Materials — SPDR (1998-12-22)",
    # ── Fixed Income ────────────────────────────────────────────────────────
    "SHY":  "US Treasury 1–3Y — iShares (2002-07-30)",
    "IEF":  "US Treasury 7–10Y — iShares (2002-07-30)",
    "TLT":  "US Treasury 20+Y — iShares (2002-07-30)",
    "LQD":  "IG Corporate — iShares iBoxx $ (2002-07-30)",
    "AGG":  "US Aggregate Bond — iShares Core (2003-09-29)",
    "TIP":  "TIPS — iShares (2003-12-05)",
    # ── Real Estate ─────────────────────────────────────────────────────────
    # "IYR":  "US Real Estate — iShares DJ (2000-06-19)",
    # "VNQ":  "US REITs — Vanguard (2004-09-29)",
    # ── Real Assets / Commodities ────────────────────────────────────────────
    # "GLD":  "Gold — SPDR Gold Shares (2004-11-18)",
    # "IAU":  "Gold — iShares Gold Trust (2005-01-28)",
    # "USO":  "Crude Oil — US Oil Fund (2006-04-10)",
    # ── Smart Beta (with 20yr history) ──────────────────────────────────────
    # "SPHQ": "S&P 500 Quality — Invesco (2005-12-06)",
}

ALL_TICKERS = list(UNIVERSE.keys())  # 33 ETFs

# Asset class mapping → drives factor fallback logic
ASSET_CLASS = {
    "SPY":"equity",  "IVV":"equity",  "VTI":"equity",  "ITOT":"equity",
    "IWB":"equity",  "IWD":"equity",  "IWF":"equity",  "IWM":"equity",
    "IWN":"equity",  "IWO":"equity",  "IJH":"equity",  "IJJ":"equity",
    "XLK":"equity",  "XLF":"equity",  "XLV":"equity",  "XLE":"equity",
    "XLI":"equity",  "XLY":"equity",  "XLP":"equity",  "XLU":"equity",
    "XLB":"equity",  "SPHQ":"equity",
    "SHY":"fixed_income", "IEF":"fixed_income", "TLT":"fixed_income",
    "LQD":"fixed_income", "AGG":"fixed_income", "TIP":"fixed_income",
    "IYR":"real_estate",  "VNQ":"real_estate",
    "GLD":"commodity",    "IAU":"commodity",    "USO":"commodity",
}

# Tickers that pay dividends (needed for Carry)
NON_DIVIDEND_TICKERS = {"GLD", "IAU", "USO"}  # commodity ETFs: no yield

print(f"Universe      : {len(ALL_TICKERS)} ETFs")
print(f"Benchmark     : {BENCHMARK}")
print(f"Factor window : {FACTOR_START}  →  {END_DATE}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def download_prices_and_dividends(tickers, start=START_DATE, end=END_DATE):
    """
    Download adjusted close prices and dividend cash flows.

    Uses auto_adjust=False so we get both:
      - 'Adj Close': split-and-dividend adjusted price (for log returns)
      - Raw dividends via Ticker.dividends (needed for yield-based Carry)

    Parameters
    ----------
    tickers : list of str
    start   : str   ISO date
    end     : str   ISO date

    Returns
    -------
    adj_close_monthly : pd.DataFrame  (T_months × N) month-end adj close
    div_monthly       : pd.DataFrame  (T_months × N) monthly dividend cash
    log_ret           : pd.DataFrame  (T_months × N) monthly log returns
    """
    print(f"[yfinance] Downloading prices for {len(tickers)} tickers …")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=False, progress=True)
    adj_close_daily   = raw["Adj Close"].ffill(limit=5)
    adj_close_monthly = adj_close_daily.resample("ME").last()
    log_ret = np.log(adj_close_monthly / adj_close_monthly.shift(1))

    print("[yfinance] Downloading dividend histories …")
    # Build monthly dividend DataFrame (raw cash per share)
    div_monthly = pd.DataFrame(
        0.0, index=adj_close_monthly.index, columns=tickers
    )
    for t in tickers:
        try:
            d = yf.Ticker(t).dividends
            d.index = d.index.tz_localize(None)
            dm = d.resample("ME").sum()
            div_monthly[t] = dm.reindex(adj_close_monthly.index, fill_value=0)
        except Exception as e:
            print(f"  [dividends] {t}: {e}")

    print(f"  adj_close_monthly : {adj_close_monthly.shape}")
    print(f"  div_monthly       : {div_monthly.shape}")
    print(f"  log_ret           : {log_ret.dropna(how='all').shape}")
    return adj_close_monthly, div_monthly, log_ret


def download_risk_free_rate(start=START_DATE, end=END_DATE):
    """
    Download 1-Year Treasury CMT (GS1) from FRED as monthly risk-free rate.
    Returns annualised % / 100 / 12 → monthly decimal.

    Fallback: FEDFUNDS if GS1 unavailable.
    """
    print("[FRED] Downloading risk-free rate …")
    for sid in ["GS1", "FEDFUNDS"]:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            df  = pd.read_csv(io.StringIO(requests.get(url, timeout=20).text))
            df.columns = ["Date", sid]
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date")
            df[sid] = pd.to_numeric(df[sid], errors="coerce")
            rf = df.loc[start:end, sid].resample("ME").last() / 100 / 12
            rf.name = "RF"
            print(f"  ✓ {sid}  ({rf.notna().sum()} monthly obs)")
            return rf
        except Exception as e:
            print(f"  ✗ {sid}: {e}")
    return pd.Series(dtype=float, name="RF")


def download_fundamentals_snapshot(tickers):
    """
    Retrieve static fundamental data from Yahoo Finance for equity ETFs.
    Returns a DataFrame (N × F) — point-in-time snapshot.

    Fields: ROE, ROA, ProfitMargin, EarningsGrowth, PE, PB, DivYield, Beta
    """
    print("[yfinance] Downloading fundamental snapshots …")
    FIELDS = {
        "trailingPE":     "PE_trailing",
        "forwardPE":      "PE_forward",
        "priceToBook":    "PB",
        "returnOnEquity": "ROE",
        "returnOnAssets": "ROA",
        "profitMargins":  "ProfitMargin",
        "earningsGrowth": "EarningsGrowth",
        "revenueGrowth":  "RevenueGrowth",
        "dividendYield":  "DivYield",
        "trailingAnnualDividendYield": "DivYield_Trailing",
        "beta":           "Beta",
        "totalAssets":    "AUM",
        "yield":          "ETF_Yield",
        "currentPrice":   "Price",
        "bookValue":      "BookValue",
    }
    records = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            records[t] = {lbl: info.get(k) for k, lbl in FIELDS.items()}
            print(f"  ✓ {t}")
        except Exception as e:
            records[t] = {lbl: np.nan for lbl in FIELDS.values()}
            print(f"  ✗ {t}: {e}")
    return pd.DataFrame(records).T.apply(pd.to_numeric, errors="coerce")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FACTOR UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def cross_section_zscore(row, winsor=3.0):
    """
    Cross-sectionally standardise a pd.Series (one date, N ETFs).
    Step 1 — Winsorise at ±winsor × σ
    Step 2 — Z-score: (x − μ) / σ
    """
    s = row.dropna()
    if len(s) < 3:
        return row
    mu, sigma = s.mean(), s.std()
    if sigma == 0:
        return pd.Series(0.0, index=row.index)
    s_win = s.clip(mu - winsor * sigma, mu + winsor * sigma)
    z = (s_win - s_win.mean()) / s_win.std()
    return z.reindex(row.index)   # restore NaNs for missing tickers


def zscore_panel(panel):
    """Apply cross_section_zscore to every row of a DataFrame."""
    return panel.apply(cross_section_zscore, axis=1)


def _init_panel(idx, tickers):
    """Initialise a (T × N) float DataFrame filled with NaN."""
    return pd.DataFrame(np.nan, index=idx, columns=tickers, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FACTOR 1 — CARRY
# ─────────────────────────────────────────────────────────────────────────────
#
#  Definition (Koijen, Moskowitz, Pedersen & Vrugt, 2018):
#    Carry_i,t = E[r_{i,t} | ΔP = 0]
#              = income yield during holding period
#
#  For equity / fixed-income / REIT ETFs:
#    Carry_i,t = (Σ_{k=1}^{12} Div_{i,t-k}) / P_{i,t}  −  RF_t
#
#    where:
#      Div_{i,t-k} = dividend per share paid in month t-k
#      P_{i,t}     = adjusted close price at month-end t
#      RF_t        = 1-Year CMT / 100 / 12  (monthly risk-free rate)
#
#  For commodity ETFs (GLD, IAU, USO):
#    No spot yield observable without futures term structure.
#    Signal = NaN  (excluded from cross-sectional ranking for Carry).
#    In production: use CME futures curve data (roll yield = F1/F2 − 1).
# ─────────────────────────────────────────────────────────────────────────────

def compute_carry(adj_close_monthly, div_monthly, rf_monthly,
                  tickers=ALL_TICKERS, factor_start=FACTOR_START):
    """
    Compute the Carry factor signal for each ETF at each month-end.

    Parameters
    ----------
    adj_close_monthly : pd.DataFrame  (T × N) month-end adjusted close prices
    div_monthly       : pd.DataFrame  (T × N) monthly dividend cash per share
    rf_monthly        : pd.Series     (T,) monthly risk-free rate (decimal)
    tickers           : list
    factor_start      : str

    Returns
    -------
    carry_raw    : pd.DataFrame  (T_factor × N)  yield - RF
    carry_zscore : pd.DataFrame  (T_factor × N)  cross-sectionally z-scored
    """
    print("\n[CARRY] Computing trailing 12M dividend yield − RF …")
    panel = _init_panel(adj_close_monthly.index, tickers)

    for t in tickers:
        if ASSET_CLASS.get(t) == "commodity":
            continue                    # no yield for commodity ETFs
        price = adj_close_monthly[t]
        # Trailing 12-month cumulative dividends
        div_12m = div_monthly[t].rolling(12).sum()
        # Dividend yield
        dy = div_12m / price.replace(0, np.nan)
        # Align RF
        rf_a = rf_monthly.reindex(price.index, method="ffill")
        panel[t] = dy - rf_a

    carry_raw = panel.loc[factor_start:]
    carry_z   = zscore_panel(carry_raw)
    print(f"  carry_raw    : {carry_raw.shape}   NaN%: "
          f"{carry_raw.isna().mean().mean():.1%}")
    return carry_raw, carry_z


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FACTOR 2 — VALUE
# ─────────────────────────────────────────────────────────────────────────────
#
#  Definition (Asness, Moskowitz & Pedersen 2013 — cross-asset value):
#    Value = relative cheapness of an asset vs. its own intrinsic value.
#
#  For equity & REIT ETFs:
#    Value_i,t = −log(P_{i,t} / P_{i,t−36})
#    = negative 36-month log return (long-run mean reversion / reversal).
#    Rationale: Asness et al. (2013) apply 5-year return reversal as value
#    signal in cross-asset context. We use 36M (3Y) for ETFs.
#    Higher score → ETF has underperformed → "cheaper" on a relative basis.
#
#  For fixed income ETFs:
#    Value_i,t = (DY_{i,t} − DY_{i,60M_avg}) / DY_{i,60M_std}
#    Normalised deviation of current yield from its own 5-year average.
#    Higher score → yield is above its long-run level → "cheap" bond ETF.
#
#  For commodity ETFs:
#    Value_i,t = −log(P_{i,t} / P_{i,t−36})
#    Same long-run reversal as equities.
# ─────────────────────────────────────────────────────────────────────────────

def compute_value(adj_close_monthly, div_monthly, tickers=ALL_TICKERS,
                  factor_start=FACTOR_START, lookback_equity=36,
                  lookback_fi_z=60):
    """
    Compute the dynamic Value factor panel.

    Parameters
    ----------
    adj_close_monthly : pd.DataFrame
    div_monthly       : pd.DataFrame
    tickers           : list
    factor_start      : str
    lookback_equity   : int   months for long-run reversal (equity/commodity)
    lookback_fi_z     : int   months for FI yield z-score normalisation

    Returns
    -------
    value_raw    : pd.DataFrame  (T_factor × N)
    value_zscore : pd.DataFrame  (T_factor × N)
    """
    print("\n[VALUE] Computing long-run reversal / yield deviation …")
    panel = _init_panel(adj_close_monthly.index, tickers)

    for t in tickers:
        price = adj_close_monthly[t]
        ac    = ASSET_CLASS.get(t, "equity")

        if ac in ("equity", "real_estate", "commodity"):
            # Negative 36M log return: cheap = fallen more than cross-section
            panel[t] = -np.log(price / price.shift(lookback_equity))

        elif ac == "fixed_income":
            # Yield deviation from own history
            dy     = div_monthly[t].rolling(12).sum() / price.replace(0, np.nan)
            dy_avg = dy.rolling(lookback_fi_z).mean()
            dy_std = dy.rolling(lookback_fi_z).std().replace(0, np.nan)
            panel[t] = (dy - dy_avg) / dy_std

    value_raw = panel.loc[factor_start:]
    value_z   = zscore_panel(value_raw)
    print(f"  value_raw    : {value_raw.shape}   NaN%: "
          f"{value_raw.isna().mean().mean():.1%}")
    return value_raw, value_z


def compute_value_static(fundamentals_df, tickers=ALL_TICKERS):
    """
    Static Value signal from Yahoo Finance fundamentals (point-in-time).

    For equity ETFs:
      Composite = mean(EarningsYield, BookYield) = mean(1/PE, 1/PB)
    For FI ETFs:
      ETF_Yield (SEC 30-day yield)
    For commodity ETFs:
      NaN (no fundamental valuation metric)

    Returns
    -------
    value_static_raw   : pd.Series  (N,)
    value_static_zscore: pd.Series  (N,)
    """
    raw = pd.Series(np.nan, index=tickers)
    for t in tickers:
        if t not in fundamentals_df.index:
            continue
        row = fundamentals_df.loc[t]
        ac  = ASSET_CLASS.get(t, "equity")

        if ac in ("equity", "real_estate"):
            pe, pb = row.get("PE_trailing"), row.get("PB")
            signals = []
            if pd.notna(pe) and pe > 0:  signals.append(1 / pe)
            if pd.notna(pb) and pb > 0:  signals.append(1 / pb)
            if signals: raw[t] = np.mean(signals)

        elif ac == "fixed_income":
            y = row.get("ETF_Yield") or row.get("DivYield_Trailing")
            if pd.notna(y): raw[t] = y

    return raw, cross_section_zscore(raw)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FACTOR 3 — MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────
#
#  Cross-Sectional Momentum (Jegadeesh & Titman 1993):
#    MOM_{i,t} = log(P_{i,t−1} / P_{i,t−13})
#              = sum of monthly log returns from t−12 to t−1
#    Skip most recent month to avoid short-term reversal.
#    Rank within cross-section: buy winners, sell losers.
#
#  Time-Series Momentum (Moskowitz, Ooi & Pedersen 2012):
#    TS_MOM_{i,t} = sign(MOM_{i,t})
#    Long (+1) if past return positive; Short (−1) if negative.
#    Each asset timed independently of others.
#
#  Applied uniformly to all asset classes (equities, bonds, commodities).
# ─────────────────────────────────────────────────────────────────────────────

def compute_momentum(adj_close_monthly, tickers=ALL_TICKERS,
                     factor_start=FACTOR_START,
                     lookback=12, skip=1):
    """
    Compute cross-sectional and time-series momentum.

    MOM_{i,t} = log(P_{i,t−skip} / P_{i,t−skip−lookback})

    Parameters
    ----------
    adj_close_monthly : pd.DataFrame
    tickers           : list
    factor_start      : str
    lookback          : int   formation period in months (default 12)
    skip              : int   skip months to avoid reversal (default 1)

    Returns
    -------
    mom_raw         : pd.DataFrame  (T × N)  12-1 log return
    mom_cs_zscore   : pd.DataFrame  (T × N)  cross-sectional z-score
    mom_ts_signal   : pd.DataFrame  (T × N)  ±1 time-series signal
    """
    print(f"\n[MOMENTUM] Computing {lookback}-{skip} month log return …")
    panel = _init_panel(adj_close_monthly.index, tickers)

    for t in tickers:
        price = adj_close_monthly[t]
        # t-1 price vs t-13 price (skip 1 month)
        panel[t] = np.log(price.shift(skip) / price.shift(skip + lookback))

    mom_raw       = panel.loc[factor_start:]
    mom_cs_zscore = zscore_panel(mom_raw)
    mom_ts_signal = mom_raw.apply(np.sign)   # ±1 for time-series momentum

    print(f"  mom_raw      : {mom_raw.shape}   NaN%: "
          f"{mom_raw.isna().mean().mean():.1%}")
    return mom_raw, mom_cs_zscore, mom_ts_signal


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FACTOR 4 — QUALITY
# ─────────────────────────────────────────────────────────────────────────────
#
#  Definition (Novy-Marx 2013; Asness, Frazzini & Pedersen 2019):
#    Quality = profitability + safety + payout + growth
#    High quality firms are more profitable, safer, growing faster.
#
#  Dynamic proxy (all asset classes):
#    Quality_i,t = Sharpe ratio over trailing 36 months
#              = (μ_{36M} / σ_{36M}) × √12   (annualised)
#    Rationale: a high Sharpe ratio reflects consistent excess return relative
#    to volatility — analogous to high operating profitability in individual
#    stocks (stable earnings / low earnings volatility).
#
#  For equity ETFs with rich fundamental data (optional static blend):
#    Quality_static = mean(z(ROE), z(ROA), z(ProfitMargin), z(EarningsGrowth))
#    Can be blended 50/50 with the rolling Sharpe signal.
# ─────────────────────────────────────────────────────────────────────────────

def compute_quality_dynamic(log_ret, tickers=ALL_TICKERS,
                             factor_start=FACTOR_START,
                             window=36, ann_factor=12):
    """
    Compute rolling annualised Sharpe ratio as the Quality proxy.

    Quality_i,t = (mean(r_{t-window:t}) / std(r_{t-window:t})) × √ann_factor

    Parameters
    ----------
    log_ret    : pd.DataFrame  monthly log returns
    tickers    : list
    factor_start: str
    window     : int           rolling window in months (default 36)
    ann_factor : int           annualisation multiplier (12 for monthly)

    Returns
    -------
    quality_raw    : pd.DataFrame  (T_factor × N)  rolling Sharpe
    quality_zscore : pd.DataFrame  (T_factor × N)  cross-sectional z-score
    """
    print(f"\n[QUALITY] Computing rolling {window}M Sharpe ratio …")
    panel = _init_panel(log_ret.index, tickers)

    for t in tickers:
        if t not in log_ret.columns:
            continue
        r = log_ret[t]
        mu_r  = r.rolling(window).mean()
        std_r = r.rolling(window).std().replace(0, np.nan)
        panel[t] = (mu_r / std_r) * np.sqrt(ann_factor)

    quality_raw    = panel.loc[factor_start:]
    quality_zscore = zscore_panel(quality_raw)
    print(f"  quality_raw  : {quality_raw.shape}   NaN%: "
          f"{quality_raw.isna().mean().mean():.1%}")
    return quality_raw, quality_zscore


def compute_quality_static(fundamentals_df, tickers=ALL_TICKERS):
    """
    Static Quality signal from Yahoo Finance fundamentals.

    Composite = equal-weight z-score of ROE, ROA, ProfitMargin, EarningsGrowth.
    Only meaningful for equity / REIT ETFs; NaN for FI and commodities.

    Returns
    -------
    quality_raw    : pd.Series  (N,)  composite score (unscaled)
    quality_zscore : pd.Series  (N,)  cross-sectionally z-scored
    """
    raw = pd.Series(np.nan, index=tickers)
    for t in tickers:
        if t not in fundamentals_df.index:
            continue
        ac  = ASSET_CLASS.get(t, "equity")
        row = fundamentals_df.loc[t]
        if ac in ("equity", "real_estate"):
            sigs = [row.get(f) for f in
                    ["ROE", "ROA", "ProfitMargin", "EarningsGrowth"]]
            sigs = [v for v in sigs if pd.notna(v)]
            if sigs: raw[t] = np.mean(sigs)
    return raw, cross_section_zscore(raw)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_composite(carry_z, value_z, momentum_z, quality_z,
                       weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Equal-weighted (or custom) composite style factor score.

    Score_i,t = w_C * Carry_z + w_V * Value_z + w_M * MOM_z + w_Q * Quality_z

    Missing factor values reduce the normalising denominator (no zero-fill bias).

    Parameters
    ----------
    carry_z, value_z, momentum_z, quality_z : pd.DataFrame  z-scored panels
    weights : tuple   (w_Carry, w_Value, w_Mom, w_Quality), must sum to 1

    Returns
    -------
    composite : pd.DataFrame  (T × N)
    """
    w = np.array(weights)
    assert abs(w.sum() - 1.0) < 1e-9, "Weights must sum to 1"

    # Align on common index and columns
    tickers = sorted(set(carry_z.columns) & set(value_z.columns)
                     & set(momentum_z.columns) & set(quality_z.columns))
    idx = (carry_z.index.intersection(value_z.index)
                        .intersection(momentum_z.index)
                        .intersection(quality_z.index))

    panels = [
        carry_z.loc[idx, tickers],
        value_z.loc[idx, tickers],
        momentum_z.loc[idx, tickers],
        quality_z.loc[idx, tickers],
    ]
    # Count valid contributors per cell to avoid NaN-dilution
    n_valid  = sum(p.notna().astype(float) * wi for p, wi in zip(panels, w))
    n_valid  = n_valid.replace(0, np.nan)
    weighted = sum(p.fillna(0) * wi for p, wi in zip(panels, w))
    composite = weighted / n_valid
    composite.columns = tickers
    return composite


# ─────────────────────────────────────────────────────────────────────────────
# 8.  BENCHMARK ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_benchmark_attribution(log_ret, carry_z, value_z, mom_z, quality_z,
                                   benchmark=BENCHMARK, window=60):
    """
    Rolling OLS: regress each ETF's excess return (vs SPY) on the four
    style factor z-scores.

    r_{i,t}^{excess} = α_i + β_{C,i}·Carry_t + β_{V,i}·Value_t
                      + β_{M,i}·MOM_t + β_{Q,i}·Quality_t + ε_{i,t}

    HAC-robust covariance: implemented as plain OLS with Newey-West correction
    available via statsmodels if installed (fallback: plain OLS).

    Parameters
    ----------
    log_ret   : pd.DataFrame  monthly log returns
    carry_z … quality_z : pd.DataFrame  z-scored factor panels
    benchmark : str           benchmark ticker
    window    : int           rolling OLS window (months)

    Returns
    -------
    rolling_loadings : dict  {ticker: pd.DataFrame(Date × params)}
    latest_loadings  : pd.DataFrame  (N_etfs × params) most recent
    """
    print(f"\n[ATTRIBUTION] Rolling OLS (window={window}M) vs. {benchmark} …")
    FACTOR_NAMES = ["Carry", "Value", "Momentum", "Quality"]

    # Common time index across all panels
    common_idx = (log_ret.index
                  .intersection(carry_z.index)
                  .intersection(value_z.index)
                  .intersection(mom_z.index)
                  .intersection(quality_z.index))

    # Build (T × 4) factor matrix (use fill=0 for sporadic NaN between dates)
    F = np.column_stack([
        carry_z.reindex(common_idx).values.astype(float),
        value_z.reindex(common_idx).values.astype(float),
        mom_z.reindex(common_idx).values.astype(float),
        quality_z.reindex(common_idx).values.astype(float),
    ])
    # F is (T × 4*N) — extract per-ETF columns
    factor_cols = {
        "Carry":    carry_z.reindex(common_idx),
        "Value":    value_z.reindex(common_idx),
        "Momentum": mom_z.reindex(common_idx),
        "Quality":  quality_z.reindex(common_idx),
    }

    log_ret_a = log_ret.reindex(common_idx)
    bmk_ret   = log_ret_a[benchmark].fillna(0) \
                if benchmark in log_ret_a.columns else \
                pd.Series(0.0, index=common_idx)

    rolling_loadings = {}
    latest = {}

    non_bmk = [t for t in ALL_TICKERS if t != benchmark
               and t in log_ret_a.columns]

    for t in non_bmk:
        excess = (log_ret_a[t] - bmk_ret).values.astype(float)

        # (T × 4) factor values for this ticker (using cross-section z-scores)
        Xf = np.column_stack([
            factor_cols["Carry"][t].values    if t in factor_cols["Carry"].columns    else np.full(len(common_idx), np.nan),
            factor_cols["Value"][t].values    if t in factor_cols["Value"].columns    else np.full(len(common_idx), np.nan),
            factor_cols["Momentum"][t].values if t in factor_cols["Momentum"].columns else np.full(len(common_idx), np.nan),
            factor_cols["Quality"][t].values  if t in factor_cols["Quality"].columns  else np.full(len(common_idx), np.nan),
        ]).astype(float)

        rows = []
        for i in range(window, len(common_idx)):
            xe  = excess[i-window:i]
            xf  = Xf[i-window:i]
            valid = ~(np.isnan(xe) | np.any(np.isnan(xf), axis=1))
            if valid.sum() < window // 2:
                continue
            X_aug = np.column_stack([np.ones(valid.sum()), xf[valid]])
            c, _, _, _ = np.linalg.lstsq(X_aug, xe[valid], rcond=None)
            rows.append(
                {"Date": common_idx[i],
                 "alpha": c[0],
                 **{f"beta_{n}": c[k+1] for k, n in enumerate(FACTOR_NAMES)}}
            )

        if rows:
            df_t = pd.DataFrame(rows).set_index("Date")
            rolling_loadings[t] = df_t
            latest[t] = df_t.iloc[-1]

    latest_df = pd.DataFrame(latest).T
    latest_df.index.name = "Ticker"
    print(f"  Attribution computed for {len(latest_df)} ETFs")
    return rolling_loadings, latest_df


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    """
    Execute the full Carry / Value / Momentum / Quality pipeline.

    Returns a results dict with all raw and z-scored factor panels,
    composite scores, and benchmark factor loadings.
    """
    out = {}

    # ── A. Market data ────────────────────────────────────────────────────
    adj_close_monthly, div_monthly, log_ret = download_prices_and_dividends(
        ALL_TICKERS, START_DATE, END_DATE)
    rf_monthly = download_risk_free_rate(START_DATE, END_DATE)

    out.update({
        "adj_close_monthly": adj_close_monthly,
        "div_monthly":       div_monthly,
        "log_ret":           log_ret,
        "rf_monthly":        rf_monthly,
    })

    # ── B. Fundamental snapshot (static) ──────────────────────────────────
    fundamentals = download_fundamentals_snapshot(ALL_TICKERS)
    out["fundamentals"] = fundamentals

    # ── C. CARRY ─────────────────────────────────────────────────────────
    carry_raw, carry_z = compute_carry(
        adj_close_monthly, div_monthly, rf_monthly)
    out.update({"carry_raw": carry_raw, "carry_z": carry_z})

    # ── D. VALUE ─────────────────────────────────────────────────────────
    value_raw, value_z = compute_value(adj_close_monthly, div_monthly)
    out.update({"value_raw": value_raw, "value_z": value_z})

    # ── E. MOMENTUM ───────────────────────────────────────────────────────
    mom_raw, mom_z, mom_ts = compute_momentum(adj_close_monthly)
    out.update({"mom_raw": mom_raw, "mom_z": mom_z, "mom_ts_signal": mom_ts})

    # ── F. QUALITY ────────────────────────────────────────────────────────
    quality_raw, quality_z = compute_quality_dynamic(log_ret, window=36)
    out.update({"quality_raw": quality_raw, "quality_z": quality_z})

    # ── G. COMPOSITE ──────────────────────────────────────────────────────
    composite = compute_composite(carry_z, value_z, mom_z, quality_z)
    out["composite"] = composite

    # ── H. BENCHMARK ATTRIBUTION ──────────────────────────────────────────
    rolling_betas, latest_loadings = compute_benchmark_attribution(
        log_ret, carry_z, value_z, mom_z, quality_z,
        benchmark=BENCHMARK, window=60)
    out["rolling_betas"]   = rolling_betas
    out["latest_loadings"] = latest_loadings

    # ── I. Pipeline summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STYLE FACTOR PIPELINE COMPLETE")
    print(f"  Universe       : {len(ALL_TICKERS)} ETFs  |  Benchmark: {BENCHMARK}")
    print(f"  log_ret        : {log_ret.dropna(how='all').shape}")
    print(f"  carry_z        : {carry_z.shape}")
    print(f"  value_z        : {value_z.shape}")
    print(f"  mom_z          : {mom_z.shape}")
    print(f"  quality_z      : {quality_z.shape}")
    print(f"  composite      : {composite.shape}")
    print(f"  latest_loadings: {latest_loadings.shape}")
    print("=" * 70)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 10. REPORTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def factor_summary(panel, factor_name):
    """
    Time-series and cross-sectional descriptive statistics for a factor panel.
    """
    desc = pd.DataFrame({
        "Mean":     panel.mean(),
        "Std":      panel.std(),
        "Skew":     panel.skew(),
        "Kurt":     panel.kurt(),
        "Min":      panel.min(),
        "Max":      panel.max(),
        "Coverage%": panel.notna().mean() * 100,
    }).round(3)
    desc.index.name = "Ticker"
    print(f"\n── {factor_name} ── Summary (time-series moments per ETF)")
    print(desc.to_string())
    return desc


def latest_ranking(panel, factor_name, n=5):
    """Print the top-N and bottom-N ETFs by most recent factor z-score."""
    last = panel.dropna(how="all").iloc[-1].dropna().sort_values(ascending=False)
    print(f"\n── {factor_name} ── Latest Cross-Section Ranking")
    print(f"  Long  (top {n})  : " + ", ".join(
          [f"{t}={v:+.2f}" for t, v in last.head(n).items()]))
    print(f"  Short (bot {n})  : " + ", ".join(
          [f"{t}={v:+.2f}" for t, v in last.tail(n).items()]))
    return last


def factor_correlation_matrix(out):
    """
    Compute the correlation matrix of the four factor z-scores
    (averaged across ETFs at each date, then time-series correlation).
    """
    mean_signals = pd.DataFrame({
        "Carry":    out["carry_z"].mean(axis=1),
        "Value":    out["value_z"].mean(axis=1),
        "Momentum": out["mom_z"].mean(axis=1),
        "Quality":  out["quality_z"].mean(axis=1),
    }).dropna()
    corr = mean_signals.corr()
    print("\n── Factor Correlation Matrix (cross-ETF mean z-scores)")
    print(corr.round(3).to_string())
    return corr


def export_excel(out, path="etf_style_factors.xlsx"):
    """Export all factor panels to an Excel workbook."""
    print(f"\n[Export] Writing {path} …")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        out["log_ret"].to_excel(writer, sheet_name="Log_Returns")
        out["rf_monthly"].to_frame("RF").to_excel(writer, sheet_name="RiskFree_RF")
        out["carry_raw"].to_excel(writer, sheet_name="Carry_Raw")
        out["carry_z"].to_excel(writer, sheet_name="Carry_ZScore")
        out["value_raw"].to_excel(writer, sheet_name="Value_Raw")
        out["value_z"].to_excel(writer, sheet_name="Value_ZScore")
        out["mom_raw"].to_excel(writer, sheet_name="MOM_Raw_12m1m")
        out["mom_z"].to_excel(writer, sheet_name="MOM_ZScore")
        out["mom_ts_signal"].to_excel(writer, sheet_name="MOM_TS_Signal")
        out["quality_raw"].to_excel(writer, sheet_name="Quality_Sharpe_36M")
        out["quality_z"].to_excel(writer, sheet_name="Quality_ZScore")
        out["composite"].to_excel(writer, sheet_name="Composite_Score")
        out["latest_loadings"].to_excel(writer, sheet_name="Factor_Loadings")
        out["fundamentals"].to_excel(writer, sheet_name="Fundamentals")
    print(f"  ✓ Saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # A. Run the full pipeline
    out = run_pipeline()

    # B. Reporting
    for fname, key in [("CARRY",    "carry_z"),
                       ("VALUE",    "value_z"),
                       ("MOMENTUM", "mom_z"),
                       ("QUALITY",  "quality_z")]:
        factor_summary(out[key], fname)
        latest_ranking(out[key], fname)

    # C. Factor correlation structure
    factor_correlation_matrix(out)

    # D. Composite score rankings
    print("\n── COMPOSITE SCORE — Latest Ranking ──")
    comp_last = out["composite"].dropna(how="all").iloc[-1].sort_values(ascending=False)
    print("  Top 5 (long)   : " + ", ".join(
          [f"{t}={v:+.2f}" for t, v in comp_last.head(5).items()]))
    print("  Bot 5 (short)  : " + ", ".join(
          [f"{t}={v:+.2f}" for t, v in comp_last.tail(5).items()]))

    # E. Benchmark attribution
    print("\n── FACTOR LOADINGS vs. SPY — Latest ──")
    print(out["latest_loadings"].round(4).to_string())

    # F. Export to Excel
    export_excel(out)

    print("\nDone.")
