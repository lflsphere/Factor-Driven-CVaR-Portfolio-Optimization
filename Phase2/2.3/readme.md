# Phase 2.3 - Volatility Scaling & Forward-Looking CVaR Simulation

## 1. Overview
The primary objective of this module is to mitigate the **procyclicality bias** inherent in traditional Historical Simulation (HS). Raw historical data often fails to reflect current market dynamics, leading to lagged risk estimates. 

This implementation adopts a **Volatility Scaling** methodology. By transforming historical returns into "regime-neutral" signals and re-projecting them onto the current volatility environment, the model ensures that tail risk estimates are both historically grounded and forward-looking.

---

## 2. Implementation Logic

### I. Data Normalization (De-volatilization)
To remove historical noise and non-stationarity, each historical log return $r_{i,t}$ is normalized using a long-horizon rolling window:

$$z_{i,t} = \frac{r_{i,t}}{\sigma_{i,t}^{long}}$$

* **$T_{long}$**: 36-period rolling window (capturing baseline historical volatility).
* **Metric**: Stationary Z-scores that preserve the original correlation structure and tail distribution shape.

### II. Adaptive Rescaling (Regime-Adjustment)
The normalized returns are rescaled using the most recent short-horizon volatility to align with the current market regime:

$$\hat{r}_{i,t} = z_{i,t} \times \sigma_{i,current}^{short}$$

* **$T_{short}$**: 12-period rolling window (reflecting current risk intensity).
* **Result**: A scenario matrix where historical shocks are scaled to match today's risk environment.

---

## 3. Risk Metrics & Deliverables

### Multi-level CVaR Forecast
The module computes non-parametric **Conditional Value at Risk (CVaR)** at multiple confidence levels ($\beta = 95\%, 97.5\%, 99\%$):

$$CVaR_{\beta} = E[R \mid R \le VaR_{\alpha}]$$

### Outputs
* **`scenario_returns_matrix.csv`**: $S \times n$ adjusted return matrix. Primary input for Phase 3.1 portfolio optimization.
* **`cvar_forecast_summary.csv`**: Summary table of multi-level CVaR estimates for each asset under the current regime.

---

## 4. Validation & Analysis

The methodology is validated through two distinct analytical plots:

1.  **`volatility_correction_plot.png`**: 
    Displays the impact of normalization. The adjusted series (Blue) maintains a consistent amplitude compared to raw log returns (Gray), confirming the successful removal of procyclicality.

2.  **`cvar_validation_comparison.png`**: 
    Contrasts **Procyclical CVaR** (Raw) with **Normalized CVaR** (Phase 2.3). The results demonstrate that the scaled CVaR adapts significantly faster to volatility spikes, providing more conservative and responsive tail risk estimates.

---

## 5. Usage
Execute the full simulation pipeline and generate all deliverables:
```bash
python Historical_Simulation_with_Volatility.py