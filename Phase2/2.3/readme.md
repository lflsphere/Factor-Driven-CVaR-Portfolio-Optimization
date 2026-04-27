# Phase 2.3 - Volatility Scaling for Historical Simulation

## 1. Overview
The goal here is to fix the "procyclicality" issue in raw historical returns. If we just use old data as-is, the risk estimates end up being too slow to react to current market shifts. This module normalizes historical returns and then rescales them so they actually match the current volatility regime.

## 2. Implementation Details

### Input
- **Price Data**: Adjusted closing prices for the ETF universe (from Phase 1).
- **Windows**: 36-month window for long-term normalization ($T_{long}$) and a 12-month window for current scaling ($T_{short}$).

### Processing (The Logic)
The scaling follows a two-step process:

1. **De-vol**: For each asset $i$, I divided the historical log returns $r_{i,t}$ by the 36-month rolling standard deviation $\sigma_{i,t}^{long}$ to get a stationary set of "z-scores":
   $$z_{i,t} = \frac{r_{i,t}}{\sigma_{i,t}^{long}}$$

2. **Rescale**: I then multiplied these normalized returns by the most recent 12-month volatility $\sigma_{i,current}^{short}$ to project them into the current environment:
   $$\hat{r}_{i,t} = z_{i,t} \times \sigma_{i,current}^{short}$$

By doing this, we keep the original distribution's "tail events" (the big moves) but adjust their scale to fit today's risk environment.

### Output
- **`scenario_returns_matrix.csv`**: The final $S \times n$ matrix where each entry is the adjusted return $\hat{r}_{i,t}$. This goes directly into Elif’s CVaR optimizer in Step 3.1.

## 3. Visualization
I've included `volatility_correction_plot.png` to show the before-and-after. You can see how the adjusted series (the blue line) is much more consistent across different market cycles compared to the raw log returns.