"""
cvar.py — Implementation Plan §2.5: CVaR Calculation Module
============================================================

Three methods for computing Conditional Value-at-Risk 
    (a) Empirical / sample CVaR — sorted average of worst ⌈(1−β)·S⌉ losses.
    (b) Parametric closed-form  — Gaussian and Student-t.
    (c) Rockafellar–Uryasev     — auxiliary 1-D minimization in α (the same
                                   convex form embedded in the Phase-3 LP).

CONVENTION (important — keep this consistent across the whole codebase):
    `losses` are POSITIVE numbers representing portfolio P&L losses.
    Given a return matrix R (S × N) and weights w (N,), portfolio losses are:
        L = -R @ w     (positive numbers in the bad-outcome tail)

    CVaR_β = E[L | L ≥ VaR_β],   VaR_β = inf{ℓ : P(L ≤ ℓ) ≥ β}

    All functions here return CVaR as a positive number (a loss).
"""

from __future__ import annotations

import numpy as np

try:
    from scipy import stats as _stats  # noqa: F401
    _HAS_SCIPY = True
except ImportError:
    _stats = None
    _HAS_SCIPY = False


def _require_scipy(name: str) -> None:
    if not _HAS_SCIPY:
        raise ImportError(
            f"{name} requires scipy. Install with `pip install scipy`."
        )


# ---------------------------------------------------------------------------
# (a) Empirical / sample CVaR
# ---------------------------------------------------------------------------
def cvar_empirical(losses: np.ndarray, beta: float = 0.95) -> float:
    """
    Sorted-sample CVaR: average of the worst ⌈(1−β)·S⌉ losses.

    Parameters
    ----------
    losses : 1-D array of S loss observations (positive = loss).
    beta   : confidence level (e.g. 0.95).

    Returns
    -------
    CVaR as a positive scalar.
    """
    losses = np.asarray(losses, dtype=float).ravel()
    if losses.size == 0:
        raise ValueError("losses is empty")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")

    S = losses.size
    # Tail sample count ⌈(1−β)·S⌉, with a small epsilon to absorb the
    # floating-point error in (1-β)·S when (1-β) is something like 0.05
    # that doesn't round-trip exactly through binary float.
    k = int(np.ceil((1.0 - beta) * S - 1e-9))
    k = max(k, 1)                              # guard against k=0
    worst_k = np.partition(losses, -k)[-k:]    # O(S), no full sort
    return float(worst_k.mean())


def effective_tail_size(S: int, beta: float) -> int:
    """How many observations actually go into the empirical CVaR tail?"""
    return max(int(np.ceil((1.0 - beta) * S - 1e-9)), 1)


# ---------------------------------------------------------------------------
# (b) Parametric closed-form
# ---------------------------------------------------------------------------
def cvar_gaussian(mu: float, sigma: float, beta: float = 0.95) -> float:
    """
    Closed-form CVaR of a Normal(mu, sigma^2) LOSS distribution.

        CVaR_β(L) = μ + σ · φ(Φ⁻¹(β)) / (1 − β)

    If your inputs describe a RETURN distribution, pass mu = -E[R],
    sigma = std(R) — i.e. losses = -returns.
    """
    _require_scipy("cvar_gaussian")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    z = _stats.norm.ppf(beta)
    return float(mu + sigma * _stats.norm.pdf(z) / (1.0 - beta))


def cvar_student_t(mu: float, sigma: float, nu: float,
                   beta: float = 0.95) -> float:
    """
    Closed-form CVaR of a location-scale Student-t LOSS distribution
    with ν > 1 degrees of freedom.

        CVaR_β(L) = μ + σ · (ν + t_ν⁻¹(β)²) / (ν − 1)
                          · f_ν(t_ν⁻¹(β)) / (1 − β)

    (Standard result; see McNeil-Frey-Embrechts, Quantitative Risk Mgmt §2.2.)
    """
    _require_scipy("cvar_student_t")
    if nu <= 1:
        raise ValueError("Student-t CVaR requires nu > 1")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    t_q = _stats.t.ppf(beta, df=nu)
    pdf = _stats.t.pdf(t_q, df=nu)
    factor = (nu + t_q ** 2) / (nu - 1)
    return float(mu + sigma * factor * pdf / (1.0 - beta))


# ---------------------------------------------------------------------------
# (c) Rockafellar–Uryasev auxiliary form
# ---------------------------------------------------------------------------
def cvar_rockafellar_uryasev(losses: np.ndarray,
                             beta: float = 0.95) -> tuple[float, float]:
    """
    Rockafellar–Uryasev (2000) representation:

        CVaR_β(L) = min_α { α + 1/((1−β)·S) · Σ_k max(L_k − α, 0) }

    The objective F(α) is piecewise-linear and convex; its minimizer is the
    empirical β-VaR. Concretely, taking α* equal to the ⌈β·S⌉-th order
    statistic of L gives a global minimum, so we can evaluate F there in
    closed form rather than running a numerical optimizer.

    Returns (CVaR, α*).  α* is the empirical VaR at level β.

    This is the same auxiliary structure that becomes the linear program in
    Phase-3 §3.1; running it standalone here is the sanity check that 2.5's
    methods (a) and (c) agree on the same scenario set.
    """
    losses = np.asarray(losses, dtype=float).ravel()
    S = losses.size
    if S == 0:
        raise ValueError("losses is empty")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")

    sorted_L = np.sort(losses)
    # ⌈β·S⌉-th order statistic, 1-indexed → index ⌈β·S⌉ - 1
    idx = int(np.ceil(beta * S)) - 1
    idx = min(max(idx, 0), S - 1)
    alpha_star = float(sorted_L[idx])

    one_over = 1.0 / ((1.0 - beta) * S)
    excess = np.maximum(losses - alpha_star, 0.0).sum()
    cvar = alpha_star + one_over * excess
    return float(cvar), alpha_star


# ---------------------------------------------------------------------------
# Portfolio wrapper
# ---------------------------------------------------------------------------
def portfolio_losses(R: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Convert a (S × N) RETURN scenario matrix and (N,) weight vector
    into (S,) portfolio LOSSES.
    """
    R = np.asarray(R, dtype=float)
    w = np.asarray(w, dtype=float).ravel()
    if R.ndim != 2 or w.ndim != 1 or R.shape[1] != w.size:
        raise ValueError(f"shape mismatch: R={R.shape}, w={w.shape}")
    return -(R @ w)


def portfolio_cvar(R: np.ndarray, w: np.ndarray, beta: float = 0.95,
                   method: str = "empirical") -> float:
    """
    Compute portfolio CVaR by the named method.

    method ∈ {"empirical", "ru", "gaussian", "student_t"}.
    For the parametric methods, the empirical mean and std (and ν via MLE
    for Student-t) of the realized portfolio losses are used.
    """
    L = portfolio_losses(R, w)

    if method == "empirical":
        return cvar_empirical(L, beta)
    if method == "ru":
        return cvar_rockafellar_uryasev(L, beta)[0]
    if method == "gaussian":
        return cvar_gaussian(L.mean(), L.std(ddof=1), beta)
    if method == "student_t":
        _require_scipy("portfolio_cvar(method='student_t')")
        # Fit location-scale Student-t to the loss series.
        nu, loc, scale = _stats.t.fit(L)
        return cvar_student_t(loc, scale, nu, beta)
    raise ValueError(f"unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Convenience: per-asset CVaR table 
# ---------------------------------------------------------------------------
def per_asset_cvar_table(R: np.ndarray, betas=(0.95, 0.975, 0.99)) -> dict:
    """
    Returns {beta: array of length N} with empirical CVaR per asset
    (positive = loss). Equivalent to looping cvar_empirical over columns.
    """
    R = np.asarray(R, dtype=float)
    out = {}
    for b in betas:
        out[b] = np.array([cvar_empirical(-R[:, j], b) for j in range(R.shape[1])])
    return out


__all__ = [
    "cvar_empirical",
    "cvar_gaussian",
    "cvar_student_t",
    "cvar_rockafellar_uryasev",
    "portfolio_losses",
    "portfolio_cvar",
    "per_asset_cvar_table",
    "effective_tail_size",
]
