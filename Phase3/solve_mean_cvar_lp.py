import cvxpy as cp
import numpy as np
from scipy.stats import norm


def solve_mean_cvar_lp(
    R,
    beta=0.95,
    mean_weight=0.0,
    min_return=None,
    lower_bound=0.0,
    upper_bound=1.0,
    solver="SCS",
):
    """
    R: scenarios x assets matrix of asset returns
    beta: CVaR confidence level, e.g. 0.95
    mean_weight: reward for expected return in objective.
        0.0 gives pure CVaR minimization.
        Positive values solve: minimize CVaR - mean_weight * expected_return
    min_return: optional required expected return
    lower_bound, upper_bound:
        use lower_bound=0 for long-only
        use lower_bound=-0.2 etc. for bounded shorts
    """

    S, n = R.shape
    mu = R.mean(axis=0)

    w = cp.Variable(n)
    alpha = cp.Variable()      # VaR variable
    u = cp.Variable(S)

    portfolio_returns = R @ w
    losses = -portfolio_returns

    cvar = alpha + (1 / ((1 - beta) * S)) * cp.sum(u)
    expected_return = mu @ w

    objective = cp.Minimize(cvar - mean_weight * expected_return)

    constraints = [
        u >= losses - alpha,
        u >= 0,
        cp.sum(w) == 1,
        w >= lower_bound,
        w <= upper_bound,
    ]

    if min_return is not None:
        constraints.append(expected_return >= min_return)

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=solver)
    except Exception as e:
        # Try alternative solvers in order of preference
        print(f"Solver {solver} failed: {e}")
        for fallback in ["OSQP", "ECOS", "CVXOPT"]:
            try:
                print(f"Trying {fallback}...")
                problem.solve(solver=fallback)
                break
            except Exception:
                continue

    return {
        "status": problem.status,
        "objective": problem.value,
        "weights": w.value,
        "var": alpha.value,
        "cvar": cvar.value,
        "expected_return": expected_return.value,
        "u": u.value,
    }

def gaussian_portfolio_cvar(w, mu, Sigma, beta=0.95):
    """
    Analytical CVaR of portfolio loss L = -w^T r,
    where r ~ N(mu, Sigma).
    """
    mean_loss = -mu @ w
    std_loss = np.sqrt(w @ Sigma @ w)

    z = norm.ppf(beta)
    tail_factor = norm.pdf(z) / (1 - beta)

    var = mean_loss + std_loss * z
    cvar = mean_loss + std_loss * tail_factor

    return var, cvar

def validate_two_asset_gaussian(beta=0.95, S=200_000, seed=0):
    rng = np.random.default_rng(seed)

    mu = np.array([0.0010, 0.0006])
    Sigma = np.array([
        [0.0004, 0.0001],
        [0.0001, 0.0002],
    ])

    R = rng.multivariate_normal(mu, Sigma, size=S)

    lp_out = solve_mean_cvar_lp(
        R,
        beta=beta,
        mean_weight=0.0,
        lower_bound=0.0,
        upper_bound=1.0,
    )

    w_lp = lp_out["weights"]

    grid = np.linspace(0, 1, 5001)
    best = None

    for x in grid:
        w = np.array([x, 1 - x])
        var, cvar = gaussian_portfolio_cvar(w, mu, Sigma, beta=beta)

        if best is None or cvar < best["cvar"]:
            best = {
                "weights": w,
                "var": var,
                "cvar": cvar,
            }

    return {
        "lp_weights": w_lp,
        "lp_sample_cvar": lp_out["cvar"],
        "analytic_grid_weights": best["weights"],
        "analytic_grid_cvar": best["cvar"],
        "weight_error": w_lp - best["weights"],
    }