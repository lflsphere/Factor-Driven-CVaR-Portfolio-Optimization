# Finance Project - Mean-CVaR Portfolio Optimization

This project implements portfolio optimization using Conditional Value at Risk (CVaR) with linear programming.

## Environment Setup

A conda environment has been created for this project with all necessary dependencies.

### Activate the environment

```bash
conda activate finance_project
```

### Deactivate the environment

```bash
conda deactivate
```

### Recreate the environment (if needed)

```bash
conda env create -f environment.yml
```

### Remove the environment (if needed)

```bash
conda deactivate
conda env remove -n finance_project
```

## Dependencies

The environment includes:
- Python 3.11
- NumPy - numerical computing
- SciPy - scientific computing (for statistical distributions)
- CVXPY - convex optimization
- Pandas - data manipulation
- Matplotlib - plotting
- Jupyter - interactive notebooks
- MOSEK - commercial optimization solver (requires license for large problems)

## Usage

After activating the environment, you can run the optimization script:

```python
python solve_mean_cvar_lp.py
```

Or use it in a Python script/notebook:

```python
from solve_mean_cvar_lp import solve_mean_cvar_lp
import numpy as np

# Load or generate scenario returns
R = np.loadtxt('scenario_returns_matrix.csv', delimiter=',')

# Solve for optimal portfolio
result = solve_mean_cvar_lp(
    R,
    beta=0.95,
    mean_weight=0.0,
    lower_bound=0.0,
    upper_bound=1.0
)

print(f"Optimal weights: {result['weights']}")
print(f"CVaR: {result['cvar']}")
print(f"Expected return: {result['expected_return']}")