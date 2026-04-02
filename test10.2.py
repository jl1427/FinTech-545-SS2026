import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def risk_budget_weights(cov, budgets):
    budgets = np.array(budgets, dtype=float)
    budgets = budgets / budgets.sum()

    def objective(x):
        return 0.5 * x @ cov @ x - np.sum(budgets * np.log(x))

    x0 = np.ones(len(budgets))
    bounds = [(1e-12, None) for _ in range(len(budgets))]

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    x = result.x
    w = x / np.sum(x)
    return w


def find_input_file():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.join(script_dir, "test5_2.csv"),
        os.path.join(script_dir, "data", "test5_2.csv"),
        "/Users/apple/Documents/FinTech-545-SS2026/testfiles/test5_2.csv",
        "/Users/apple/Documents/FinTech-545-SS2026/testfiles/data/test5_2.csv",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("Cannot find test5_2.csv")


def main():
    input_file = find_input_file()
    df = pd.read_csv(input_file)

    cov = df.values

    budgets = [1.0, 1.0, 1.0, 1.0, 0.5]

    w = risk_budget_weights(cov, budgets)

    out_df = pd.DataFrame({"W": w})
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()