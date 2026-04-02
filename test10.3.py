import os
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def negative_sharpe_ratio(w, mu, cov, rf):
    portfolio_return = w @ mu
    portfolio_vol = math.sqrt(max(w @ cov @ w, 1e-18))
    return -(portfolio_return - rf) / portfolio_vol


def max_sharpe_weights(mu, cov, rf, n_starts=100):
    n = len(mu)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    ]
    bounds = [(0.0, 1.0) for _ in range(n)]

    best_w = None
    best_obj = float("inf")

    rng = np.random.default_rng(42)

    starting_points = [np.ones(n) / n]
    for _ in range(n_starts - 1):
        w0 = rng.random(n)
        w0 = w0 / np.sum(w0)
        starting_points.append(w0)

    for w0 in starting_points:
        result = minimize(
            negative_sharpe_ratio,
            w0,
            args=(mu, cov, rf),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-15, "maxiter": 1000}
        )

        if not result.success:
            continue

        w = result.x
        obj = negative_sharpe_ratio(w, mu, cov, rf)

        if obj < best_obj:
            best_obj = obj
            best_w = w.copy()

    if best_w is None:
        raise ValueError("Optimization failed for all starting points")

    return best_w


def find_file(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.join(script_dir, filename),
        os.path.join(script_dir, "data", filename),
        f"/Users/apple/Documents/FinTech-545-SS2026/testfiles/{filename}",
        f"/Users/apple/Documents/FinTech-545-SS2026/testfiles/data/{filename}",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Cannot find {filename}")


def main():
    cov_file = find_file("test5_2.csv")
    mean_file = find_file("test10_3_means.csv")

    cov = pd.read_csv(cov_file).values
    mu = pd.read_csv(mean_file).values.flatten()

    rf = 0.04

    w = max_sharpe_weights(mu, cov, rf, n_starts=100)

    out_df = pd.DataFrame({"W": w})
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()