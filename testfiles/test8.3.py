import numpy as np
from scipy.stats import t
import csv
import os


def _resolve_path(base_dir: str, filename: str) -> str:
    candidates = [
        os.path.join(base_dir, filename),
        os.path.join(base_dir, "data", filename),
        os.path.join(os.path.dirname(base_dir), "data", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def load_returns(csv_path: str) -> np.ndarray:
    """
    Load returns from CSV.
    If file has 1 numeric column -> returns 1D.
    If file has multiple columns -> use first column.
    """
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=float)
    data = np.asarray(data)

    if data.ndim == 1:
        x = data
    else:
        x = data[:, 0]

    # drop NaNs if any
    x = x[~np.isnan(x)]
    return x


def main():
    base_dir = os.path.dirname(__file__)
    in_path = _resolve_path(base_dir, "test7_2.csv")
    out_path = _resolve_path(base_dir, "testout_8.3_mk.csv")

    # 1) load returns
    x = load_returns(in_path)

    # 2) fit t distribution (same as 8.2)
    df, loc, scale = t.fit(x)

    # 3) simulate 100,000 (seed must match)
    np.random.seed(0)
    simulated = t.rvs(df, loc=loc, scale=scale, size=100000)

    # 4) compute 5% VaR
    q05 = np.quantile(simulated, 0.05)
    var_abs = -q05
    var_diff_mean = var_abs + simulated.mean()

    # 5) write output EXACT header format
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["VaR Absolute", "VaR Diff from Mean"])
        w.writerow([float(var_abs), float(var_diff_mean)])


if __name__ == "__main__":
    main()
