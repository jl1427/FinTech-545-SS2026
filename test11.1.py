import os
import math
import numpy as np
import pandas as pd


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
    returns_file = find_file("test11_1_returns.csv")
    weights_file = find_file("test11_1_weights.csv")

    returns_df = pd.read_csv(returns_file)
    w0 = pd.read_csv(weights_file).iloc[:, 0].values.astype(float)

    R = returns_df.values
    asset_names = returns_df.columns.tolist()
    T, n = R.shape


    asset_values = np.zeros((T + 1, n))
    asset_values[0] = w0

    for t in range(T):
        asset_values[t + 1] = asset_values[t] * (1.0 + R[t])

    begin_weights = asset_values[:-1] / asset_values[:-1].sum(axis=1, keepdims=True)


    contribution_stream = begin_weights * R
    portfolio_returns = contribution_stream.sum(axis=1)


    total_asset_returns = np.prod(1.0 + R, axis=0) - 1.0
    total_portfolio_return = asset_values[-1].sum() - 1.0


    k = np.array([
        math.log(1.0 + rp) / rp if abs(rp) > 1e-14 else 1.0
        for rp in portfolio_returns
    ])
    k_bar = math.log(1.0 + total_portfolio_return) / total_portfolio_return

    return_attribution = np.sum(
        contribution_stream * (k[:, None] / k_bar),
        axis=0
    )

    sigma_p = np.std(portfolio_returns, ddof=1)
    vol_attribution = np.array([
        np.cov(contribution_stream[:, i], portfolio_returns, ddof=1)[0, 1] / sigma_p
        for i in range(n)
    ])
    total_vol = vol_attribution.sum()

    columns = ["Value"] + asset_names + ["Portfolio"]

    out_df = pd.DataFrame([
        ["TotalReturn"] + list(total_asset_returns) + [total_portfolio_return],
        ["Return Attribution"] + list(return_attribution) + [total_portfolio_return],
        ["Vol Attribution"] + list(vol_attribution) + [total_vol],
    ], columns=columns)

    print(out_df.to_csv(index=False, lineterminator="\n"), end="")


if __name__ == "__main__":
    main()