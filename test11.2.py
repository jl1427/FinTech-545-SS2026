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
    factor_file = find_file("test11_2_factor_returns.csv")
    stock_file = find_file("test11_2_stock_returns.csv")
    beta_file = find_file("test11_2_beta.csv")
    weight_file = find_file("test11_2_weights.csv")

    F_df = pd.read_csv(factor_file)
    R_df = pd.read_csv(stock_file)
    B_df = pd.read_csv(beta_file)
    w0 = pd.read_csv(weight_file).iloc[:, 0].values.astype(float)

    F = F_df.values                  
    R = R_df.values                 
    B = B_df.iloc[:, 1:].values.astype(float)  

    factor_names = F_df.columns.tolist()
    T, n = R.shape
    k_factors = F.shape[1]


    asset_values = np.zeros((T + 1, n))
    asset_values[0] = w0

    for t in range(T):
        asset_values[t + 1] = asset_values[t] * (1.0 + R[t])

    begin_weights = asset_values[:-1] / asset_values[:-1].sum(axis=1, keepdims=True)


    stock_contrib_stream = begin_weights * R
    Rp = stock_contrib_stream.sum(axis=1)

    total_portfolio_return = asset_values[-1].sum() - 1.0


    factor_exposure = begin_weights @ B       
    factor_contrib_stream = factor_exposure * F 
    alpha_stream = Rp - factor_contrib_stream.sum(axis=1)


    total_factor_returns = np.prod(1.0 + F, axis=0) - 1.0
    total_alpha_return = np.prod(1.0 + alpha_stream) - 1.0


    k_carino = np.array([
        math.log(1.0 + rp) / rp if abs(rp) > 1e-14 else 1.0
        for rp in Rp
    ])
    k_bar = math.log(1.0 + total_portfolio_return) / total_portfolio_return

    all_streams = np.column_stack([factor_contrib_stream, alpha_stream])
    return_attribution = np.sum(all_streams * (k_carino[:, None] / k_bar), axis=0)


    sigma_p = np.std(Rp, ddof=1)
    vol_attribution = np.array([
        np.cov(all_streams[:, j], Rp, ddof=1)[0, 1] / sigma_p
        for j in range(all_streams.shape[1])
    ])
    total_vol = vol_attribution.sum()


    columns = ["Value"] + factor_names + ["Alpha", "Portfolio"]

    out_df = pd.DataFrame([
        ["TotalReturn"] + list(total_factor_returns) + [total_alpha_return, total_portfolio_return],
        ["Return Attribution"] + list(return_attribution[:-1]) + [return_attribution[-1], total_portfolio_return],
        ["Vol Attribution"] + list(vol_attribution[:-1]) + [vol_attribution[-1], total_vol],
    ], columns=columns)

    print(out_df.to_csv(index=False, lineterminator="\n"), end="")


if __name__ == "__main__":
    main()