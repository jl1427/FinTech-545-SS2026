import os
import numpy as np


def arithmetic_returns_from_csv(input_file: str, output_file: str) -> None:
    """
    6.1 Calculate arithmetic returns from price data.

    Input:  CSV with header. First column is Date (string), remaining columns are prices (float).
    Output: CSV with same header, but rows start from the 2nd date, and numeric columns are
            arithmetic returns: P_t / P_{t-1} - 1
    """

    with open(input_file, "r", newline="") as f:
        header = f.readline().strip().split(",")

    raw = np.genfromtxt(input_file, delimiter=",", skip_header=1, dtype=str)

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    dates = raw[:, 0]
    prices = raw[:, 1:].astype(float)

    rets = prices[1:] / prices[:-1] - 1.0
    out_dates = dates[1:]

    with open(output_file, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for d, row in zip(out_dates, rets):
            f.write(d + "," + ",".join(f"{x}" for x in row) + "\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    input_path = os.path.join(data_dir, "test6.csv")
    output_path = os.path.join(data_dir, "testout_6.1_mk.csv")

    arithmetic_returns_from_csv(input_path, output_path)
