import os
import numpy as np
from scipy.stats import norm


def var_normal_from_csv(input_file: str, output_file: str) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")

    data = np.genfromtxt(input_file, delimiter=",", skip_header=1)

    data = np.atleast_2d(data)

    if np.isnan(data).all(axis=0)[0]:
        data = data[:, 1:]

    r = data.reshape(-1)
    r = r[~np.isnan(r)]

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1)) 

    alpha = 0.05
    z = float(norm.ppf(alpha))

    var_abs = -(mu + z * sigma)
    var_diff_from_mean = -(z * sigma)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        f.write("VaR Absolute,VaR Diff from Mean\n")
        f.write(f"{var_abs},{var_diff_from_mean}\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    input_path = os.path.join(data_dir, "test7_1.csv")
    output_path = os.path.join(data_dir, "testout_8.1_mk.csv")

    var_normal_from_csv(input_path, output_path)
