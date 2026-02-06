import numpy as np
import os

def covariance_skip_missing(input_file, output_file):
    with open(input_file, "r") as f:
        header = f.readline().strip()

    X = np.genfromtxt(input_file, delimiter=",", skip_header=1)

    X_clean = X[~np.isnan(X).any(axis=1)]

    cov = np.cov(X_clean, rowvar=False, bias=False)

    np.savetxt(
        output_file,
        cov,
        delimiter=",",
        fmt="%.16f",
        header=header,
        comments=""
    )

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    covariance_skip_missing(
        os.path.join(data_dir, "test1.csv"),
        os.path.join(data_dir, "testout_1.1_mk.csv")
    )
