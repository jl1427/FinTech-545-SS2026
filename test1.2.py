import numpy as np
import os

def correlation_skip_missing(input_file, output_file):
    # Read header
    with open(input_file, "r") as f:
        header = f.readline().strip()

    # Load as float64 first
    X = np.genfromtxt(input_file, delimiter=",", skip_header=1)

    # Drop rows with ANY missing value
    X = X[~np.isnan(X).any(axis=1)]

    # ---- high-precision correlation ----
    X = X.astype(np.longdouble)
    n = X.shape[0]

    mean = np.mean(X, axis=0)
    Xc = X - mean

    # sample covariance (n-1)
    cov = (Xc.T @ Xc) / np.longdouble(n - 1)

    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    # cast back to float and force exact 1.0 on diagonal
    corr = np.array(corr, dtype=np.float64)
    np.fill_diagonal(corr, 1.0)

    # Write using Python float repr (matches "1.0" style)
    with open(output_file, "w") as out:
        out.write(header + "\n")
        for row in corr:
            out.write(",".join(repr(float(v)) for v in row) + "\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    correlation_skip_missing(
        os.path.join(data_dir, "test1.csv"),
        os.path.join(data_dir, "testout_1.2_mk.csv")
    )
