import numpy as np
import os

def covariance_pairwise(input_file, output_file):
    # Read header
    with open(input_file, "r") as f:
        header = f.readline().strip()

    # Load data (missing -> nan)
    X = np.genfromtxt(input_file, delimiter=",", skip_header=1)
    p = X.shape[1]

    cov = np.empty((p, p), dtype=np.float64)

    for i in range(p):
        xi = X[:, i]
        for j in range(p):
            xj = X[:, j]

            mask = (~np.isnan(xi)) & (~np.isnan(xj))
            a = xi[mask]
            b = xj[mask]

            n = a.size
            if n <= 1:
                cov[i, j] = np.nan
            else:
                # sample covariance (divide by n-1)
                cov[i, j] = np.sum((a - a.mean()) * (b - b.mean())) / (n - 1)

    # Save with header, match formatting style
    # Using repr-style writing avoids trailing-zero / rounding issues.
    with open(output_file, "w") as out:
        out.write(header + "\n")
        for row in cov:
            out.write(",".join(repr(float(v)) for v in row) + "\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    covariance_pairwise(
        os.path.join(data_dir, "test1.csv"),
        os.path.join(data_dir, "testout_1.3_mk.csv")
    )
