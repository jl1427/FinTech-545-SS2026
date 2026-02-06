import numpy as np
import os
import math

def correlation_pairwise(input_file, output_file):
    with open(input_file, "r") as f:
        header = f.readline().strip()

    X = np.genfromtxt(input_file, delimiter=",", skip_header=1)
    p = X.shape[1]

    corr = np.empty((p, p), dtype=np.float64)

    for i in range(p):
        xi = X[:, i]
        for j in range(p):
            if i == j:
                corr[i, j] = 1.0
                continue

            xj = X[:, j]
            mask = (~np.isnan(xi)) & (~np.isnan(xj))
            a = xi[mask]
            b = xj[mask]

            n = a.size
            if n <= 1:
                corr[i, j] = np.nan
                continue

            da = a - a.mean()
            db = b - b.mean()

            var_i = np.sum(da * da) / (n - 1)
            var_j = np.sum(db * db) / (n - 1)

            if var_i <= 0.0 or var_j <= 0.0:
                corr[i, j] = np.nan
            else:
                cov_ij = np.sum(da * db) / (n - 1)
                corr[i, j] = cov_ij / math.sqrt(var_i * var_j)

    np.fill_diagonal(corr, 1.0)

    with open(output_file, "w") as out:
        out.write(header + "\n")
        for row in corr:
            out.write(",".join(repr(float(v)) for v in row) + "\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    correlation_pairwise(
        os.path.join(data_dir, "test1.csv"),
        os.path.join(data_dir, "testout_1.4_mk.csv")
    )
