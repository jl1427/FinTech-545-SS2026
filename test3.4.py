import numpy as np
import os

def higham_correlation(input_file, output_file, tol=1e-10, max_iter=100):
    with open(input_file, "r") as f:
        header = f.readline().strip()

    R = np.genfromtxt(input_file, delimiter=",", skip_header=1)
    n = R.shape[0]

    Y = R.copy()
    delta = np.zeros_like(R)

    for _ in range(max_iter):
        Rk = Y - delta
        vals, vecs = np.linalg.eigh(Rk)
        vals[vals < 0] = 0.0
        X = vecs @ np.diag(vals) @ vecs.T
        delta = X - Rk
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)

        if np.linalg.norm(Y - R, ord="fro") < tol:
            break

    with open(output_file, "w") as out:
        out.write(header + "\n")
        for row in Y:
            out.write(",".join(repr(float(v)) for v in row) + "\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    higham_correlation(
        os.path.join(data_dir, "testout_1.4.csv"),
        os.path.join(data_dir, "testout_3.4_mk.csv")
    )
