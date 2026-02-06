import numpy as np
import os

def ew_covariance(input_file, output_file, lam=0.97):
    with open(input_file, "r") as f:
        header = f.readline().strip()

    X = np.genfromtxt(input_file, delimiter=",", skip_header=1)
    T, p = X.shape

    w = np.array([(1 - lam) * lam ** (T - 1 - t) for t in range(T)])
    w = w / w.sum()

    mu = np.sum(w[:, None] * X, axis=0)

    Xc = X - mu
    cov = np.zeros((p, p))
    for t in range(T):
        cov += w[t] * np.outer(Xc[t], Xc[t])

    with open(output_file, "w") as out:
        out.write(header + "\n")
        for row in cov:
            out.write(",".join(repr(float(v)) for v in row) + "\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    ew_covariance(
        os.path.join(data_dir, "test2.csv"),
        os.path.join(data_dir, "testout_2.1_mk.csv"),
        lam=0.97
    )
