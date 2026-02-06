import numpy as np
import os
import math

def ew_covariance_mixed(input_file, output_file, lam_var=0.97, lam_corr=0.94):
    with open(input_file, "r") as f:
        header = f.readline().strip()

    X = np.genfromtxt(input_file, delimiter=",", skip_header=1)
    T, p = X.shape

    # ---- EW variance ----
    wv = np.array([(1 - lam_var) * lam_var ** (T - 1 - t) for t in range(T)])
    wv = wv / wv.sum()
    mu_v = np.sum(wv[:, None] * X, axis=0)
    Xv = X - mu_v

    var = np.zeros(p)
    for t in range(T):
        var += wv[t] * Xv[t] ** 2

    # ---- EW correlation ----
    wc = np.array([(1 - lam_corr) * lam_corr ** (T - 1 - t) for t in range(T)])
    wc = wc / wc.sum()
    mu_c = np.sum(wc[:, None] * X, axis=0)
    Xc = X - mu_c

    cov_c = np.zeros((p, p))
    for t in range(T):
        cov_c += wc[t] * np.outer(Xc[t], Xc[t])

    std_c = np.sqrt(np.diag(cov_c))
    corr = cov_c / np.outer(std_c, std_c)
    np.fill_diagonal(corr, 1.0)

    # ---- Combine ----
    D = np.diag(np.sqrt(var))
    cov = D @ corr @ D

    with open(output_file, "w") as out:
        out.write(header + "\n")
        for row in cov:
            out.write(",".join(repr(float(v)) for v in row) + "\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    ew_covariance_mixed(
        os.path.join(data_dir, "test2.csv"),
        os.path.join(data_dir, "testout_2.3_mk.csv")
    )
