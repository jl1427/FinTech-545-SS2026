# test2.2.py
import os
import numpy as np


def _resolve_path(base_dir: str, filename: str) -> str:
    """
    Works whether this script is in testfiles/ or testfiles/data/ by trying:
      1) base_dir/filename
      2) base_dir/data/filename
      3) base_dir/../data/filename
    """
    candidates = [
        os.path.join(base_dir, filename),
        os.path.join(base_dir, "data", filename),
        os.path.join(os.path.dirname(base_dir), "data", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def ew_correlation_from_csv(input_file: str, output_file: str, lam: float = 0.94) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        header = f.readline().strip()

    X = np.genfromtxt(input_file, delimiter=",", skip_header=1, dtype=float)
    X = np.atleast_2d(X)
    T, n = X.shape

    w = np.array([lam ** (T - 1 - t) for t in range(T)], dtype=float)
    w = w / w.sum()

    mu = (w[:, None] * X).sum(axis=0)

    Xc = X - mu
    cov = (w[:, None, None] * (Xc[:, :, None] * Xc[:, None, :])).sum(axis=0)

    std = np.sqrt(np.diag(cov))
    std[std == 0.0] = 1.0
    corr = cov / np.outer(std, std)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        f.write(header + "\n")
        for row in corr:
            f.write(",".join(str(float(x)) for x in row) + "\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)

    input_path = _resolve_path(base_dir, "test2.csv")
    data_dir = os.path.dirname(input_path)
    output_path = os.path.join(data_dir, "testout_2.2_mk.csv")

    ew_correlation_from_csv(input_path, output_path, lam=0.94)
