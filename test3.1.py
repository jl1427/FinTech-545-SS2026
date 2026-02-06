import os
import numpy as np


def _resolve_path(base_dir: str, filename: str) -> str:
    """
    Works whether script is in testfiles/ or testfiles/data/.
    Tries:
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
    raise FileNotFoundError(f"Cannot find {filename}. Tried:\n  " + "\n  ".join(candidates))


def near_psd_covariance_from_csv(input_file: str, output_file: str) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        header = f.readline().strip()

    sigma = np.genfromtxt(input_file, delimiter=",", skip_header=1, dtype=float)
    sigma = np.atleast_2d(sigma)

    sigma = 0.5 * (sigma + sigma.T)

    diag = np.diag(sigma)
    d = np.sqrt(diag)
    d[d == 0.0] = 1.0

    corr = sigma / np.outer(d, d)
    corr = 0.5 * (corr + corr.T)

    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 0.0, None)

    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    corr_psd = 0.5 * (corr_psd + corr_psd.T)

    s = np.sqrt(np.diag(corr_psd))
    s[s == 0.0] = 1.0
    corr_psd = corr_psd / np.outer(s, s)
    corr_psd = 0.5 * (corr_psd + corr_psd.T)

    sigma_psd = corr_psd * np.outer(d, d)
    sigma_psd = 0.5 * (sigma_psd + sigma_psd.T)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        f.write(header + "\n")
        for row in sigma_psd:
            f.write(",".join(str(float(x)) for x in row) + "\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)

    input_path = _resolve_path(base_dir, "testout_1.3.csv")
    out_dir = os.path.dirname(input_path)
    output_path = os.path.join(out_dir, "testout_3.1_mk.csv")

    near_psd_covariance_from_csv(input_path, output_path)
