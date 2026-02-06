import os
import numpy as np


def _resolve_path(base_dir: str, filename: str) -> str:
    """
    Try these locations so it works whether your script is in:
      - testfiles/
      - testfiles/data/
    """
    candidates = [
        os.path.join(base_dir, filename),
        os.path.join(base_dir, "data", filename),
        os.path.join(os.path.dirname(base_dir), "data", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find {filename}. Tried:\n" + "\n".join(candidates))


def chol_psd_from_csv(input_file: str, output_file: str) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        header = f.readline().strip()

    sigma = np.genfromtxt(input_file, delimiter=",", skip_header=1, dtype=float)
    sigma = np.atleast_2d(sigma)

    sigma = 0.5 * (sigma + sigma.T)

    L = np.linalg.cholesky(sigma)

    L = np.tril(L)
    L[np.abs(L) < 1e-15] = 0.0

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        f.write(header + "\n")
        for row in L:
            f.write(",".join(str(float(x)) for x in row) + "\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)

    input_path = _resolve_path(base_dir, "testout_3.1.csv")

    out_dir = os.path.dirname(input_path)
    output_path = os.path.join(out_dir, "testout_4.1_mk.csv")

    chol_psd_from_csv(input_path, output_path)
    print("Wrote:", output_path)
