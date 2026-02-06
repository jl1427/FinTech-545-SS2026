import os
import numpy as np


def _proj_psd(A: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone."""
    A = 0.5 * (A + A.T)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, 0.0, None)
    X = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return 0.5 * (X + X.T)


def _proj_unit_diag(A: np.ndarray) -> np.ndarray:
    """Project matrix onto unit diagonal."""
    A = A.copy()
    np.fill_diagonal(A, 1.0)
    return 0.5 * (A + A.T)


def _higham_nearest_corr(A: np.ndarray, tol=1e-12, max_iter=200) -> np.ndarray:
    """
    Higham (2002) nearest correlation matrix
    Alternating projections with Dykstra correction
    """
    Y = 0.5 * (A + A.T)
    deltaS = np.zeros_like(Y)

    for _ in range(max_iter):
        Y_old = Y.copy()

        R = Y - deltaS
        X = _proj_psd(R)
        deltaS = X - R

        Y = _proj_unit_diag(X)

        if np.linalg.norm(Y - Y_old, ord="fro") <= tol * max(1.0, np.linalg.norm(Y_old, ord="fro")):
            break

    return 0.5 * (Y + Y.T)


def higham_covariance_from_csv(input_file: str, output_file: str) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        header = f.readline().strip()

    sigma = np.genfromtxt(input_file, delimiter=",", skip_header=1)
    sigma = np.atleast_2d(sigma)
    sigma = 0.5 * (sigma + sigma.T)

    d = np.sqrt(np.diag(sigma))
    d_safe = np.where(d == 0.0, 1.0, d)

    Dinv = np.diag(1.0 / d_safe)
    corr = Dinv @ sigma @ Dinv
    corr = 0.5 * (corr + corr.T)

    corr_psd = _higham_nearest_corr(corr)

    D = np.diag(d)
    sigma_psd = D @ corr_psd @ D
    sigma_psd = 0.5 * (sigma_psd + sigma_psd.T)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        f.write(header + "\n")
        for row in sigma_psd:
            f.write(",".join(str(float(x)) for x in row) + "\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    input_path = os.path.join(data_dir, "testout_1.3.csv")
    output_path = os.path.join(data_dir, "testout_3.3_mk.csv")

    higham_covariance_from_csv(input_path, output_path)
