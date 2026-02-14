import numpy as np
import csv


def read_cov_csv(path: str) -> np.ndarray:
    # file has header row: x1,x2,...
    return np.genfromtxt(path, delimiter=",", skip_header=1)


def near_psd_cov(sigma: np.ndarray, eig_floor: float = 0.0) -> np.ndarray:
    """
    Convert covariance -> correlation, project to PSD by eigenvalue clipping,
    renormalize correlation to unit diagonal, then convert back to covariance.
    """
    sigma = 0.5 * (sigma + sigma.T)

    d = np.sqrt(np.diag(sigma))
    # avoid division by 0
    d = np.where(d == 0.0, 1.0, d)

    inv_d = 1.0 / d
    corr = np.diag(inv_d) @ sigma @ np.diag(inv_d)
    corr = 0.5 * (corr + corr.T)

    w, v = np.linalg.eigh(corr)
    w = np.maximum(w, eig_floor)

    corr_psd = v @ np.diag(w) @ v.T
    corr_psd = 0.5 * (corr_psd + corr_psd.T)

    # renormalize to unit diagonal
    cdiag = np.sqrt(np.diag(corr_psd))
    cdiag = np.where(cdiag == 0.0, 1.0, cdiag)
    corr_psd = corr_psd / np.outer(cdiag, cdiag)

    sigma_psd = np.diag(d) @ corr_psd @ np.diag(d)
    sigma_psd = 0.5 * (sigma_psd + sigma_psd.T)
    return sigma_psd


def cholesky_with_jitter(sigma: np.ndarray, start: float = 1e-12) -> np.ndarray:
    """
    Ensure Cholesky works even if sigma is only semidefinite / numerically not PD.
    """
    sigma = 0.5 * (sigma + sigma.T)
    jitter = start
    for _ in range(20):
        try:
            return np.linalg.cholesky(sigma + jitter * np.eye(sigma.shape[0]))
        except np.linalg.LinAlgError:
            jitter *= 10.0
    # last resort
    return np.linalg.cholesky(sigma + (jitter * 10.0) * np.eye(sigma.shape[0]))


def main():
    in_path = "testfiles/data/test5_2.csv"
    out_path = "testfiles/data/testout_5.5_mk.csv"

    sigma = read_cov_csv(in_path)

    # 1) fix to PSD (near-PSD)
    sigma_psd = near_psd_cov(sigma, eig_floor=0.0)

    # 2) simulate 100,000 correlated normal draws (mean 0)
    L = cholesky_with_jitter(sigma_psd, start=1e-12)

    np.random.seed(0)
    Z = np.random.normal(size=(100000, sigma.shape[0]))
    X = Z @ L.T

    # 3) simulated covariance
    sigma_sim = np.cov(X, rowvar=False)  # ddof=1 (numpy default)

    # 4) write output
    n = sigma_sim.shape[0]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"x{i+1}" for i in range(n)])
        for row in sigma_sim:
            writer.writerow([float(x) for x in row])


if __name__ == "__main__":
    main()
