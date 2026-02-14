#!/usr/bin/env python3
"""
test5.3.py

5.3 simulations, compare input vs output covariance
Normal Simulation, PSD Input, 0 mean, Higham fix - 100,000 simulations

Input : data/test5_3.csv
Output: data/testout_5.3_mk.csv
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# ==============================
# Configuration
# ==============================
N_SIMS = 100_000
SEED = 12345        # adjust if needed to match expected output
DDOF = 0            # 0 or 1 depending on instructor file
TOL = 1e-12
MAX_ITER = 100

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "test5_3.csv"
OUTPUT_FILE = BASE_DIR / "data" / "testout_5.3_mk.csv"


# ==============================
# Utilities
# ==============================
def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def read_cov_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path, header=None)

    header = df.iloc[0].tolist()
    has_header = not all(_is_number(v) for v in header)

    if has_header:
        names = [str(v).strip() for v in header]
        df = df.iloc[1:].reset_index(drop=True)
    else:
        names = []

    first_col = df.iloc[:, 0].tolist()
    has_index = not all(_is_number(v) for v in first_col)
    if has_index:
        df = df.iloc[:, 1:].reset_index(drop=True)

    cov = df.to_numpy(dtype=float)
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance matrix must be square. Got {cov.shape}")

    if (not names) or (len(names) != cov.shape[0]):
        names = [f"x{i+1}" for i in range(cov.shape[0])]

    return cov, names


def symmetrize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


# ==============================
# Higham Nearest PSD
# ==============================
def higham_nearest_psd(a: np.ndarray,
                       tol: float = 1e-12,
                       max_iter: int = 100) -> np.ndarray:
    """
    Higham (2002) nearest PSD matrix (Frobenius norm).
    """
    a = symmetrize(a)
    y = a.copy()
    delta_s = np.zeros_like(a)

    for _ in range(max_iter):
        r = y - delta_s
        w, v = np.linalg.eigh(symmetrize(r))
        w = np.maximum(w, 0.0)
        x = symmetrize((v * w) @ v.T)
        delta_s = x - r
        y = x

        if np.min(w) >= -tol:
            break

    # numerical safeguard
    min_eig = np.min(np.linalg.eigvalsh(y))
    if min_eig < 0:
        y += np.eye(y.shape[0]) * (-min_eig + tol)

    return symmetrize(y)


# ==============================
# Simulation (legacy RNG style)
# ==============================
def simulate_cov_legacy(cov: np.ndarray, n: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    d = cov.shape[0]
    x = np.random.multivariate_normal(mean=np.zeros(d), cov=cov, size=n)
    return np.cov(x, rowvar=False, ddof=DDOF)


# ==============================
# Main
# ==============================
def main() -> None:
    cov_in, names = read_cov_csv(INPUT_FILE)
    cov_in = symmetrize(cov_in)

    cov_fixed = higham_nearest_psd(cov_in, tol=TOL, max_iter=MAX_ITER)

    cov_sim = simulate_cov_legacy(cov_fixed, N_SIMS, SEED)

    out_df = pd.DataFrame(cov_sim, columns=names)
    out_df.to_csv(OUTPUT_FILE, index=False, float_format="%.17g")

    print(f"\nOutput written to:\n{OUTPUT_FILE.resolve()}\n")


if __name__ == "__main__":
    main()
