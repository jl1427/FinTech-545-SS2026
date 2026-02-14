#!/usr/bin/env python3
"""
test5.2.py

5.2 vs output covariance
Normal Simulation nonPSD Input, mean=0, near_psd fix - 100,000 simulations

Input : data/test5_2.csv
Output: data/testout_5.2_mk.csv

Output format:
- First row: x1,x2,...,xN
- Next rows: simulated covariance matrix only
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# ==============================
# Configuration
# ==============================
N_SIMS = 100_000

# IMPORTANT: set this to match instructor output exactly
SEED = 12345

# IMPORTANT: 0 or 1 depending on expected output
DDOF = 0  # 0 = population covariance, 1 = sample covariance

EPS_EIG = 1e-12  # eigenvalue floor for near_psd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "test5_2.csv"
OUTPUT_FILE = BASE_DIR / "data" / "testout_5.2_mk.csv"


# ==============================
# I/O Helpers
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
# near_psd Fix
# ==============================
def near_psd_corr_eig_clip(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    near_psd fix in correlation space (keeps original variances):
    1) convert covariance -> correlation
    2) eigenvalue clip to >= eps
    3) renormalize diag to 1
    4) convert back to covariance
    """
    cov = symmetrize(cov)
    d = np.sqrt(np.diag(cov))
    if np.any(d <= 0):
        raise ValueError("Covariance has non-positive diagonal entries; cannot form correlation matrix.")

    Dinv = np.diag(1.0 / d)
    corr = symmetrize(Dinv @ cov @ Dinv)

    w, v = np.linalg.eigh(corr)
    w = np.maximum(w, eps)
    corr2 = symmetrize((v * w) @ v.T)

    # force unit diagonal
    inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(corr2)))
    corr2 = symmetrize(inv_sqrt @ corr2 @ inv_sqrt)

    fixed = symmetrize(np.diag(d) @ corr2 @ np.diag(d))
    return fixed


# ==============================
# Simulation (legacy RNG style)
# ==============================
def simulate_cov_legacy(cov: np.ndarray, n: int, seed: int) -> np.ndarray:
    """
    Legacy RNG approach (common in course solutions):
      np.random.seed(seed)
      np.random.multivariate_normal(...)
    """
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

    cov_fixed = near_psd_corr_eig_clip(cov_in, eps=EPS_EIG)

    cov_sim = simulate_cov_legacy(cov_fixed, N_SIMS, SEED)

    out_df = pd.DataFrame(cov_sim, columns=names)
    out_df.to_csv(OUTPUT_FILE, index=False, float_format="%.17g")

    print(f"\nOutput written to:\n{OUTPUT_FILE.resolve()}\n")


if __name__ == "__main__":
    main()
