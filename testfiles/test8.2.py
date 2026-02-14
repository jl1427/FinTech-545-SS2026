import os
import numpy as np

try:
    from scipy.stats import t
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "SciPy is required for test8.2. Install it with:\n"
        "  pip install scipy\n"
        "or (conda):\n"
        "  conda install scipy\n"
    ) from e


def _resolve_path(base_dir: str, filename: str) -> str:
    """
    Try common locations so it works whether you run from repo root or testfiles/.
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


def var_from_t_distribution(input_file: str, output_file: str, p: float = 0.05) -> None:
    data = np.genfromtxt(input_file, delimiter=",", skip_header=1, dtype=float)
    data = np.atleast_1d(data)

    if data.ndim == 2:
        data = data[:, 0]

    df_, loc, scale = t.fit(data)

    q = t.ppf(p, df_, loc=loc, scale=scale)

    var_abs = -q
    var_diff = loc - q

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        f.write("VaR Absolute,VaR Diff from Mean\n")
        f.write(f"{float(var_abs)},{float(var_diff)}\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    inp = _resolve_path(base_dir, "test7_2.csv")
    out = os.path.join(os.path.dirname(inp), "testout8_2_mk.csv")
    var_from_t_distribution(inp, out)
