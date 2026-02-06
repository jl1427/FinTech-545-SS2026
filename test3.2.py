import numpy as np
import os

def near_psd_correlation(input_file, output_file):
    with open(input_file, "r") as f:
        header = f.readline().strip()

    R = np.genfromtxt(input_file, delimiter=",", skip_header=1)

    vals, vecs = np.linalg.eigh(R)
    vals[vals < 0] = 0.0

    R_psd = vecs @ np.diag(vals) @ vecs.T

    d = np.sqrt(np.diag(R_psd))
    R_psd = R_psd / np.outer(d, d)
    np.fill_diagonal(R_psd, 1.0)

    with open(output_file, "w") as out:
        out.write(header + "\n")
        for row in R_psd:
            out.write(",".join(repr(float(v)) for v in row) + "\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    near_psd_correlation(
        os.path.join(data_dir, "testout_1.4.csv"),
        os.path.join(data_dir, "testout_3.2_mk.csv")
    )
