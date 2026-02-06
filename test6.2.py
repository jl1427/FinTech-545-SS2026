import numpy as np
import os

def log_returns(input_file, output_file):
    with open(input_file, "r") as f:
        header = f.readline().strip()

    P = np.genfromtxt(input_file, delimiter=",", skip_header=1)

    L = np.log(P[1:] / P[:-1])

    with open(output_file, "w") as out:
        out.write(header + "\n")
        for row in L:
            out.write(",".join(repr(float(v)) for v in row) + "\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    log_returns(
        os.path.join(data_dir, "test6.csv"),
        os.path.join(data_dir, "testout_6.2_mk.csv")
    )
