import numpy as np
import os

def fit_normal_from_csv(input_file, output_file):
    x = np.loadtxt(input_file, delimiter=",", skiprows=1)

    mu = np.mean(x)
    sigma = np.sqrt(np.mean((x - mu) ** 2))

    header = "mu,sigma"
    np.savetxt(
        output_file,
        np.array([[mu, sigma]]),
        delimiter=",",
        header=header,
        comments=""
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)

    input_path = os.path.join(base_dir, "test7_1.csv")
    output_path = os.path.join(base_dir, "testout_7.1output.csv")

    fit_normal_from_csv(input_path, output_path)
