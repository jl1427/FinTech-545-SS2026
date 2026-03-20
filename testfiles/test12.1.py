import math
import os
import pandas as pd


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def gbsm_with_greeks(S, K, T, r, q, sigma, option_type):
    option_type = option_type.strip().lower()

    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if option_type == "call":
            value = max(S - K, 0.0)
            delta = 1.0 if S > K else 0.0
        elif option_type == "put":
            value = max(K - S, 0.0)
            delta = -1.0 if S < K else 0.0
        else:
            raise ValueError(f"Unknown option type: {option_type}")

        gamma = 0.0
        vega = 0.0
        theta = 0.0
        rho = 0.0
        return value, delta, gamma, vega, theta, rho

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if option_type == "call":
        value = S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        delta = math.exp(-q * T) * norm_cdf(d1)
        theta = (
            -S * math.exp(-q * T) * norm_pdf(d1) * sigma / (2.0 * sqrtT)
            - r * K * math.exp(-r * T) * norm_cdf(d2)
            + q * S * math.exp(-q * T) * norm_cdf(d1)
        )
        rho = K * T * math.exp(-r * T) * norm_cdf(d2)

    elif option_type == "put":
        value = K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)
        delta = math.exp(-q * T) * (norm_cdf(d1) - 1.0)
        theta = (
            -S * math.exp(-q * T) * norm_pdf(d1) * sigma / (2.0 * sqrtT)
            + r * K * math.exp(-r * T) * norm_cdf(-d2)
            - q * S * math.exp(-q * T) * norm_cdf(-d1)
        )
        rho = -K * T * math.exp(-r * T) * norm_cdf(-d2)

    else:
        raise ValueError(f"Unknown option type: {option_type}")

    gamma = math.exp(-q * T) * norm_pdf(d1) / (S * sigma * sqrtT)
    vega = S * math.exp(-q * T) * norm_pdf(d1) * sqrtT

    return value, delta, gamma, vega, theta, rho


def find_input_file():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.join(script_dir, "test12_1.csv"),
        os.path.join(script_dir, "data", "test12_1.csv"),
        "/Users/apple/Documents/FinTech-545-SS2026/testfiles/test12_1.csv",
        "/Users/apple/Documents/FinTech-545-SS2026/testfiles/data/test12_1.csv",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find test12_1.csv. Checked:\n" + "\n".join(candidates)
    )


def main():
    input_file = find_input_file()
    df = pd.read_csv(input_file)
    df = df.dropna(how="all")

    print(f"Reading input from: {input_file}")
    print(f"{'ID':<5} {'Value':>12} {'Delta':>12} {'Gamma':>12} {'Vega':>12} {'Theta':>12} {'Rho':>12}")

    for _, row in df.iterrows():
        option_id = int(row["ID"])
        option_type = str(row["Option Type"])
        S = float(row["Underlying"])
        K = float(row["Strike"])
        T = float(row["DaysToMaturity"]) / float(row["DayPerYear"])
        r = float(row["RiskFreeRate"])
        q = float(row["DividendRate"])
        sigma = float(row["ImpliedVol"])

        value, delta, gamma, vega, theta, rho = gbsm_with_greeks(
            S, K, T, r, q, sigma, option_type
        )

        print(
            f"{option_id:<5}"
            f"{value:12.6f}"
            f"{delta:12.6f}"
            f"{gamma:12.6f}"
            f"{vega:12.6f}"
            f"{theta:12.6f}"
            f"{rho:12.6f}"
        )


if __name__ == "__main__":
    main()