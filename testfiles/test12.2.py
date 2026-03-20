import math
import os
import pandas as pd


def american_option_binomial(S, K, T, r, q, sigma, option_type, steps=200):
    option_type = option_type.strip().lower()

    if option_type not in ("call", "put"):
        raise ValueError(f"Unknown option type: {option_type}")

    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    if sigma <= 0:
        forward = S * math.exp(-q * T) - K * math.exp(-r * T)
        if option_type == "call":
            return max(forward, 0.0)
        return max(-forward, 0.0)

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    p = max(0.0, min(1.0, p))

    values = []
    for j in range(steps + 1):
        stock = S * (u ** j) * (d ** (steps - j))
        if option_type == "call":
            values.append(max(stock - K, 0.0))
        else:
            values.append(max(K - stock, 0.0))

    for i in range(steps - 1, -1, -1):
        new_values = []
        for j in range(i + 1):
            stock = S * (u ** j) * (d ** (i - j))
            continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j])

            if option_type == "call":
                exercise = max(stock - K, 0.0)
            else:
                exercise = max(K - stock, 0.0)

            new_values.append(max(continuation, exercise))

        values = new_values

    return values[0]


def american_option_with_greeks(S, K, T, r, q, sigma, option_type, steps=200):
    value = american_option_binomial(S, K, T, r, q, sigma, option_type, steps)

    dS = max(0.01, 0.01 * S)
    dSigma = 0.0001
    dr = 0.0001
    dT = min(1.0 / 365.0, T / 2.0) if T > 0 else 1.0 / 365.0

    value_up = american_option_binomial(S + dS, K, T, r, q, sigma, option_type, steps)
    value_dn = american_option_binomial(max(1e-8, S - dS), K, T, r, q, sigma, option_type, steps)
    delta = (value_up - value_dn) / (2.0 * dS)
    gamma = (value_up - 2.0 * value + value_dn) / (dS ** 2)

    vega_up = american_option_binomial(S, K, T, r, q, sigma + dSigma, option_type, steps)
    vega_dn = american_option_binomial(S, K, T, r, q, max(1e-8, sigma - dSigma), option_type, steps)
    vega = (vega_up - vega_dn) / (2.0 * dSigma)

    rho_up = american_option_binomial(S, K, T, r + dr, q, sigma, option_type, steps)
    rho_dn = american_option_binomial(S, K, T, r - dr, q, sigma, option_type, steps)
    rho = (rho_up - rho_dn) / (2.0 * dr)

    if T > dT:
        value_shorter = american_option_binomial(S, K, T - dT, r, q, sigma, option_type, steps)
        theta = (value_shorter - value) / dT
    else:
        theta = 0.0

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

        value, delta, gamma, vega, theta, rho = american_option_with_greeks(
            S, K, T, r, q, sigma, option_type, steps=200
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