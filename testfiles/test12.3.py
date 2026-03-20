import math
import os
import pandas as pd


def parse_list(cell):
    if pd.isna(cell):
        return []
    text = str(cell).strip()
    if text == "":
        return []
    return [x.strip() for x in text.split(",") if x.strip() != ""]


def present_value_dividends(div_dates, div_amts, r, day_per_year, maturity_days):
    pv = 0.0

    for d_date, d_amt in zip(div_dates, div_amts):
        div_day = float(d_date)
        amt = float(d_amt)

        if 0 < div_day <= maturity_days:
            t = div_day / day_per_year
            pv += amt * math.exp(-r * t)

    return pv


def american_option_binomial(S, K, T, r, sigma, option_type, steps=200):
    option_type = option_type.strip().lower()

    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        elif option_type == "put":
            return max(K - S, 0.0)
        else:
            raise ValueError(f"Unknown option type: {option_type}")

    if sigma <= 0:
        if option_type == "call":
            return max(S - K * math.exp(-r * T), 0.0)
        elif option_type == "put":
            return max(K * math.exp(-r * T) - S, 0.0)
        else:
            raise ValueError(f"Unknown option type: {option_type}")

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp(r * dt) - d) / (u - d)

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


def american_discrete_dividend_price(
    S, K, maturity_days, day_per_year, r, sigma, option_type,
    dividend_dates, dividend_amounts, steps=200
):
    T = maturity_days / day_per_year

    pv_div = present_value_dividends(
        dividend_dates,
        dividend_amounts,
        r,
        day_per_year,
        maturity_days
    )

    adjusted_spot = max(S - pv_div, 1e-8)

    return american_option_binomial(
        adjusted_spot, K, T, r, sigma, option_type, steps=steps
    )


def find_input_file():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.join(script_dir, "test12_3.csv"),
        os.path.join(script_dir, "data", "test12_3.csv"),
        "/Users/apple/Documents/FinTech-545-SS2026/testfiles/test12_3.csv",
        "/Users/apple/Documents/FinTech-545-SS2026/testfiles/data/test12_3.csv",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find test12_3.csv. Checked:\n" + "\n".join(candidates)
    )


def main():
    input_file = find_input_file()
    df = pd.read_csv(input_file)
    df = df.dropna(how="all")

    print(f"Reading input from: {input_file}")
    print(f"{'ID':<5} {'Price':>12}")

    for _, row in df.iterrows():
        option_id = int(row["ID"])
        option_type = str(row["Option Type"])
        S = float(row["Underlying"])
        K = float(row["Strike"])
        maturity_days = float(row["DaysToMaturity"])
        day_per_year = float(row["DayPerYear"])
        r = float(row["RiskFreeRate"])
        sigma = float(row["ImpliedVol"])

        dividend_dates = parse_list(row["DividendDates"])
        dividend_amounts = parse_list(row["DividendAmts"])

        price = american_discrete_dividend_price(
            S=S,
            K=K,
            maturity_days=maturity_days,
            day_per_year=day_per_year,
            r=r,
            sigma=sigma,
            option_type=option_type,
            dividend_dates=dividend_dates,
            dividend_amounts=dividend_amounts,
            steps=200
        )

        print(f"{option_id:<5} {price:12.6f}")


if __name__ == "__main__":
    main()