import os
import numpy as np
import pandas as pd
from scipy.stats import norm, t, spearmanr

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")

portfolio_path = os.path.join(data_dir, "test9_1_portfolio.csv")
returns_path = os.path.join(data_dir, "test9_1_returns.csv")
output_path = os.path.join(data_dir, "testout_9.1_mk.csv")

portfolio = pd.read_csv(portfolio_path)
historical_returns = pd.read_csv(returns_path)

mu_A = np.mean(historical_returns['A'])
sigma_A = np.std(historical_returns['A'], ddof=1)

df_B, mu_B, sigma_B = t.fit(historical_returns['B'])

rho = spearmanr(
    historical_returns['A'],
    historical_returns['B']
)[0]

Sigma = np.array([[1.0, rho], [rho, 1.0]])
L = np.linalg.cholesky(Sigma)

np.random.seed(12345)
num_sim = 200000

Z = np.random.normal(size=(2, num_sim))
corr_Z = np.dot(L, Z)

u = norm.cdf(corr_Z)

sim_returns_A = norm.ppf(u[0, :], loc=mu_A, scale=sigma_A)
sim_returns_B = t.ppf(u[1, :], df=df_B, loc=mu_B, scale=sigma_B)

pos_A = portfolio.loc[portfolio['Stock'] == 'A', 'Holding'].values[0] * \
        portfolio.loc[portfolio['Stock'] == 'A', 'Starting Price'].values[0]

pos_B = portfolio.loc[portfolio['Stock'] == 'B', 'Holding'].values[0] * \
        portfolio.loc[portfolio['Stock'] == 'B', 'Starting Price'].values[0]

pos_total = pos_A + pos_B

pnl_A = pos_A * sim_returns_A
pnl_B = pos_B * sim_returns_B
pnl_total = pnl_A + pnl_B

alpha = 0.05

def compute_var_es(pnl_series):
    sorted_pnl = np.sort(pnl_series)
    idx = int(num_sim * alpha)

    var_threshold = sorted_pnl[idx]
    var = -var_threshold

    tail_losses = sorted_pnl[:idx + 1]
    es = -np.mean(tail_losses)

    return var, es

var_A, es_A = compute_var_es(pnl_A)
var_B, es_B = compute_var_es(pnl_B)
var_total, es_total = compute_var_es(pnl_total)

var_A_pct = var_A / pos_A
es_A_pct = es_A / pos_A
var_B_pct = var_B / pos_B
es_B_pct = es_B / pos_B
var_total_pct = var_total / pos_total
es_total_pct = es_total / pos_total


out_df = pd.DataFrame({
    'Stock': ['A', 'B', 'Total'],
    'VaR95': [var_A, var_B, var_total],
    'ES95': [es_A, es_B, es_total],
    'VaR95_Pct': [var_A_pct, var_B_pct, var_total_pct],
    'ES95_Pct': [es_A_pct, es_B_pct, es_total_pct]
})

out_df.to_csv(output_path, index=False)

print("✅ Output saved to:", output_path)
print(out_df)
