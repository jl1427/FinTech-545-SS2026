import os
import numpy as np
import pandas as pd
from scipy.stats import norm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

inp  = os.path.join(BASE_DIR, "data", "test7_1.csv")
outp = os.path.join(BASE_DIR, "data", "testout_8.4_mk.csv")

df = pd.read_csv(inp)
x = df.iloc[:, 0].to_numpy(dtype=float)

mu = np.mean(x)
sigma = np.std(x, ddof=1)

alpha = 0.95
z = norm.ppf(1 - alpha)
phi = norm.pdf(z)

es_return = mu - sigma * (phi / (1 - alpha))

es_abs = -es_return

es_diff_from_mean = es_abs + mu

df_out = pd.DataFrame({
    "ES Absolute": [es_abs],
    "ES Diff from Mean": [es_diff_from_mean]
}).round(6)

df_out.to_csv(outp, index=False)
print("Saved to:", outp)
print(df_out)
