
import os
import numpy as np
import pandas as pd
from scipy.stats import t

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

inp  = os.path.join(BASE_DIR, "data", "test7_2.csv")
outp = os.path.join(BASE_DIR, "data", "testout_8.5_mk.csv")

df = pd.read_csv(inp)
x = df.iloc[:, 0].to_numpy(dtype=float)

nu, loc, scale = t.fit(x)

alpha = 0.95

q = t.ppf(1 - alpha, df=nu)


pdf_q = t.pdf(q, df=nu)
es_std_left = -((nu + q**2) / (nu - 1.0)) * (pdf_q / (1 - alpha))

es_return = loc + scale * es_std_left


es_abs = -es_return

es_diff_from_mean = es_abs + loc

df_out = pd.DataFrame(
    {"ES Absolute": [es_abs], "ES Diff from Mean": [es_diff_from_mean]}
).round(6)

df_out.to_csv(outp, index=False)
print("Saved to:", outp)
print(df_out)
