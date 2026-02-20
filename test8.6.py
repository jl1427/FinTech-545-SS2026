
import os
import numpy as np
import pandas as pd
from scipy.stats import t

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

inp  = os.path.join(BASE_DIR, "data", "test7_2.csv")
outp = os.path.join(BASE_DIR, "data", "testout_8.6_mk.csv")

x = pd.read_csv(inp).iloc[:, 0].to_numpy(dtype=float)

nu, loc, scale = t.fit(x)

alpha = 0.95

N = 5_000
seed = 101
rng = np.random.default_rng(seed)

sim = loc + scale * rng.standard_t(nu, size=N)

q = np.quantile(sim, 1 - alpha)
es_return = sim[sim <= q].mean()


es_abs = -es_return

es_diff_from_mean = es_abs + sim.mean()

out = pd.DataFrame({
    "ES Absolute": [es_abs],
    "ES Diff from Mean": [es_diff_from_mean]
})

out.to_csv(outp, index=False)
print("Saved to:", outp)
print(out)
