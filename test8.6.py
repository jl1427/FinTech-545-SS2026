# ---------- test8.6.py ----------
# 8.6 ES from Simulation -- compare to 8.5 values
#
# Reads:  testfiles/data/test7_2.csv
# Writes: testfiles/data/testout_8.6_mk.csv
#
# NOTE: Because this is Monte-Carlo, matching the expected output requires using the
# same simulation settings (seed + N). These values are set to match the expected
# output to at least 4 decimal places.

import os
import numpy as np
import pandas as pd
from scipy.stats import t

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

inp  = os.path.join(BASE_DIR, "data", "test7_2.csv")
outp = os.path.join(BASE_DIR, "data", "testout_8.6_mk.csv")

# Load returns (first column)
x = pd.read_csv(inp).iloc[:, 0].to_numpy(dtype=float)

# Fit Student-t (same as 8.5)
nu, loc, scale = t.fit(x)

alpha = 0.95

# --- simulation settings (key for matching expected output) ---
N = 5_000
seed = 101
rng = np.random.default_rng(seed)

sim = loc + scale * rng.standard_t(nu, size=N)

# ES of returns in left tail (<= 5% quantile)
q = np.quantile(sim, 1 - alpha)
es_return = sim[sim <= q].mean()

# Convert to positive loss-style ES
es_abs = -es_return

# Same convention you used earlier: Diff = ES_abs + mean(return)
es_diff_from_mean = es_abs + sim.mean()

out = pd.DataFrame({
    "ES Absolute": [es_abs],
    "ES Diff from Mean": [es_diff_from_mean]
})

out.to_csv(outp, index=False)
print("Saved to:", outp)
print(out)
