import numpy as np
import os
import math

def neg_loglik_t_errors(theta, X, y):
    """
    theta = [Alpha, B1, B2, B3, a, s]
      - nu = 2 + exp(a)  => nu > 2
      - sigma = exp(s)   => sigma > 0
    """
    alpha = theta[0]
    beta = theta[1:4]
    a = theta[4]
    s = theta[5]

    nu = 2.0 + math.exp(a)
    sigma = math.exp(s)

    yhat = alpha + X @ beta
    e = y - yhat
    z = e / sigma
    n = y.size

    c = (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log(nu * math.pi)
        - math.log(sigma)
    )

    ll = n * c - ((nu + 1.0) / 2.0) * np.sum(np.log1p((z * z) / nu))
    return -ll


def nelder_mead(f, x0, step=0.05, max_iter=30000, tol=1e-12):
    x0 = np.array(x0, dtype=float)
    n = x0.size

    simplex = np.zeros((n + 1, n), dtype=float)
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1, i] += step

    vals = np.array([f(p) for p in simplex])

    alpha = 1.0
    gamma = 2.0
    rho = 0.5 
    shrink = 0.5

    for _ in range(max_iter):
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]

        best = simplex[0]
        worst = simplex[-1]

        if np.max(np.abs(simplex[1:] - best)) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        xr = centroid + alpha * (centroid - worst)
        fr = f(xr)

        if fr < vals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                vals[-1] = fe
            else:
                simplex[-1] = xr
                vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr
            vals[-1] = fr
        else:
            if fr < vals[-1]:
                xc = centroid + rho * (xr - centroid)
            else:
                xc = centroid + rho * (worst - centroid)

            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1] = xc
                vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])

    order = np.argsort(vals)
    return simplex[order][0]


def fit_t_regression_from_csv(input_file, output_file):
    data = np.loadtxt(input_file, delimiter=",", skiprows=1)
    X = data[:, 0:3]
    y = data[:, 3]
    n = y.size

    X_ols = np.column_stack([np.ones(n), X])
    b_ols = np.linalg.lstsq(X_ols, y, rcond=None)[0]
    alpha0 = float(b_ols[0])
    beta0 = b_ols[1:4]

    resid = y - (alpha0 + X @ beta0)
    sigma0 = float(np.sqrt(np.mean(resid ** 2)))
    nu0 = 5.0

    a0 = math.log(nu0 - 2.0)
    s0 = math.log(max(sigma0, 1e-12))

    theta0 = np.array([alpha0, beta0[0], beta0[1], beta0[2], a0, s0], dtype=float)

    f = lambda th: neg_loglik_t_errors(th, X, y)
    theta_star = nelder_mead(f, theta0, step=0.05, max_iter=40000, tol=1e-12)

    alpha_hat = theta_star[0]
    b1_hat, b2_hat, b3_hat = theta_star[1:4]
    nu_hat = 2.0 + math.exp(theta_star[4])
    sigma_hat = math.exp(theta_star[5])

    mu_hat = 0.0

    with open(output_file, "w") as out:
        out.write("mu,sigma,nu,Alpha,B1,B2,B3\n")
        out.write(
            ("%.1f,%.17f,%.15f,%.17f,%.17f,%.17f,%.17f\n")
            % (mu_hat, sigma_hat, nu_hat, alpha_hat, b1_hat, b2_hat, b3_hat)
        )


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    fit_t_regression_from_csv(
        os.path.join(base_dir, "test7_3.csv"),
        os.path.join(base_dir, "testout_7.3output.csv"),
    )
