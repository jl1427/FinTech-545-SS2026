import numpy as np
import os
import math

def neg_loglik_t(theta, x
    a, mu, b = theta
    nu = 2.0 + math.exp(a)
    sigma = math.exp(b)

    n = x.size
    z = (x - mu) / sigma

    c = (math.lgamma((nu + 1) / 2)
         - math.lgamma(nu / 2)
         - 0.5 * math.log(nu * math.pi)
         - math.log(sigma))

    ll = n * c - ((nu + 1) / 2) * np.sum(np.log1p((z * z) / nu))
    return -ll

def nelder_mead(f, x0, step=0.2, max_iter=10000, tol=1e-12):
    x0 = np.array(x0, float)
    n = len(x0)

    simplex = [x0]
    for i in range(n):
        xi = x0.copy()
        xi[i] += step
        simplex.append(xi)

    simplex = np.array(simplex)
    vals = np.array([f(p) for p in simplex])

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    for _ in range(max_iter):
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]

        best = simplex[0]
        if np.max(np.abs(simplex[1:] - best)) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]

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
                # shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])

    order = np.argsort(vals)
    return simplex[order][0]

def fit_t_from_csv(input_file, output_file):
    x = np.loadtxt(input_file, delimiter=",", skiprows=1)

    mu0 = float(np.mean(x))
    sigma0 = float(np.std(x, ddof=0))
    nu0 = 10.0

    a0 = math.log(nu0 - 2.0)
    b0 = math.log(sigma0)

    theta_star = nelder_mead(lambda th: neg_loglik_t(th, x), (a0, mu0, b0))

    a, mu, b = theta_star
    nu = 2.0 + math.exp(a)
    sigma = math.exp(b)

    with open(output_file, "w") as f:
        f.write("mu,sigma,nu\n")
        f.write(("%.17f,%.17f,%.15f\n") % (mu, sigma, nu))

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    fit_t_from_csv(
        os.path.join(base_dir, "test7_2.csv"),
        os.path.join(base_dir, "testout_7.2output.csv")
    )
