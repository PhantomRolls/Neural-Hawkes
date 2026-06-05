import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MultivariateExpHawkesMLE:
    def __init__(self, eps=1e-10, penalty_stationarity=1e6):
        self.eps = eps
        self.penalty_stationarity = penalty_stationarity

    def _prepare_data(self, df):
        df = df[["time", "process"]].copy()
        df = df.sort_values("time").reset_index(drop=True)

        self.processes_ = sorted(df["process"].unique())
        self.proc_to_id_ = {p: i for i, p in enumerate(self.processes_)}

        times = df["time"].to_numpy(float)
        self.t0_ = times.min()
        self.times_ = times - self.t0_
        self.ids_ = df["process"].map(self.proc_to_id_).to_numpy(int)

        self.T_ = self.times_.max()
        self.D_ = len(self.processes_)

    def _unpack(self, theta):
        D = self.D_
        mu = np.exp(theta[:D])
        alpha = np.exp(theta[D:D + D * D]).reshape(D, D)
        beta = np.exp(theta[D + D * D:]).reshape(D, D)
        return mu, alpha, beta

    def _neg_loglik(self, theta):
        mu, alpha, beta = self._unpack(theta)
        D, T = self.D_, self.T_

        # branching matrix K_ij = alpha_ij / beta_ij
        K = alpha / beta
        rho = max(abs(np.linalg.eigvals(K)))

        penalty = 0.0
        if rho >= 1:
            penalty = self.penalty_stationarity * (rho - 0.999) ** 2

        # R[i, j] = sum_{events k of process j before t} exp(-beta[i,j]*(t-t_k))
        R = np.zeros((D, D))

        loglik = 0.0
        last_t = 0.0

        for t, proc_j in zip(self.times_, self.ids_):
            dt = t - last_t

            # decay for each target i and source j
            R *= np.exp(-beta * dt)

            lambdas = mu + np.sum(alpha * R, axis=1)

            loglik += np.log(max(lambdas[proc_j], self.eps))

            # event from source proc_j updates all target intensities
            R[:, proc_j] += 1.0

            last_t = t

        integral = np.sum(mu) * T

        for j in range(D):
            tj = self.times_[self.ids_ == j]
            if len(tj) == 0:
                continue

            for i in range(D):
                integral += np.sum(
                    alpha[i, j] / beta[i, j]
                    * (1 - np.exp(-beta[i, j] * (T - tj)))
                )

        return -(loglik - integral) + penalty

    def fit(self, df, maxiter=2000):
        self._prepare_data(df)

        D = self.D_
        counts = np.bincount(self.ids_, minlength=D)

        mu0 = np.maximum(0.5 * counts / self.T_, 1e-4)
        alpha0 = np.full((D, D), 0.1)
        beta0 = np.full((D, D), 2.0)

        theta0 = np.concatenate([
            np.log(mu0),
            np.log(alpha0.ravel()),
            np.log(beta0.ravel())
        ])

        res = minimize(
            self._neg_loglik,
            theta0,
            method="L-BFGS-B",
            options={"maxiter": maxiter}
        )

        self.result_ = res
        self.mu_, self.alpha_, self.beta_ = self._unpack(res.x)
        self.branching_ = self.alpha_ / self.beta_
        self.spectral_radius_ = max(abs(np.linalg.eigvals(self.branching_)))

        return self

    def params(self):
        names = self.processes_
        return {
            "mu": pd.Series(self.mu_, index=names),
            "alpha": pd.DataFrame(self.alpha_, index=names, columns=names),
            "beta": pd.DataFrame(self.beta_, index=names, columns=names),
            "branching": pd.DataFrame(self.branching_, index=names, columns=names),
            "spectral_radius": self.spectral_radius_,
            "success": self.result_.success,
            "message": self.result_.message,
        }
    

def run_MLE(df):
    model = MultivariateExpHawkesMLE()
    model.fit(df)
    params = model.params()
    print("mu")
    print(params["mu"])
    print("alpha")
    print(params["alpha"])
    print("beta")
    print(params["beta"])
    print("branching alpha/beta")
    print(params["branching"])

def save_mle_exp_kernel(df, save_path, t_grid=None, maxiter=2000):
    model = MultivariateExpHawkesMLE()
    model.fit(df, maxiter=maxiter)

    params = model.params()

    alpha = model.alpha_
    beta = model.beta_
    branching = model.branching_
    mu = model.mu_

    if t_grid is None:
        t_grid = np.linspace(0, 3, 400)

    D = model.D_

    phi_exp = np.zeros((len(t_grid), D, D))

    for i in range(D):
        for j in range(D):
            phi_exp[:, i, j] = alpha[i, j] * np.exp(-beta[i, j] * t_grid)

    np.savez(
        save_path,
        t=t_grid,
        phi=phi_exp,
        mu=mu,
        alpha=alpha,
        beta=beta,
        branching=branching,
        spectral_radius=model.spectral_radius_,
        proc_names=np.array(model.processes_, dtype=object),
        success=model.result_.success,
        message=model.result_.message,
    )

    print(f"Saved MLE exponential kernel to {save_path}")
    print("spectral radius =", model.spectral_radius_)
    print("success =", model.result_.success)

    return model