import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tick.hawkes import HawkesExpKern
import matplotlib.pyplot as plt
from data.real.utils import load_real_data

df = load_real_data(["USDC/WETH 0.05"], file="30days")
df = df[df["event_type"] == "swap"]
df["process"] = np.where(df["SwapX2Y"], "X2Y", "Y2X")
timestamps = df.groupby("process")["time"].apply(lambda s: s.to_numpy()).to_list()

# data_path = "data/simulation/d=2.parquet"
# df = pd.read_parquet(data_path)
# timestamps = df.groupby("process")["time"].apply(lambda s: s.to_numpy()).to_list()

def fit_hawkes_exp_gridsearch(
    timestamps,
    beta_grid,
    penalty="none",     
    C=1e3,
    solver="svrg",     
    step=1e-5,
    tol=1e-7,
    max_iter=2000
):
    # recentrage temporel
    t0 = min(arr[0] for arr in timestamps if len(arr) > 0)
    timestamps = [np.asarray(arr, dtype=float) - t0 for arr in timestamps]

    best = None
    results = []

    for beta in beta_grid:
        learner = HawkesExpKern(
            decays=beta,
            gofit="likelihood",
            penalty=penalty,
            C=C,
            solver=solver,
            step=step,
            tol=tol,
            max_iter=max_iter,
            verbose=False
        )

        learner.fit(timestamps)
        score = learner.score()

        result = {
            "beta": beta,
            "score": score,
            "baseline": learner.baseline.copy(),
            "adjacency": learner.adjacency.copy(),
            "learner": learner
        }
        results.append(result)

        if best is None or score > best["score"]:
            best = result

    return best, results


def plot_kernels(alpha_mat, beta, t_max=100.0, n_points=1000):
    d = alpha_mat.shape[0]
    t = np.linspace(0, t_max, n_points)

    fig, axes = plt.subplots(d, d, figsize=(4*d, 3*d), squeeze=False)

    for i in range(d):
        for j in range(d):
            phi = alpha_mat[i, j] * beta * np.exp(-beta * t)
            ax = axes[i, j]
            ax.plot(t, phi)
            ax.set_title(f"phi_{i}{j}(t)")
            ax.set_xlabel("t")
            ax.set_ylabel("phi")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

beta_grid = np.logspace(-4, -1, 10)

best, all_results = fit_hawkes_exp_gridsearch(
    timestamps=timestamps,
    beta_grid=beta_grid,
    penalty="none",
    solver="svrg",
    step=1e-5,
    max_iter=5000
)

print("Best beta:", best["beta"])
print("Best score:", best["score"])
print("Baseline mu:\n", best["baseline"])
print("Alpha:\n", best["adjacency"])

plot_kernels(best["adjacency"], best["beta"], t_max=500)