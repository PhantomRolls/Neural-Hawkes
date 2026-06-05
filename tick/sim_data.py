import numpy as np
from tick.hawkes import SimuHawkesExpKernels
from tick.plot import plot_hawkes_kernels
import pandas as pd

D = 2

G = np.array([
    [0.16, 0.04],
    [0.03, 0.15],
], dtype=float)

beta = np.array([
    [2.5, 3.0],
    [2.7, 2.2],
], dtype=float)

alpha = G * beta

lambda_bar = np.ones(D)
mu = (np.eye(D) - G) @ lambda_bar


print("mu =", mu)
print("alpha=\n", alpha)
print("beta=\n", beta)

hawkes = SimuHawkesExpKernels(
    adjacency=alpha,
    decays=beta,
    baseline=mu,
    end_time=1e6,
)

hawkes.simulate()
timestamps = hawkes.timestamps
print([len(ts) for ts in timestamps])
print("Premiers events dim0:", timestamps[0][:5])

dfs = []
for i in range(D):
    dfs.append(pd.DataFrame({
        "time": timestamps[i],
        "process": f"P{i+1}"
    }))
df = pd.concat([df for df in dfs], ignore_index=True).sort_values("time")
df.to_csv("data/d=2_big.csv")

import matplotlib.pyplot as plt
import numpy as np

def plot_counts(timestamps):
    plt.figure(figsize=(8,4))
    for m, ts in enumerate(timestamps):
        if len(ts) == 0:
            continue
        ts = np.asarray(ts)
        plt.step(ts, np.arange(1, len(ts)+1), where="post", label=f"dim {m}")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Cumulative count")
    plt.title("Cumulative events")
    plt.tight_layout()
    plt.show()

plot_counts(timestamps)
    
def plot_exp_kernels(alpha, beta, t_max=10.0, n=400):
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    M = alpha.shape[0]
    t = np.linspace(0, t_max, n)

    fig, axes = plt.subplots(M, M, figsize=(2.2*M, 2.0*M),
                             sharex=True, sharey=True)

    if M == 1:
        axes = np.array([[axes]])

    for i in range(M):
        for j in range(M):
            y = alpha[i, j] * np.exp(-beta[i, j] * t)
            axes[i, j].plot(t, y)
            axes[i, j].set_title(f"{i}→{j}", fontsize=9)

    fig.suptitle("Hawkes exponential kernels")
    plt.tight_layout()
    plt.show()
    
plot_exp_kernels(alpha, beta, t_max=10)