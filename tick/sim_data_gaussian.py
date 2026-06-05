import numpy as np
from tick.hawkes import SimuHawkesExpKernels
from tick.plot import plot_hawkes_kernels
import pandas as pd
from tick.base import TimeFunction
from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc, HawkesKernel0

D = 2

def gaussian_bump_kernel(amplitude=0.25, center=1.5, width=0.35,
                         support=4.0, n_points=400):
    t = np.linspace(0.0, support, n_points)

    y = amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)
    y[0] = 0.0

    tf = TimeFunction([t, y], inter_mode=TimeFunction.InterLinear)
    return HawkesKernelTimeFunc(tf)


def make_2d_bump_hawkes(end_time=5000, seed=123):

    kernels = np.empty((2, 2), dtype=object)

    kernels[0, 0] = gaussian_bump_kernel(0.18, 1.0, 0.25, 3.0)
    kernels[0, 1] = gaussian_bump_kernel(0.10, 2.0, 0.40, 4.5)
    kernels[1, 0] = gaussian_bump_kernel(0.08, 1.4, 0.30, 3.5)
    kernels[1, 1] = gaussian_bump_kernel(0.16, 1.8, 0.50, 5.0)

    mu = np.array([0.35, 0.30])

    hawkes = SimuHawkes(
        kernels=kernels,
        baseline=mu,
        end_time=end_time,
        seed=seed
    )

    return hawkes, kernels


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


    
def plot_custom_kernels(kernels, t_max=6, n=400):
    import matplotlib.pyplot as plt
    import numpy as np

    M = kernels.shape[0]
    t = np.linspace(0, t_max, n)

    fig, axes = plt.subplots(M, M, figsize=(3*M, 2.5*M),
                             sharex=True, sharey=True)

    if M == 1:
        axes = np.array([[axes]])

    for i in range(M):
        for j in range(M):
            kernel = kernels[i, j]
            y = np.array([kernel.get_value(x) for x in t])

            axes[i, j].plot(t, y)
            axes[i, j].set_title(f"{i} → {j}")

    fig.suptitle("Custom Hawkes kernels")
    plt.tight_layout()
    plt.show()
    
hawkes, kernels = make_2d_bump_hawkes(end_time=1e7)


plot_custom_kernels(kernels)

hawkes.simulate()

timestamps = hawkes.timestamps
print([len(ts) for ts in timestamps])
print("Premiers events dim0:", timestamps[0][:5])
plot_counts(timestamps)

dfs = []
for i in range(D):
    dfs.append(pd.DataFrame({
        "time": timestamps[i],
        "process": f"P{i+1}"
    }))
df = pd.concat([df for df in dfs], ignore_index=True).sort_values("time")
df.to_csv("data/d=2_gaussian_big.csv")
