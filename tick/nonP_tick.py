import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tick.hawkes import HawkesEM
from data.real.utils import load_real_data


def prepare_swap_timestamps():
    df = load_real_data(["USDC/WETH 0.05"], file="30days")
    df = df[df["event_type"] == "swap"].copy()
    df["process"] = np.where(df["SwapX2Y"], "X2Y", "Y2X")
    processes = ["X2Y", "Y2X"]

    timestamps = []
    for p in processes:
        arr = df.loc[df["process"] == p, "time"].to_numpy(dtype=float)
        arr = np.sort(arr)
        timestamps.append(arr)
    non_empty = [arr for arr in timestamps if len(arr) > 0]
    if not non_empty:
        raise ValueError("Aucun timestamp trouvé après filtrage.")

    t0 = min(arr[0] for arr in non_empty)
    timestamps = [arr - t0 for arr in timestamps]

    return timestamps, processes


def fit_hawkes_em(
    timestamps,
    kernel_support=500.0,
    kernel_size=80,
    tol=1e-5,
    max_iter=200,
    n_threads=1,
    verbose=True,
):
    """
    Estimation non paramétrique par EM.
    Les noyaux sont estimés comme fonctions par morceaux constantes
    sur une grille régulière de [0, kernel_support].
    """
    learner = HawkesEM(
        kernel_support=kernel_support,
        kernel_size=kernel_size,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
        n_threads=n_threads,
        print_every=10,
        record_every=10,
    )
    learner.fit(timestamps)
    return learner


def get_kernel_grid(learner):
    """
    Construit la grille temporelle associée aux noyaux estimés.
    Pour HawkesEM, learner.kernel a shape (d, d, kernel_size).
    """
    kernel = learner.kernel
    kernel_size = kernel.shape[2]
    t = np.linspace(0.0, learner.kernel_support, kernel_size, endpoint=False)
    dt = learner.kernel_support / kernel_size
    return t, dt


def compute_branching_matrix(learner):
    """
    Approximation des normes L1 des noyaux.
    Comme le noyau est par morceaux constant :
        ||phi_ij|| ~= sum_k phi_ij[k] * dt
    """
    _, dt = get_kernel_grid(learner)
    branching = learner.kernel.sum(axis=2) * dt
    spectral_radius = np.max(np.abs(np.linalg.eigvals(branching)))
    return branching, spectral_radius


def plot_kernels_em(learner, process_names=None):
    kernel = learner.kernel  # shape (d, d, K)
    d = kernel.shape[0]
    t, dt = get_kernel_grid(learner)

    fig, axes = plt.subplots(d, d, figsize=(4 * d, 3 * d), squeeze=False)

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]

            # step plot pour refléter le côté "piecewise constant"
            ax.step(t, kernel[i, j], where="post")
            src = process_names[j] if process_names is not None else str(j)
            dst = process_names[i] if process_names is not None else str(i)
            ax.set_title(f"$\\phi_{{{dst} \\leftarrow {src}}}(t)$")
            ax.set_xlabel("t")
            ax.set_ylabel("phi")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    timestamps, process_names = prepare_swap_timestamps()

    learner = fit_hawkes_em(
        timestamps=timestamps,
        kernel_support=500.0,  
        kernel_size=100,      
        tol=1e-5,
        max_iter=200,
        n_threads=1,
        verbose=True,
    )

    branching, spectral_radius = compute_branching_matrix(learner)

    print("Baseline mu:")
    print(learner.baseline)
    print("\nBranching matrix (approx. L1 des noyaux):")
    print(branching)
    print("\nSpectral radius:")
    print(spectral_radius)

    plot_kernels_em(learner, process_names=process_names)