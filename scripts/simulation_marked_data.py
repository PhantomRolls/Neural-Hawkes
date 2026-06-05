import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class KernelSpec:
    D: int
    M: int
    phi: callable         
    mark_probs: np.ndarray
    t_max: float = 10.0
    n_grid: int = 5000


def precompute_kernels(spec: KernelSpec):
    t_grid = np.linspace(0.0, spec.t_max, spec.n_grid)

    norms = np.zeros((spec.D, spec.D, spec.M))
    cdfs = {}

    for i in range(spec.D):
        for j in range(spec.D):
            for m in range(spec.M):
                vals = spec.phi(i, j, t_grid, m)
                vals = np.maximum(vals, 0.0)

                norm = np.trapezoid(vals, t_grid)
                norms[i, j, m] = norm

                if norm > 0:
                    pdf = vals / norm
                    cdf = np.zeros_like(t_grid)
                    cdf[1:] = np.cumsum(
                        0.5 * (pdf[1:] + pdf[:-1]) * np.diff(t_grid)
                    )
                    cdf /= cdf[-1]
                    cdfs[(i, j, m)] = cdf
                else:
                    cdfs[(i, j, m)] = None

    B = np.sum(norms * spec.mark_probs[None, None, :], axis=2)
    rho = max(abs(np.linalg.eigvals(B)))

    return t_grid, norms, cdfs, B, rho

def sample_delay(rng, t_grid, cdf):
    u = rng.random()
    return np.interp(u, cdf, t_grid)

def simulate_cluster_hawkes(lambda0, spec: KernelSpec, T: float, seed=None):
    """
    Simulation par représentation cluster.

    Chaque événement de type j et marque m génère des descendants de type i
    avec moyenne ∫ phi_ij(t,m) dt.
    """
    rng = np.random.default_rng(seed)

    lambda0 = np.asarray(lambda0, dtype=float)
    t_grid, norms, cdfs, B, rho = precompute_kernels(spec)

    print("Branching matrix:")
    print(B)
    print(f"Spectral radius = {rho:.4f}")

    if rho >= 1:
        raise ValueError("Processus instable : rayon spectral >= 1.")

    events = []

    for i in range(spec.D):
        n_imm = rng.poisson(lambda0[i] * T)
        times = rng.uniform(0, T, size=n_imm)

        for t in times:
            m = rng.choice(spec.M, p=spec.mark_probs)
            events.append((t, i, m))

    queue = list(events)

    while queue:
        t_parent, j, m_parent = queue.pop()

        for i in range(spec.D):
            mean_offspring = norms[i, j, m_parent]
            n_child = rng.poisson(mean_offspring)

            cdf = cdfs[(i, j, m_parent)]
            if cdf is None:
                continue

            for _ in range(n_child):
                delay = sample_delay(rng, t_grid, cdf)
                t_child = t_parent + delay

                if t_child <= T:
                    m_child = rng.choice(spec.M, p=spec.mark_probs)
                    child = (t_child, i, m_child)
                    events.append(child)
                    queue.append(child)

    df = pd.DataFrame(events, columns=["time", "process", "mark"])
    df = df.sort_values("time").reset_index(drop=True)
    return df, B, rho


def phi_non_marked(i, j, t):
    if i == 0 and j == 0:
        return 0.45 * np.exp(-3.0 * t)
    if i == 1 and j == 1:
        return 0.35 * np.exp(-2.0 * t)
    if i == 2 and j == 2:
        return 0.40 * np.exp(-4.0 * t)
    # excitation retardée
    if i == 0 and j == 1:
        return 0.30 * np.exp(-2.5 * (t - 0.4)) * (t >= 0.4)
    # gaussien retardé
    if i == 1 and j == 2:
        sigma = 0.15
        return 0.25 * np.exp(-0.5 * ((t - 0.6) / sigma) ** 2)
    # bi-exponentiel
    if i == 2 and j == 0:
        return 0.20 * np.exp(-1.0 * t) + 0.12 * np.exp(-6.0 * t)
    return np.zeros_like(t)


spec_non_marked = KernelSpec(
    D=3,
    M=1,
    phi=phi_non_marked,
    mark_probs=np.array([1.0]),
    t_max=8.0,
)
df_non_marked, B, rho = simulate_cluster_hawkes(
    lambda0=np.array([0.5, 0.45, 0.4]),
    spec=spec_non_marked,
    T=1_000_000,
    seed=42,
)
df_non_marked.to_parquet("data/simulation/D3.parquet", index=False)

# ============================================================
# Save true kernels (unmarked)
# ============================================================

data_path = f"data/simulation/D3.parquet"
df = pd.read_parquet(data_path)
from src.neural_hawkes.preprocessing import estimate_T_kernel
T_kernel = estimate_T_kernel(df, q=0.95)
print("T_kernel : ", T_kernel)
t_grid = np.linspace(1e-4, T_kernel, 79)
phi_true = np.zeros((len(t_grid), spec_non_marked.D, spec_non_marked.D))

for i in range(spec_non_marked.D):
    for j in range(spec_non_marked.D):
        # m = 0 car M = 1
        phi_true[:, i, j] = phi_non_marked(i, j, t_grid, 0)

np.savez(
    "data/simulation/D3.npz",
    t=t_grid,
    phi_true=phi_true,
    proc_names=np.array(["P0", "P1", "P2"]),
)
print("True unmarked kernels saved.")
print("True kernels saved.")





# M = 5
# alpha = np.array([
#     [0.55, 0.28],
#     [0.18, 0.42],
# ])
# beta = np.array([
#     [1.4, 1.8],
#     [1.6, 1.3],
# ])
# mark_probs = np.ones(M) / M
# f = np.zeros((2, 2, M))
# # forte dépendance en marque
# f[0, 0] = [0.35, 0.60, 0.90, 1.25, 1.70]
# # quasi linéaire modérée
# f[1, 1] = [0.50, 0.75, 1.00, 1.25, 1.50]
# # faible effet de marque
# f[0, 1] = [0.85, 0.92, 1.00, 1.08, 1.15]
# # intermédiaire
# f[1, 0] = [0.60, 0.78, 1.00, 1.22, 1.45]
# for i in range(2):
#     for j in range(2):
#         f[i, j] = f[i, j] / np.sum(mark_probs * f[i, j])
# def phi_marked(i, j, t, m):
#     return (
#         alpha[i, j]
#         * f[i, j, m]
#         * np.exp(-beta[i, j] * t)
#     )
# spec_marked = KernelSpec(
#     D=2,
#     M=M,
#     phi=phi_marked,
#     mark_probs=mark_probs,
#     t_max=8.0,
# )
# df_marked, B, rho = simulate_cluster_hawkes(
#     lambda0=np.array([0.7, 0.6]),
#     spec=spec_marked,
#     T=1_000_000,
#     seed=123,
# )
# df_marked.to_parquet(f"data/simulation/M{M}.parquet", index=False)





# # ============================================================
# # Save true kernels (marked)
# # ============================================================

# data_path = f"data/simulation/M{M}.parquet"
# df = pd.read_parquet(data_path)
# from src.neural_hawkes.preprocessing import estimate_T_kernel
# T_kernel = estimate_T_kernel(df, q=0.95)
# t_grid = np.linspace(1e-4, T_kernel, 79)

# phi_true = np.zeros((
#     len(t_grid),
#     spec_marked.D,
#     spec_marked.D,
#     spec_marked.M,
# ))

# for i in range(spec_marked.D):
#     for j in range(spec_marked.D):
#         for m in range(spec_marked.M):

#             phi_true[:, i, j, m] = phi_marked(
#                 i,
#                 j,
#                 t_grid,
#                 m,
#             )

# # ============================================================
# # Aggregated kernels varphi^{ij}(t)
# # ============================================================

# phi_true_agg = np.zeros((
#     len(t_grid),
#     spec_marked.D,
#     spec_marked.D,
# ))

# for i in range(spec_marked.D):
#     for j in range(spec_marked.D):

#         phi_true_agg[:, i, j] = np.sum(
#             phi_true[:, i, j, :] * mark_probs[None, :],
#             axis=1,
#         )

# # ============================================================
# # Mark functions f^{ij}(x)
# # ============================================================

# norms = np.trapezoid(
#     phi_true,
#     x=t_grid,
#     axis=0,
# )

# f_true = np.zeros_like(norms)

# for i in range(spec_marked.D):
#     for j in range(spec_marked.D):

#         denom = np.sum(
#             mark_probs * norms[i, j]
#         )

#         if denom > 0:
#             f_true[i, j] = norms[i, j] / denom

# # ============================================================
# # Save
# # ============================================================

# np.savez(
#     f"data/simulation/M{M}.npz",

#     t=t_grid,

#     phi_true=phi_true,

#     phi_true_agg=phi_true_agg,

#     f_true=f_true,

#     mark_probs=mark_probs,

#     proc_names=np.array(["P0", "P1"]),

#     mark_names=np.array([
#         "small",
#         "medium",
#         "large",
#     ]),
# )

# print("True marked kernels saved.")