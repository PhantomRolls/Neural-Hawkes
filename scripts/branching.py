import numpy as np


def aggregate_time_kernel(phi_marked, mark_probs=None):
    """
    phi_marked : shape (n_t, D, D, M)
    mark_probs : shape (M,) ou (D, M)
    """
    n_t, D, _, M = phi_marked.shape

    if mark_probs is None:
        mark_probs = np.ones((D, M)) / M
    else:
        mark_probs = np.asarray(mark_probs, dtype=float)
        if mark_probs.ndim == 1:
            mark_probs = np.tile(mark_probs, (D, 1))

    phi_agg = np.zeros((n_t, D, D))

    for i in range(D):
        for j in range(D):
            phi_agg[:, i, j] = np.sum(
                phi_marked[:, i, j, :] * mark_probs[j][None, :],
                axis=1,
            )

    return phi_agg


def compute_branching_from_npz(path, phi_key="phi", t_key="t"):
    data = np.load(path, allow_pickle=True)

    t = data[t_key]
    phi = data[phi_key]

    # Cas où phi est marqué : shape (n_t, D, D, M)
    if phi.ndim == 4:
        mark_probs = data["mark_probs"] if "mark_probs" in data else None
        phi = aggregate_time_kernel(phi, mark_probs)

    # Cas où phi a une dernière dimension inutile : shape (n_t, D, D, 1)
    if phi.ndim == 4 and phi.shape[-1] == 1:
        phi = phi[..., 0]

    if phi.ndim != 3:
        raise ValueError(f"Shape inattendue pour phi : {phi.shape}")

    # A_ij = int phi_ij(t) dt
    A = np.trapezoid(phi, x=t, axis=0)

    eigvals = np.linalg.eigvals(A)
    rho = np.max(np.abs(eigvals))

    return A, rho, eigvals


if __name__ == "__main__":
    path = "results/D2_USDC_WETH_005.npz"  # à modifier

    A, rho, eigvals = compute_branching_from_npz(path)

    print("\nMatrice de branching A_ij = ∫ phi_ij(t) dt :\n")
    print(A)

    print("\nBranching ratios par interaction :")
    D = A.shape[0]
    for i in range(D):
        for j in range(D):
            print(f"A[{i+1},{j+1}] = {A[i, j]:.6f}")

    print("\nValeurs propres de A :")
    print(eigvals)

    print(f"\nRayon spectral rho(A) = {rho:.6f}")

    if rho < 1:
        print("Processus stable : rho(A) < 1")
    else:
        print("Processus instable : rho(A) >= 1")