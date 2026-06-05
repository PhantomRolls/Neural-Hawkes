import numpy as np
import pandas as pd


EPS = 1e-12


def aggregate_time_kernel(phi_marked, mark_probs=None):
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


def compute_mark_function(phi_marked, t, mark_probs=None):
    _, D, _, M = phi_marked.shape

    if mark_probs is None:
        mark_probs = np.ones((D, M)) / M
    else:
        mark_probs = np.asarray(mark_probs, dtype=float)
        if mark_probs.ndim == 1:
            mark_probs = np.tile(mark_probs, (D, 1))

    norms = np.trapezoid(phi_marked, x=t, axis=0)  # (D, D, M)
    f = np.zeros_like(norms)

    for i in range(D):
        for j in range(D):
            denom = np.sum(mark_probs[j] * norms[i, j])
            if denom > 0:
                f[i, j] = norms[i, j] / denom

    return f


def branching_matrix(phi, t):
    """
    phi shape (n_t, D, D)
    Retourne A_ij = int phi_ij(t) dt
    """
    return np.trapezoid(phi, x=t, axis=0)


def spectral_radius(A):
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def kernel_metrics(phi_est, phi_true, t):
    """
    Métriques globales sur phi(t), shape (n_t, D, D)
    """
    err = phi_est - phi_true

    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))

    l2_true = np.sqrt(np.trapezoid(phi_true**2, x=t, axis=0))
    l2_err = np.sqrt(np.trapezoid(err**2, x=t, axis=0))

    l1_true = np.trapezoid(np.abs(phi_true), x=t, axis=0)
    l1_err = np.trapezoid(np.abs(err), x=t, axis=0)

    rel_l2 = l2_err / (l2_true + EPS)
    rel_l1 = l1_err / (l1_true + EPS)

    max_abs = np.max(np.abs(err))
    max_rel = np.max(np.abs(err) / (np.abs(phi_true) + EPS))

    # Similarité de forme, utile pour savoir si la forme temporelle est bien capturée
    flat_est = phi_est.reshape(phi_est.shape[0], -1)
    flat_true = phi_true.reshape(phi_true.shape[0], -1)

    cosine_num = np.sum(flat_est * flat_true, axis=0)
    cosine_den = np.sqrt(np.sum(flat_est**2, axis=0)) * np.sqrt(np.sum(flat_true**2, axis=0))
    cosine = cosine_num / (cosine_den + EPS)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "relative_L2_mean": float(np.mean(rel_l2)),
        "relative_L2_max": float(np.max(rel_l2)),
        "relative_L1_mean": float(np.mean(rel_l1)),
        "relative_L1_max": float(np.max(rel_l1)),
        "max_absolute_error": float(max_abs),
        "max_pointwise_relative_error": float(max_rel),
        "cosine_similarity_mean": float(np.mean(cosine)),
    }


def branching_metrics(phi_est, phi_true, t):
    A_est = branching_matrix(phi_est, t)
    A_true = branching_matrix(phi_true, t)

    err = A_est - A_true

    rel = np.abs(err) / (np.abs(A_true) + EPS)

    return {
        "branching_relative_error_mean": float(np.mean(rel)),
        "branching_relative_error_max": float(np.max(rel)),
        "branching_absolute_error_mean": float(np.mean(np.abs(err))),
        "spectral_radius_true": spectral_radius(A_true),
        "spectral_radius_est": spectral_radius(A_est),
        "spectral_radius_abs_error": abs(spectral_radius(A_est) - spectral_radius(A_true)),
        "branching_true": A_true,
        "branching_est": A_est,
    }


def pairwise_table(phi_est, phi_true, t):
    """
    Table D x D : une ligne par noyau phi^{ij}
    """
    D = phi_est.shape[1]
    rows = []

    for i in range(D):
        for j in range(D):
            e = phi_est[:, i, j]
            q = phi_true[:, i, j]
            err = e - q

            l2_err = np.sqrt(np.trapezoid(err**2, x=t))
            l2_true = np.sqrt(np.trapezoid(q**2, x=t))

            l1_err = np.trapezoid(np.abs(err), x=t)
            l1_true = np.trapezoid(np.abs(q), x=t)

            mass_est = np.trapezoid(e, x=t)
            mass_true = np.trapezoid(q, x=t)

            rows.append({
                "i": i + 1,
                "j": j + 1,
                "RMSE": np.sqrt(np.mean(err**2)),
                "MAE": np.mean(np.abs(err)),
                "relative_L2": l2_err / (l2_true + EPS),
                "relative_L1": l1_err / (l1_true + EPS),
                "branching_true": mass_true,
                "branching_est": mass_est,
                "branching_relative_error": abs(mass_est - mass_true) / (abs(mass_true) + EPS),
                "max_abs_error": np.max(np.abs(err)),
            })

    return pd.DataFrame(rows)


def evaluate_unmarked(est_path, true_path, out_prefix="metrics_unmarked"):
    est = np.load(est_path, allow_pickle=True)
    true = np.load(true_path, allow_pickle=True)

    t = est["t"]
    phi_est = est["phi"]

    if phi_est.ndim == 4:
        phi_est = phi_est[..., 0]

    phi_true = true["phi_true"]

    global_metrics = {}
    global_metrics.update(kernel_metrics(phi_est, phi_true, t))
    global_metrics.update({
        k: v for k, v in branching_metrics(phi_est, phi_true, t).items()
        if not isinstance(v, np.ndarray)
    })

    df_global = pd.DataFrame([global_metrics])
    df_pairwise = pairwise_table(phi_est, phi_true, t)

    df_global.to_csv(f"{out_prefix}_global.csv", index=False)
    df_pairwise.to_csv(f"{out_prefix}_pairwise.csv", index=False)

    print("\n=== Global metrics ===")
    print(df_global.T)

    print("\n=== Pairwise metrics ===")
    print(df_pairwise)

    return df_global, df_pairwise


def evaluate_marked(est_path, true_path, out_prefix="metrics_marked"):
    est = np.load(est_path, allow_pickle=True)
    true = np.load(true_path, allow_pickle=True)

    t = est["t"]
    phi_est = est["phi"]

    mark_probs = est["mark_probs"] if "mark_probs" in est else (
        true["mark_probs"] if "mark_probs" in true else None
    )

    # Comparaison principale : noyaux agrégés
    phi_est_agg = aggregate_time_kernel(phi_est, mark_probs)

    if "phi_true_agg" in true:
        phi_true_agg = true["phi_true_agg"]
    else:
        phi_true_agg = aggregate_time_kernel(true["phi_true"], mark_probs)

    global_metrics = {}
    global_metrics.update(kernel_metrics(phi_est_agg, phi_true_agg, t))
    global_metrics.update({
        k: v for k, v in branching_metrics(phi_est_agg, phi_true_agg, t).items()
        if not isinstance(v, np.ndarray)
    })

    df_global = pd.DataFrame([global_metrics])
    df_pairwise = pairwise_table(phi_est_agg, phi_true_agg, t)

    df_global.to_csv(f"{out_prefix}_global.csv", index=False)
    df_pairwise.to_csv(f"{out_prefix}_pairwise_aggregated_kernels.csv", index=False)

    # Comparaison secondaire : fonctions de marques
    if "phi_true" in true:
        f_est = compute_mark_function(phi_est, t, mark_probs)

        if "f_true" in true:
            f_true = true["f_true"]
        else:
            f_true = compute_mark_function(true["phi_true"], t, mark_probs)

        f_err = f_est - f_true

        mark_metrics = {
            "mark_function_RMSE": float(np.sqrt(np.mean(f_err**2))),
            "mark_function_MAE": float(np.mean(np.abs(f_err))),
            "mark_function_relative_L2": float(
                np.linalg.norm(f_err.ravel()) / (np.linalg.norm(f_true.ravel()) + EPS)
            ),
            "mark_function_max_abs_error": float(np.max(np.abs(f_err))),
        }

        df_marks = pd.DataFrame([mark_metrics])
        df_marks.to_csv(f"{out_prefix}_mark_functions.csv", index=False)

        print("\n=== Mark function metrics ===")
        print(df_marks.T)

    print("\n=== Global metrics on aggregated kernels ===")
    print(df_global.T)

    print("\n=== Pairwise metrics on aggregated kernels ===")
    print(df_pairwise)

    return df_global, df_pairwise


if __name__ == "__main__":
    # Non marqué
    evaluate_unmarked(
        est_path="results/D3.npz",
        true_path="data/simulation/D3.npz",
        out_prefix=None,
    )

    # Marqué
    evaluate_marked(
        est_path="results/M5.npz",
        true_path="data/simulation/M5.npz",
        out_prefix=None,
    )