from pathlib import Path
import numpy as np
import pandas as pd


def matrix(df, time_col, process_col, value_col):
    df = df.copy()
    df[time_col] = df[time_col].astype(int)
    df[process_col] = df[process_col].astype(str)

    names = sorted(df[process_col].unique())

    tab = df.pivot(index=time_col, columns=process_col, values=value_col)
    tab = tab.reindex(columns=names).sort_index().fillna(0)

    times = np.arange(tab.index.min(), tab.index.max() + 1)
    tab = tab.reindex(times, fill_value=0)

    return tab.to_numpy(float), list(names), times


def design(X, p):
    n, d = X.shape

    Y = X[p:]
    Z = []

    for k in range(1, p + 1):
        Z.append(X[p-k:n-k])

    Z = np.concatenate(Z, axis=1)
    Z = np.c_[np.ones(n - p), Z]

    return Y, Z


def fit(X, p):
    Y, Z = design(X, p)

    beta = np.linalg.lstsq(Z, Y, rcond=None)[0]

    d = X.shape[1]
    phi = np.zeros((p, d, d))

    for k in range(p):
        a = 1 + k * d
        phi[k] = beta[a:a+d].T

    fitted = Z @ beta

    return {
        "intercept": beta[0].copy(),
        "phi": phi,
        "fitted": fitted,
        "residuals": Y - fitted,
    }


def save_phi(phi, names, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []

    for k in range(phi.shape[0]):
        for i, target in enumerate(names):
            for j, source in enumerate(names):
                rows.append({
                    "lag": k + 1,
                    "target_process": target,
                    "source_process": source,
                    "phi": float(phi[k, i, j]),
                })

    pd.DataFrame(rows).to_csv(out / "phi_estimates.csv", index=False)


def save_npz(res, path, delta=12):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    p = res["phi"].shape[0]

    np.savez(
        path,
        t=delta * np.arange(1, p + 1),
        phi=res["phi"],
        proc_names=np.array(res["process_names"], dtype=object),
        delta=delta,
        intercept=res["intercept"],
    )


def estimate_discrete_hawkes_from_df(
    df,
    support,
    time_col="time",
    process_col="process",
    value_col="count",
    output_dir=None,
):
    X, names, times = matrix(df, time_col, process_col, value_col)
    res_fit = fit(X, support)

    res = {
        "X": X,
        "times": times,
        "process_names": names,
        "intercept": res_fit["intercept"],
        "phi": res_fit["phi"],
        "fitted": res_fit["fitted"],
        "residuals": res_fit["residuals"],
    }

    if output_dir is not None:
        save_phi(res["phi"], names, output_dir)
        save_npz(res, Path(output_dir) / "discrete_hawkes_fit.npz")

    return res