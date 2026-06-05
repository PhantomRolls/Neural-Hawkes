import numpy as np
import pandas as pd
import torch

def events_from_dataframe_marked(
    df,
    process_col="process",
    time_col="time",
    mark_col="mark",
):
    df = df[[process_col, time_col, mark_col]].dropna().copy()
    df[time_col] = df[time_col].astype(float)
    df[mark_col] = df[mark_col].astype(int)

    df = df.sort_values(time_col).reset_index(drop=True)

    t0 = df[time_col].min()
    df[time_col] = df[time_col] - t0

    proc_names = sorted(df[process_col].unique())
    mark_names = sorted(df[mark_col].unique())

    D = len(proc_names)
    M = len(mark_names)

    proc_to_id = {p: i for i, p in enumerate(proc_names)}
    mark_to_id = {m: r for r, m in enumerate(mark_names)}

    events = []
    marks = []
    counts = []
    mark_probs = torch.zeros(D, M, dtype=torch.float64)

    for p in proc_names:
        sub = df[df[process_col] == p]

        times_p = sub[time_col].to_numpy(dtype=np.float64)
        marks_p = sub[mark_col].map(mark_to_id).to_numpy(dtype=np.int64)

        events.append(torch.tensor(times_p, dtype=torch.float64))
        marks.append(torch.tensor(marks_p, dtype=torch.long))
        counts.append(len(times_p))

        i = proc_to_id[p]
        for m in range(M):
            mark_probs[i, m] = np.mean(marks_p == m)

    T_obs = float(df[time_col].max())
    lambdas = torch.tensor(counts, dtype=torch.float64) / T_obs

    return events, marks, mark_probs, lambdas, T_obs, proc_names, mark_names

import numpy as np


def estimate_T_kernel(
    df,
    time_col="time",
    process_col="process",
    q=0.95,
    min_T=None,
    max_T=None,
):
    T_values = []

    for _, group in df.groupby(process_col):

        times = np.sort(group[time_col].to_numpy(dtype=float))

        if len(times) < 2:
            continue

        dt = np.diff(times)
        dt = dt[dt > 0]

        if len(dt) == 0:
            continue

        T_values.append(np.quantile(dt, q))
    T_kernel = max(T_values)

    if min_T is not None:
        T_kernel = max(T_kernel, min_T)
    if max_T is not None:
        T_kernel = min(T_kernel, max_T)
    return T_kernel

def _count_in_windows(event_times_i, left_edges, right_edges):
    left_idx = np.searchsorted(event_times_i, left_edges, side="left")
    right_idx = np.searchsorted(event_times_i, right_edges, side="left")
    return right_idx - left_idx


def estimate_G_tensor_marked(
    events,
    marks,
    g_edges,
    lambdas,
    T_obs,
    n_marks,
):
    D = len(events)
    Q = g_edges.numel() - 1

    G = torch.zeros(D, D, n_marks, Q, dtype=torch.float64)
    events_np = [ev.detach().cpu().numpy().astype(np.float64) for ev in events]
    marks_np = [mk.detach().cpu().numpy().astype(np.int64) for mk in marks]
    g_edges_np = g_edges.detach().cpu().numpy().astype(np.float64)
    lambdas_np = lambdas.detach().cpu().numpy().astype(np.float64)

    for i in range(D):
        times_i = events_np[i]

        for j in range(D):
            times_j_all = events_np[j]
            marks_j_all = marks_np[j]

            for m in range(n_marks):
                times_j = times_j_all[marks_j_all == m]

                if len(times_j) == 0:
                    continue

                for q in range(Q):
                    left = g_edges_np[q]
                    right = g_edges_np[q + 1]
                    width = right - left

                    valid = times_j + right <= T_obs
                    anchor_times = times_j[valid]

                    if len(anchor_times) == 0:
                        continue
                    left_edges = anchor_times + left
                    right_edges = anchor_times + right

                    counts = _count_in_windows(times_i, left_edges, right_edges)
                    conditional_rate = counts.mean() / width

                    G[i, j, m, q] = float(conditional_rate - lambdas_np[i])
    return G


def plot_G_marked(G_tensor, g_grid, mark_names=None):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": "#dfe8f5",
        "axes.facecolor": "#c8d6eb",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.1,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "legend.frameon": True,
        "legend.facecolor": "#dfe8f5",
        "legend.edgecolor": "#666666",
    })

    if hasattr(g_grid, "detach"):
        g_np = g_grid.detach().cpu().numpy()
    else:
        g_np = np.asarray(g_grid)

    if hasattr(G_tensor, "detach"):
        G_np = G_tensor.detach().cpu().numpy()
    else:
        G_np = np.asarray(G_tensor)
    D, _, M, _ = G_np.shape

    if mark_names is None:
        mark_names = [str(m + 1) for m in range(M)]

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, M))

    fig, axes = plt.subplots(
        1,
        D * D,
        figsize=(3.0 * D * D, 2.45),
        squeeze=False,
    )

    pairs = [(i, j) for i in range(D) for j in range(D)]
    for k, (i, j) in enumerate(pairs):
        ax = axes[0, k]

        for m in range(M):
            ax.plot(g_np, G_np[i, j, m], lw=1.4, color=colors[m],
                    label=str(mark_names[m]))

        ax.set_xlabel(r"$t$")
        ax.set_ylabel(rf"$G^{{{i+1}{j+1}}}(t,x)$")

        ax.tick_params(direction="out", width=0.9, length=3.5)
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=M,
        title="Mark",
        frameon=True,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(
        "figures/G.pdf",
        format="pdf",
        bbox_inches="tight"
    )
    plt.show()


def interp_G_pair_marked(G_tensor, t_centers, a, b, m, x):
    x_flat = x.reshape(-1)
    device = x_flat.device

    tg = t_centers.to(device)
    G_abm = G_tensor[a, b, m].to(device)

    out = torch.zeros_like(x_flat, dtype=G_abm.dtype, device=device)

    in_mask = (x_flat >= tg[0]) & (x_flat <= tg[-1])

    if in_mask.any():
        x_in = x_flat[in_mask]

        idx = torch.searchsorted(tg, x_in).clamp(1, len(tg) - 1)

        x0 = tg[idx - 1]
        x1 = tg[idx]
        g0 = G_abm[idx - 1]
        g1 = G_abm[idx]

        w = (x_in - x0) / (x1 - x0 + 1e-12)
        out[in_mask] = g0 + w * (g1 - g0)

        left_exact = x_in == tg[0]
        if left_exact.any():
            out[in_mask.nonzero(as_tuple=True)[0][left_exact]] = G_abm[0]

    return out.reshape(x.shape)


def H_from_G_marked(G_tensor, t_centers, lambdas, k, j, tau, m_x, m_z):
    device = tau.device
    out = torch.zeros_like(tau, device=device)
    pos_mask = tau > 0
    neg_mask = tau < 0

    if pos_mask.any():
        out[pos_mask] = interp_G_pair_marked(
            G_tensor, t_centers, k, j, m_x, tau[pos_mask])
    if neg_mask.any():
        ratio = lambdas[k] / (lambdas[j] + 1e-12)
        out[neg_mask] = ratio.to(device) * interp_G_pair_marked(
            G_tensor, t_centers, j, k, m_z, -tau[neg_mask])
    return out