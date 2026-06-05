import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def set_article_style():
    mpl.rcParams.update({

        # fond global très clair
        "figure.facecolor": "#dfe8f5",

        # fond des sous-graphes légèrement plus foncé
        "axes.facecolor": "#c8d6eb",

        # bordures discrètes
        "axes.edgecolor": "#000000",
        "axes.linewidth": 1.1,

        # police
        "font.family": "serif",
        "mathtext.fontset": "cm",

        # ticks
        "xtick.direction": "in",
        "ytick.direction": "in",

        # ticks fins
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,

        # légende
        "legend.frameon": True,
        "legend.facecolor": "#dfe8f5",
        "legend.edgecolor": "#666666",

        # savefig cohérent
        "savefig.facecolor": "#dfe8f5",
    })


def _as_names(names, D):
    if names is None:
        return [f"P{i}" for i in range(D)]
    return [str(x) for x in names]


# ============================================================
# Core computations
# ============================================================

def aggregate_time_kernel(phi_marked, mark_probs=None):
    """
    Agrège les noyaux marqués sur les marques.

    phi_marked : shape (n_t, D, D, M)
    mark_probs : shape (M,) ou (D, M)

    Retour :
        phi_agg : shape (n_t, D, D)
        phi_agg[t,i,j] = sum_m p_j(m) phi[t,i,j,m]
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


def compute_mark_function(phi_marked, t, mark_probs=None):
    """
    Calcule f^{ij}(m) avec la normalisation :
        E[f^{ij}(M)] = 1.

    phi_marked : shape (n_t, D, D, M)
    retour : shape (D, D, M)
    """
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


def rmse(phi_est, phi_true):
    return np.sqrt(np.mean((phi_est - phi_true) ** 2))


# ============================================================
# Plotting functions
# ============================================================

def plot_kernel_grid(
    t,
    phi_est,
    phi_true=None,
    process_names=None,
    kernel_symbol=r"\phi",
    title=None,
    true_label="True",
    est_label="Predicted",
    save_path=None,
    shared_ylim=True,
):
    import os

    set_article_style()

    _, D, _ = phi_est.shape
    process_names = _as_names(process_names, D)

    # =========================
    # bornes globales
    # =========================
    if shared_ylim:
        vals = [phi_est]

        if phi_true is not None:
            vals.append(phi_true)

        ymin = min(np.nanmin(v) for v in vals)
        ymax = max(np.nanmax(v) for v in vals)

        pad = 0.05 * (ymax - ymin + 1e-12)

        ymin -= pad
        ymax += pad

    # =========================
    # figure
    # =========================
    fig, axes = plt.subplots(
        D,
        D,
        figsize=(2.6 * D / 0.90 * 0.98, 2.8 * D),
        squeeze=False,
    )

    # =========================
    # plots
    # =========================
    for i in range(D):
        for j in range(D):

            ax = axes[i, j]

            if phi_true is not None:
                ax.plot(
                    t,
                    phi_true[:, i, j],
                    color="#5577cc",
                    lw=1.5,
                    label=true_label,
                )

            ax.plot(
                t,
                phi_est[:, i, j],
                "--",
                color="black",
                lw=1.2,
                label=est_label,
            )

            # =========================
            # ylim partagé
            # =========================
            if shared_ylim:
                ax.set_ylim(ymin, ymax)

            # =========================
            # labels
            # =========================
            if i == D - 1:
                ax.set_xlabel(r"$t$")
            else:
                ax.set_xticklabels([])

            ax.set_ylabel(
                rf"${kernel_symbol}^{{{i+1}{j+1}}}(t)$"
            )

            # =========================
            # ticks
            # =========================
            ax.tick_params(
                axis="x",
                which="both",
                bottom=(i == D - 1),
                top=False,
                labelbottom=(i == D - 1),
                direction="out",
                width=0.9,
                length=3.5,
            )

            ax.tick_params(
                axis="y",
                which="both",
                left=True,
                right=False,
                labelleft=True,
                direction="out",
                width=0.9,
                length=3.5,
            )

            # =========================
            # style
            # =========================
            ax.grid(False)

            for spine in ax.spines.values():
                spine.set_linewidth(1.1)

    # =========================
    # titre
    # =========================
    if title is not None:
        fig.suptitle(title)

    # =========================
    # légende
    # =========================
    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=True,
    )

    # =========================
    # layout
    # =========================

    fig.patch.set_facecolor("#d7e2f1")
    fig.subplots_adjust(
        left=0.13,
        right=0.90,
        bottom=0.14,
        top=0.98,
        wspace=0.35,
        hspace=0.18,
    )
    # =========================
    # save
    # =========================
    if save_path is not None:

        folder = os.path.dirname(save_path)

        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.subplots_adjust(right=0.90)
        plt.savefig(
            save_path,
            pad_inches=0.08,
        )

    plt.show()

def plot_kernel_grid_three(
    t,
    phi_1,
    phi_2,
    phi_3,
    process_names=None,
    kernel_symbol=r"\phi",
    title=None,
    labels=("True integrated", "Neural Hawkes integrated", "Discrete Hawkes"),
    save_path=None,
    shared_ylim=True,
):
    import os

    set_article_style()

    _, D, _ = phi_2.shape
    process_names = _as_names(process_names, D)

    if shared_ylim:
        vals = [phi_2, phi_3]
        if phi_1 is not None:
            vals.append(phi_1)
        ymin = min(np.nanmin(v) for v in vals)
        ymax = max(np.nanmax(v) for v in vals)
        pad = 0.05 * (ymax - ymin + 1e-12)
        ymin -= pad
        ymax += pad

    fig, axes = plt.subplots(
        D,
        D,
        figsize=(2.6 * D / 0.90 * 0.98, 2.8 * D),
        squeeze=False,
    )
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]

            # true
            if phi_1 is not None:
                ax.plot(t, phi_1[:, i, j], color="#5577cc", lw=1.6, label=labels[0])

            # neural
            ax.plot(t, phi_2[:, i, j], "--", color="black", lw=1.3, label=labels[1])

            # discrete
            print(t[1]-t[0], t)
            ax.step(
                t,
                phi_3[:, i, j],
                where="post",
                color="#9e02b3",
                lw=1.2,
                label=labels[2],
            )

            ax.plot(
                t,
                phi_3[:, i, j],
                "o",
                color="#9e02b3",
                ms=2.5,
            )


            if shared_ylim:
                ax.set_ylim(ymin, ymax)

            if i == D - 1:
                ax.set_xlabel(r"$t$ (s)")
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(rf"${kernel_symbol}^{{{i+1}{j+1}}}(t)$")
            else:
                ax.set_yticklabels([])

            ax.tick_params(
                axis="x",
                which="both",
                bottom=(i == D - 1),
                top=False,
                labelbottom=(i == D - 1),
                direction="out",
                width=0.9,
                length=3.5,
            )

            ax.tick_params(
                axis="y",
                which="both",
                left=(j == 0),
                right=False,
                labelleft=(j == 0),
                direction="out",
                width=0.9,
                length=3.5,
            )

            ax.grid(False)

            for spine in ax.spines.values():
                spine.set_linewidth(1.1)

    if title is not None:
        fig.suptitle(title)

    handles, labels_out = axes[0, 0].get_legend_handles_labels()
    # retire doublons
    by_label = dict(zip(labels_out, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=3 if phi_3 is not None else 2,
        frameon=True,
)

    fig.patch.set_facecolor("#d7e2f1")
    fig.subplots_adjust(
        left=0.13,
        right=0.90,
        bottom=0.15,
        top=0.98,
        wspace=0.35,
        hspace=0.18,
    )

    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(
            save_path,
            pad_inches=0.08,
        )

    plt.show()


def plot_mark_function_grid(
    f_est,
    f_true=None,
    process_names=None,
    mark_names=None,
    save_path=None,
    shared_ylim=True,
):
    """
    Plot grille D x D des fonctions de marque f^{ij}(x)

    Parameters
    ----------
    f_est : array shape (D, D, M)
    f_true : array shape (D, D, M) or None
    """

    import os

    set_article_style()

    D, _, M = f_est.shape
    process_names = _as_names(process_names, D)

    # =========================
    # marques
    # =========================
    if mark_names is None:
        marks = np.arange(1, M + 1)

    else:
        try:
            marks = np.asarray(mark_names, dtype=float)

        except ValueError:
            marks = np.arange(1, M + 1)

    # =========================
    # bornes globales
    # =========================
    if shared_ylim:

        vals = [f_est]

        if f_true is not None:
            vals.append(f_true)

        ymin = min(np.nanmin(v) for v in vals)
        ymax = max(np.nanmax(v) for v in vals)

        pad = 0.05 * (ymax - ymin + 1e-12)

        ymin -= pad
        ymax += pad

    # =========================
    # figure
    # =========================
    fig, axes = plt.subplots(
        D,
        D,
        figsize=(2.6 * D, 2.8 * D),
        squeeze=False,
    )

    # =========================
    # plots
    # =========================
    for i in range(D):
        for j in range(D):

            ax = axes[i, j]

            if f_true is not None:
                ax.plot(
                    marks,
                    f_true[i, j],
                    color="#5577cc",
                    lw=1.5,
                    label="True",
                )

            ax.plot(
                marks,
                f_est[i, j],
                "--",
                color="black",
                lw=1.2,
                label="Predicted",
            )

            # =========================
            # ylim partagé
            # =========================
            if shared_ylim:
                ax.set_ylim(ymin, ymax)

            # =========================
            # labels
            # =========================
            if i == D - 1:
                ax.set_xlabel(r"$x$")
            else:
                ax.set_xticklabels([])

            ax.set_ylabel(
                rf"$f^{{{i+1}{j+1}}}(x)$"
            )

            # =========================
            # ticks
            # =========================
            ax.tick_params(
                axis="x",
                which="both",
                bottom=(i == D - 1),
                top=False,
                labelbottom=(i == D - 1),
                direction="out",
                width=0.9,
                length=3.5,
            )

            ax.tick_params(
                axis="y",
                which="both",
                left=True,
                right=False,
                labelleft=True,
                direction="out",
                width=0.9,
                length=3.5,
            )

            # =========================
            # style
            # =========================
            ax.grid(False)

            for spine in ax.spines.values():
                spine.set_linewidth(1.1)

    # =========================
    # légende
    # =========================
    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=True,
    )

    # =========================
    # layout
    # =========================
    fig.patch.set_facecolor("#d7e2f1")
    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.15,
        top=0.98,
        wspace=0.45,
        hspace=0.18,
    )
    # =========================
    # save
    # =========================
    if save_path is not None:

        folder = os.path.dirname(save_path)

        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(
            save_path,
            pad_inches=0.08,
        )

    plt.show()

def plot_marked_article_style(
    t,
    phi_est,
    phi_true=None,
    phi_true_agg=None,
    f_true=None,
    mark_probs=None,
    process_names=None,
    mark_names=None,
    save_prefix=None,
):
    phi_est_agg = aggregate_time_kernel(phi_est, mark_probs)

    if phi_true_agg is None and phi_true is not None:
        phi_true_agg = aggregate_time_kernel(phi_true, mark_probs)

    save_path_time = None
    save_path_mark = None

    if save_prefix is not None:
        save_path_time = f"figures/{save_prefix}_time_kernels.pdf"
        save_path_mark = f"figures/{save_prefix}_mark_functions.pdf"

    plot_kernel_grid(
        t=t,
        phi_est=phi_est_agg,
        phi_true=phi_true_agg,
        process_names=process_names,
        save_path=save_path_time,
    )

    f_est = compute_mark_function(phi_est, t, mark_probs)

    plot_mark_function_grid(
        f_est=f_est,
        f_true=f_true,
        process_names=process_names,
        mark_names=mark_names,
        save_path=save_path_mark,
    )


def plot_phi_by_mark(
    t,
    phi_est,
    phi_true=None,
    i=0,
    j=0,
    mark_names=None,
    save_path="figures/phi_by_mark.pdf",
):
    set_article_style()

    _, D, _, M = phi_est.shape

    if mark_names is None:
        mark_names = [str(m) for m in range(M)]

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, M))

    fig, ax = plt.subplots(figsize=(6, 4))

    for m in range(M):

        if phi_true is not None:
            ax.plot(
                t,
                phi_true[:, i, j, m],
                color=colors[m],
                lw=1.8,
            )

        ax.plot(
            t,
            phi_est[:, i, j, m],
            "--",
            color=colors[m],
            lw=1.4,
            label=f"mark {mark_names[m]}",
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(rf"$\phi^{{{i+1}{j+1}}}(t,x)$")

    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )

    ax.grid(False)

    ax.legend(frameon=True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def aggregate_continuous_kernel_to_blocks(t, phi, delta, support, n_per_block=200):
    t = np.asarray(t, dtype=float)
    phi = np.asarray(phi, dtype=float)

    order = np.argsort(t)
    t = t[order]
    phi = phi[order]

    D = phi.shape[1]
    out = np.zeros((support, D, D))
    block_t = delta * (np.arange(support) + 0.5)

    for k in range(support):
        left = k * delta
        right = (k + 1) * delta
        grid = np.linspace(left, right, n_per_block)

        for i in range(D):
            for j in range(D):
                vals = np.interp(
                    grid,
                    t,
                    phi[:, i, j],
                    left=0.0,
                    right=0.0,
                )
                out[k, i, j] = np.trapezoid(vals, x=grid)

    return block_t, out


# ============================================================
# Run examples
# ============================================================

def run_unmarked(
    estimated_path="results/sim_unmarked_kernels.npz",
    true_path="data/simulation/true_non_marked_D3.npz",
    save_path=None,
):
    est = np.load(estimated_path, allow_pickle=True)
    true = np.load(true_path, allow_pickle=True)

    t = est["t"]
    phi_est = est["phi"]
    phi_true = true["phi_true"]
    phi_est = phi_est[..., 0]
    proc_names = est["proc_names"].tolist() if "proc_names" in est else None

    # print(f"RMSE unmarked = {rmse(phi_est, phi_true):.4e}")

    plot_kernel_grid(
        t=t,
        phi_est=phi_est,
        phi_true=phi_true,
        process_names=proc_names,
        save_path=save_path,
    )


def run_marked(
    kernel1_path=None,
    kernel2_path=None,
    save_prefix=None,
):
    est = np.load(kernel2_path, allow_pickle=True)

    t = est["t"]
    phi_est = est["phi"]

    proc_names = est["proc_names"].tolist() if "proc_names" in est else None
    mark_names = est["mark_names"].tolist() if "mark_names" in est else None
    mark_probs = est["mark_probs"] if "mark_probs" in est else None

    phi_true = None
    phi_true_agg = None
    f_true = None

    if kernel1_path is not None:
        true = np.load(kernel1_path, allow_pickle=True)

        phi_true_agg = true["phi_true_agg"]
        phi_true = true["phi_true"]
        f_true = true["f_true"]

    # plot_phi_by_mark(
    #     t,
    #     phi_est,
    #     phi_true,
    #     i=0,
    #     j=0,
    #     mark_names=mark_names,
    # )

    plot_marked_article_style(
        t=t,
        phi_est=phi_est,
        phi_true=phi_true,
        phi_true_agg=phi_true_agg,
        f_true=f_true,
        mark_probs=mark_probs,
        process_names=proc_names,
        mark_names=mark_names,
        save_prefix=save_prefix,
    )

def plot_loss_histories(
    folder="results/loss",
    prefix="loss_history_marked_row_",
    rows=4,
    loss_col="train_loss",
    save_path=None,
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    set_article_style()

    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, rows))

    for r in range(rows):
        path = os.path.join(folder, f"{prefix}{r}.csv")

        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue

        df = pd.read_csv(path)

        ax.plot(
            df["epoch"],
            df[loss_col],
            lw=1.7,
            color=colors[r],
            label=str(r + 1),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel(r"$\mathcal{L}$", fontsize=18)

    ax.grid(False)

    ax.tick_params(
        which="both",
        direction="in",
        top=False,
        right=False,
        width=0.9,
        length=3.5,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout()

    if save_path is not None:
        folder_out = os.path.dirname(save_path)
        if folder_out:
            os.makedirs(folder_out, exist_ok=True)

        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    # plot_loss_histories(folder="results/loss", prefix="loss_history_row_", rows=5, save_path="figures/loss.pdf")
    # run_unmarked(estimated_path="results/kernels_marked.npz", true_path="data/simulation/D3.npz", save_path="figures/sim_unmarked_kernels.pdf")
    # run_marked(kernel1_path="data/simulation/M5.npz", kernel2_path="results/kernels_marked.npz", save_prefix="temp")

    disc = np.load("results/discrete_hawkes_fit.npz", allow_pickle=True)
    t_disc = disc["t"]
    phi_disc = disc["phi"]


    cont = np.load("results/D2_USDC_USDT_001.npz", allow_pickle=True)
    t_cont = cont["t"]
    phi_cont = cont["phi"]
    mark_probs = cont["mark_probs"] if "mark_probs" in cont else None

    phi_cont = aggregate_time_kernel(phi_cont, mark_probs)

    t_cont_block, phi_cont_block = aggregate_continuous_kernel_to_blocks(
        t_cont,
        phi_cont,
        delta=12,
        support=phi_disc.shape[0],
    )

    true = np.load("data/simulation/M5.npz", allow_pickle=True)
    t_true = true["t"]
    phi_true = true["phi_true"]
    mark_probs_true = true["mark_probs"] if "mark_probs" in true else mark_probs
    
    phi_true = aggregate_time_kernel(phi_true, mark_probs_true)

    t_true_block, phi_true_block = aggregate_continuous_kernel_to_blocks(
        t_true,
        phi_true,
        delta=12,
        support=phi_disc.shape[0],
    )
    plot_kernel_grid_three(
        t=t_disc,
        phi_1=None,
        phi_2=phi_cont_block,
        phi_3=phi_disc,
        labels=(
            "True",
            "Neural Hawkes",
            "Kirchner"
        ),
        save_path="results/sim_discrete.pdf"
    )