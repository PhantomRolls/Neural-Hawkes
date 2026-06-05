import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from src.neural_hawkes.preprocessing import (
    events_from_dataframe_marked,
    estimate_G_tensor_marked,
    H_from_G_marked,
    plot_G_marked,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Réseau DGM =================================================
class DGMCell(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int):
        super().__init__()
        self.Z = nn.Linear(dim_in + dim_hidden, dim_hidden)
        self.G = nn.Linear(dim_in + dim_hidden, dim_hidden)
        self.R = nn.Linear(dim_in + dim_hidden, dim_hidden)
        self.H = nn.Linear(dim_in + dim_hidden, dim_hidden)

    def forward(self, x: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, S], dim=1)

        Z = torch.sigmoid(self.Z(inp))
        G = torch.sigmoid(self.G(inp))
        R = torch.sigmoid(self.R(inp))

        HR_inp = torch.cat([x, S * R], dim=1)
        H = F.relu(self.H(HR_inp))

        S_new = (1.0 - G) * H + Z * S
        return S_new


class DGMNet(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dim_hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.input_layer = nn.Linear(dim_in, dim_hidden)
        self.layers = nn.ModuleList([DGMCell(dim_in, dim_hidden) for _ in range(n_layers)])
        self.output_layer = nn.Linear(dim_hidden, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = F.relu(self.input_layer(x))
        for layer in self.layers:
            S = layer(x, S)
        out = self.output_layer(S)
        # out = F.softplus(self.output_layer(S))
        return out
# ============================================================



def make_linear_grid(t_min, T, n):
    return torch.linspace(t_min, T, n, device=device)

def make_log_linear_grid(t_min, h, T, n_lin, n_log):
    g_lin = torch.linspace(
        t_min,
        h,
        n_lin + 1,
        device=device,
        dtype=torch.float64,
    )

    g_log = h * (T / h) ** torch.linspace(
        1 / n_log,
        1.0,
        n_log,
        device=device,
        dtype=torch.float64,
    )

    return torch.cat([g_lin, g_log])

def sample_collocation_points(batch_size: int,
                              tau: float,
                              T: float,
                              short_ratio: float = 0.3,
                              t_min: float = 1e-4) -> torch.Tensor:
    n_short = int(batch_size * short_ratio)
    n_long = batch_size - n_short

    u1 = torch.rand(n_short, 1, device=device)
    t_short = t_min + (tau - t_min) * u1
    
    u2 = torch.rand(n_long, 1, device=device)
    t_long = tau + (T - tau) * u2

    t_batch = torch.cat([t_short, t_long], dim=0)
    t_batch, _ = torch.sort(t_batch, dim=0)

    return t_batch

def prepare_hawkes_data_marked(df, T_kernel):
    events, marks, mark_probs, lambdas, T_obs, proc_names, mark_names = events_from_dataframe_marked(df)
    print(mark_probs)
    g_edges = make_linear_grid(
        t_min=1e-2,
        T=T_kernel,
        n=80,
    )
    # g_edges = make_log_linear_grid(
    #     t_min=1e-2,
    #     h=0.5,
    #     T=T_kernel,
    #     n_lin=50,
    #     n_log=80,
    # )

    g_grid = 0.5 * (g_edges[:-1] + g_edges[1:])

    G_tensor = estimate_G_tensor_marked(
        events=events,
        marks=marks,
        g_edges=g_edges,
        lambdas=lambdas,
        T_obs=T_obs,
        n_marks=len(mark_names),
    )
    plot_G_marked(G_tensor, g_grid, mark_names)

    return {
        "events": events,
        "marks": marks,
        "mark_probs": mark_probs.to(device),
        "lambdas": lambdas.to(device),
        "T_obs": T_obs,
        "proc_names": proc_names,
        "mark_names": mark_names,
        "g_grid": g_grid.to(device),
        "T_kernel": T_kernel,
        "G_tensor": G_tensor.to(device),
    }

def build_quadrature(q: int,
                     s_min: float,
                     s_max: float):
    u = torch.linspace(0, 1, q + 1, device=device, dtype=torch.float64)
    s_grid = s_min * (s_max / s_min) ** u

    w_s = torch.empty_like(s_grid)
    if s_grid.numel() == 1:
        w_s[0] = s_max - s_min
    else:
        w_s[0] = 0.5 * (s_grid[1] - s_grid[0])
        w_s[-1] = 0.5 * (s_grid[-1] - s_grid[-2])
        w_s[1:-1] = 0.5 * (s_grid[2:] - s_grid[:-2])

    return s_grid, w_s

def interpolate_G_row_marked(
    t_batch,
    G_tensor,
    g_grid,
    row_index,
    mark_index,
):
    t = t_batch.squeeze(-1)
    tg = g_grid.to(t.device)
    G_row = G_tensor[row_index, :, mark_index]

    idx = torch.searchsorted(tg, t).clamp(1, tg.numel() - 1)

    t0 = tg[idx - 1]
    t1 = tg[idx]

    w = ((t - t0) / (t1 - t0 + 1e-12)).unsqueeze(1)

    g0 = G_row[:, idx - 1].transpose(0, 1)
    g1 = G_row[:, idx].transpose(0, 1)

    return g0 + w * (g1 - g0)              

def build_H_cache_marked(
    t_batch,
    s_grid,
    G_tensor,
    g_grid,
    lambdas,
    n_marks,
):
    D = G_tensor.shape[0]
    B = t_batch.shape[0]
    Qs = s_grid.shape[0]

    tau = t_batch - s_grid.view(1, Qs)

    H_cache = torch.empty(
        D,
        D,
        n_marks,
        n_marks,
        B,
        Qs,
        device=device,
        dtype=G_tensor.dtype,
    )

    for k in range(D):
        for j in range(D):
            for x in range(n_marks):
                for z in range(n_marks):

                    H_cache[k, j, x, z] = H_from_G_marked(
                        G_tensor,
                        g_grid,
                        lambdas,
                        k,
                        j,
                        tau,
                        m_x=x,
                        m_z=z,
                    )

    return H_cache


def integral_term_from_cache_marked(
    model_i,
    s_grid_std,
    mark_grid_std,
    w_s,
    H_cache,
    mark_probs,
    m_batch,
):
    D = mark_probs.shape[0]
    M = mark_probs.shape[1]
    Q = s_grid_std.shape[0]
    B = m_batch.shape[0]

    phi_all = []

    for z in range(M):
        z_col = torch.full(
            (Q, 1),
            mark_grid_std[z],
            device=device,
            dtype=s_grid_std.dtype,
        )

        x_in = torch.cat([s_grid_std, z_col], dim=1)

        phi_z = model_i(x_in)  # (Q, D)

        phi_all.append(phi_z)

    # (Q, M, D)
    phi_all = torch.stack(phi_all, dim=1)

    # On construit H_selected[b,k,j,z,q] = H_cache[k,j,x_b,z,b,q]
    H_selected = torch.empty(
        B,
        D,
        D,
        M,
        Q,
        device=device,
        dtype=H_cache.dtype,
    )

    for b in range(B):
        x_b = int(m_batch[b].item())
        H_selected[b] = H_cache[:, :, x_b, :, b, :]

    # I[b,j] = sum_{k,z,q} phi_i,k(s_q,z) H_k,j(t_b-s_q,x_b,z) w_q p_k(z)
    I = torch.einsum(
        "qzk,bkjzq,q,kz->bj",
        phi_all,
        H_selected,
        w_s,
        mark_probs,
    )
    return I


def loss_for_row_weighted_marked(
    model_i,
    x_batch,
    G_hat_i,
    s_grid_std,
    mark_grid_std,
    w_s,
    H_cache,
    mark_probs,
    m_batch,
    omega_eps=5.0,
):
    dtype = next(model_i.parameters()).dtype

    x_batch = x_batch.to(dtype=dtype)
    G_hat_i = G_hat_i.to(dtype=dtype)

    phi_t = model_i(x_batch)

    I = integral_term_from_cache_marked(
        model_i=model_i,
        s_grid_std=s_grid_std.to(dtype=dtype),
        mark_grid_std=mark_grid_std.to(dtype=dtype),
        w_s=w_s.to(dtype=dtype),
        H_cache=H_cache.to(dtype=dtype),
        mark_probs=mark_probs.to(dtype=dtype),
        m_batch=m_batch,
    )

    residual = G_hat_i - phi_t - I
    sq = residual.pow(2)

    cum_sq = torch.cumsum(sq, dim=0)
    total_sq = cum_sq[-1:, :].clamp_min(1e-12)

    weights = torch.ones_like(sq)
    if sq.shape[0] > 1:
        weights[1:] = torch.exp(-omega_eps * cum_sq[:-1] / total_sq)

    return (weights * sq).mean()


def train_one_row_marked(
    precomp: dict,
    row_index: int = 0,
    quad_points: int = 250,
    dim_hidden: int = 64,
    n_layers: int = 1,
    lr: float = 1e-3,
    num_epochs: int = 1000,
    batch_size: int = 8,
    training_size: int = 1024,
):
    G_tensor = precomp["G_tensor"]
    g_grid = precomp["g_grid"]
    T_kernel = precomp["T_kernel"]
    lambdas = precomp["lambdas"]
    proc_names = precomp["proc_names"]
    mark_names = precomp["mark_names"]
    mark_probs = precomp["mark_probs"]

    D = G_tensor.shape[0]
    M = G_tensor.shape[2]

    ref_t = g_grid.view(-1, 1)
    mean_t = ref_t.mean(dim=0, keepdim=True)
    std_t = ref_t.std(dim=0, keepdim=True).clamp_min(1e-8)

    mark_grid = torch.arange(M, device=device, dtype=torch.float64).view(-1, 1)
    mean_m = mark_grid.mean(dim=0, keepdim=True)
    std_m = mark_grid.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
    mark_grid_std = ((mark_grid - mean_m) / std_m).view(-1)

    def lr(epoch: int, num_epochs: int, lr0: float) -> float:
        return lr0 * (100.0 ** (-(epoch + 1) / num_epochs))

    model_i = DGMNet(
        dim_in=2,
        dim_out=D,
        dim_hidden=dim_hidden,
        n_layers=n_layers,
    ).to(device, dtype=torch.float64)

    optimizer = torch.optim.Adam(model_i.parameters(), lr=lr)

    s_grid, w_s = build_quadrature(
        quad_points,
        s_min=1e-4,
        s_max=T_kernel,
    )
    s_grid_raw = s_grid.view(-1, 1)
    s_grid_std = (s_grid_raw - mean_t) / std_t

    # Validation
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    patience = 100
    rel_tol = 0.01
    from collections import deque
    val_window = deque(maxlen=20)

    history = []

    for epoch in range(num_epochs):
        current_lr = lr(epoch, num_epochs, lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        t_min = float(g_grid[0])
        T = float(g_grid[-1])

        t_train = sample_collocation_points(
            batch_size=training_size,
            tau=1.0,
            t_min=t_min,
            T=T,
            short_ratio=0.3,
        )

        m_train = torch.randint(
            low=0,
            high=M,
            size=(training_size, 1),
            device=device,
        )

        t_train_std = (t_train - mean_t) / std_t
        m_train_std = (m_train.to(torch.float64) - mean_m) / std_m

        x_train = torch.cat([t_train_std, m_train_std], dim=1)

        G_hat_train = torch.empty(
            training_size,
            D,
            device=device,
            dtype=torch.float64,
        )

        for m in range(M):
            mask = (m_train.squeeze(1) == m)
            if mask.any():
                G_hat_train[mask] = interpolate_G_row_marked(
                    t_batch=t_train[mask],
                    G_tensor=G_tensor,
                    g_grid=g_grid,
                    row_index=row_index,
                    mark_index=m,
                )
        H_cache_train = build_H_cache_marked(
            t_batch=t_train,
            s_grid=s_grid,
            G_tensor=G_tensor,
            g_grid=g_grid,
            lambdas=lambdas,
            n_marks=M,
        )

        perm = torch.randperm(training_size, device=device)

        train_loss_sum = 0.0
        n_steps = 0

        for start in range(0, training_size, batch_size):
            idx = perm[start:start + batch_size]
            x_mb = x_train[idx]
            G_hat_mb = G_hat_train[idx]
            H_cache_mb = H_cache_train[:, :, :, :, idx, :]
            m_mb = m_train.squeeze(1)[idx]

            loss = loss_for_row_weighted_marked(
                model_i=model_i,
                x_batch=x_mb,
                G_hat_i=G_hat_mb,
                s_grid_std=s_grid_std,
                mark_grid_std=mark_grid_std,
                w_s=w_s,
                H_cache=H_cache_mb,
                mark_probs=mark_probs,
                m_batch=m_mb,
                omega_eps=5.0,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_steps += 1
        train_loss_epoch = train_loss_sum / max(n_steps, 1)

        with torch.no_grad():

            t_val = sample_collocation_points(
                batch_size=128,
                tau=1.0,
                t_min=t_min,
                T=T,
                short_ratio=0.3,
            )

            m_val = torch.randint(
                low=0,
                high=M,
                size=(128, 1),
                device=device,
            )

            t_val_std = (t_val - mean_t) / std_t
            m_val_std = (m_val.to(torch.float64) - mean_m) / std_m

            x_val = torch.cat([t_val_std, m_val_std], dim=1)

            G_hat_val = torch.empty(
                128,
                D,
                device=device,
                dtype=torch.float64,
            )

            for m in range(M):
                mask = (m_val.squeeze(1) == m)
                if mask.any():
                    G_hat_val[mask] = interpolate_G_row_marked(
                        t_batch=t_val[mask],
                        G_tensor=G_tensor,
                        g_grid=g_grid,
                        row_index=row_index,
                        mark_index=m,
                    )

            H_cache_val = build_H_cache_marked(
                t_batch=t_val,
                s_grid=s_grid,
                G_tensor=G_tensor,
                g_grid=g_grid,
                lambdas=lambdas,
                n_marks=M,
            )
            m_val_batch = m_val.squeeze(1)
            val_loss = loss_for_row_weighted_marked(
                model_i=model_i,
                x_batch=x_val,
                G_hat_i=G_hat_val,
                s_grid_std=s_grid_std,
                mark_grid_std=mark_grid_std,
                w_s=w_s,
                H_cache=H_cache_val,
                mark_probs=mark_probs,
                m_batch=m_val_batch,
                omega_eps=0.0,
            ).item()

            val_window.append(val_loss)

            if len(val_window) == val_window.maxlen:
                smoothed_val = np.mean(val_window)

                if smoothed_val < best_val_loss * (1 - rel_tol):
                    best_val_loss = smoothed_val
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model_i.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                smoothed_val = np.nan

            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch+1} | "
                    f"Best val = {best_val_loss:.6e}"
                )
                break

        history.append({
            "row": row_index,
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss_epoch,
            "val_loss": val_loss,
            "val_loss_smooth": smoothed_val,
        })

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch+1:4d} | "
                f"LR = {current_lr:.3e} | "
                f"Train = {train_loss_epoch:.6e} | "
                f"Val = {val_loss:.6e}"
            )

    if best_state is not None:
        model_i.load_state_dict(best_state) 

    history_df = pd.DataFrame(history)
    history_df.to_csv(f"results/loss/loss_history_row_{row_index}.csv", index=False)

    model_i.input_mean_t = mean_t.detach().clone()
    model_i.input_std_t = std_t.detach().clone()
    model_i.input_mean_m = mean_m.detach().clone()
    model_i.input_std_m = std_m.detach().clone()

    return model_i, G_tensor, g_grid, lambdas, proc_names, mark_names

def train_all_rows_marked(df, T_kernel, **train_kwargs):

    precomp = prepare_hawkes_data_marked(df=df, T_kernel=T_kernel)

    D = precomp["G_tensor"].shape[0]
    M = precomp["G_tensor"].shape[2]

    print(f"Processus : {precomp['proc_names']}")
    print(f"Marques   : {precomp['mark_names']}")
    print(f"T_obs     : {precomp['T_obs']:.4f}")
    print(f"Lambdas   : {precomp['lambdas'].cpu().numpy()}")
    print(f"Mark probs:\n{precomp['mark_probs'].cpu().numpy()}")
    print("G_tensor prêt :", tuple(precomp["G_tensor"].shape))

    models = []

    for i in range(D):
        print(f"\n=== Training marked row {i} ===")

        model_i, G_tensor, g_grid, lambdas, proc_names, mark_names = train_one_row_marked(
            precomp=precomp,
            row_index=i,
            **train_kwargs,
        )

        models.append(model_i)

    return models, G_tensor, lambdas, g_grid, proc_names, mark_names

def evaluate_all_rows_marked(models, t_plot, mark_names):
    t_torch = t_plot.clone().detach().to(device=device, dtype=torch.float64).view(-1, 1)

    D = len(models)
    M = len(mark_names)
    n_points = len(t_plot)

    phi_all = np.zeros((n_points, D, D, M), dtype=np.float64)
    for i, model_i in enumerate(models):
        for m in range(M):
            m_torch = torch.full(
                (n_points, 1),
                float(m),
                device=device,
                dtype=torch.float64,
            )

            z_t = (t_torch - model_i.input_mean_t.to(device)) / model_i.input_std_t.to(device)
            z_m = (m_torch - model_i.input_mean_m.to(device)) / model_i.input_std_m.to(device)

            x = torch.cat([z_t, z_m], dim=1)

            with torch.no_grad():
                pred = model_i(x).cpu().numpy()
            phi_all[:, i, :, m] = pred
    return phi_all  

