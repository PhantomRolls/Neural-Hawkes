import pandas as pd
import numpy as np
from src.neural_hawkes.preprocessing import estimate_T_kernel
from src.neural_hawkes.neural_model import train_all_rows_marked, evaluate_all_rows_marked
from src.load_data import load_real_data
from src.discrete_hawkes_estimation import estimate_discrete_hawkes_from_df

if __name__ == "__main__":
    # For simulated data =====================================
    # data_path = "data/simulation/M5_block_redispatched.parquet"
    # df = pd.read_parquet(data_path)
    # print(df.head(5))
    # print(df.mark.unique())
    # print(len(df), "events")
    # ========================================================

    # For real data ==========================================
    df = load_real_data(["USDC/USDT 0.01"], file="30days")
    df = df[df["event_type"] == "swap"].copy()
    # df["process"] = df["event_type"]
    df["process"] = np.where(df["SwapX2Y"], "X2Y", "Y2X")
    bins = [0, 100, 1_000, 10_000, 100_000, np.inf]
    labels = ["tiny", "small", "medium", "large", "whale"]
    df["mark_name"] = pd.cut(
        df["amountUSD"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    df["mark"] = df["mark_name"].cat.codes
    print(df[["time", "process", "amountUSD", "mark", "mark_name"]].head())
    print(len(df), "events")
    print(df.groupby(["process", "mark_name"]).size())
    # ========================================================

    # Continuous model =======================================
    # T_kernel = estimate_T_kernel(df, q=0.95)
    # print("T_kernel = ", T_kernel)
    # models, G_tensor, lambdas, g_grid, proc_names, mark_names = train_all_rows_marked(
    #     df=df,
    #     T_kernel=T_kernel,
    #     quad_points=250,
    #     dim_hidden=64,
    #     n_layers=1,
    #     lr=1e-3,
    #     num_epochs=1000,       
    #     batch_size=8,          
    #     training_size=1024,     
    # )
    # phi_all = evaluate_all_rows_marked(models, g_grid, mark_names)
    # np.savez(
    #     "results/kernels_marked.npz",
    #     t=g_grid,
    #     phi=phi_all,
    #     G_emp=G_tensor,
    #     lambdas=lambdas,
    #     proc_names=np.array(proc_names, dtype=str),
    #     mark_names=np.array(mark_names, dtype=str),
    # )
    # ========================================================



    # Discrete model =========================================
    T_kernel = estimate_T_kernel(df, q=0.95)
    support = int(T_kernel // 12 + 1)
    print(T_kernel)
    # df.rename(columns={"block_number": "transaction.blockNumber"}, inplace=True)
    df = df.groupby(['transaction.blockNumber', 'process'], as_index=False).agg(count=('process', 'count'))
    df["time"] = df["transaction.blockNumber"] - df["transaction.blockNumber"].iloc[0]
    print(df.head(5))
    result = estimate_discrete_hawkes_from_df(
        df=df,
        support=support,
        time_col="time",
        process_col="process",
        value_col="count",
        output_dir="results/",
    )
    # ========================================================