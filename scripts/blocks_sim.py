import numpy as np
import pandas as pd
from pathlib import Path


def choose_block_size_for_target_empty_rate(times, target_empty_rate=0.5):
    times = np.asarray(times, dtype=float)
    T = times.max() - times.min()
    n = len(times)
    intensity = n / T

    delta = -np.log(target_empty_rate) / intensity
    return float(delta)


def block_and_redispatch_simulated_hawkes(
    data_path="data/simulation/M5.parquet",
    output_path="data/simulation/M5_block_redispatched.parquet",
    target_empty_rate=0.4678,
    n_events_target=230_249,
):
    rng = np.random.default_rng(123)

    df = pd.read_parquet(data_path).copy()
    df = df.sort_values("time").reset_index(drop=True)
    df = df.iloc[:n_events_target].copy()

    original_time = df["time"].to_numpy(dtype=float)
    t0 = original_time.min()


    block_size = choose_block_size_for_target_empty_rate(
            original_time,
            target_empty_rate=target_empty_rate,
        )

    block_number = np.floor((original_time - t0) / block_size).astype(int)
    block_start = t0 + block_number * block_size

    df["original_time"] = original_time
    df["block_number"] = block_number
    df["block_start"] = block_start

    new_time = np.empty(len(df), dtype=float)

    for _, idx in df.groupby("block_number", sort=True).groups.items():
        idx = np.asarray(list(idx))
        u = np.sort(rng.random(len(idx)))
        new_time[idx] = df.loc[idx, "block_start"].to_numpy() + block_size * u

    df["time"] = new_time
    df = df.sort_values("time").reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    n_blocks = int(block_number.max() + 1)
    counts = pd.Series(block_number).value_counts()
    counts_all = counts.reindex(np.arange(n_blocks), fill_value=0)

    empty_rate = (counts_all == 0).mean()
    mean_events_per_block = counts_all.mean()
    median_events_per_block = counts_all.median()

    print(f"Saved to: {output_path}")
    print(f"block_size = {block_size:.6f}")
    print(f"n_events = {len(df)}")
    print(f"n_blocks = {n_blocks}")
    print(f"empty_rate = {100*empty_rate:.2f}%")
    print(f"mean_events_per_block = {mean_events_per_block:.2f}")
    print(f"median_events_per_block = {median_events_per_block:.2f}")

    return df

if __name__ == "__main__":
    df_blocked = block_and_redispatch_simulated_hawkes(
        data_path="data/simulation/M5.parquet",
        output_path="data/simulation/M5_block_redispatched.parquet",
        target_empty_rate=0.4678,
        n_events_target=230_249,
    )