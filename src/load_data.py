import pandas as pd
import numpy as np

def load_real_data(pools=None, start=0, end=30, file="30days"):
    name_to_adress = {"USDC/WETH 0.05": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
                      "USDC/WETH 0.30": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
                      "WBTC/WETH 0.05": "0x11b815efb8f581194ae79006d24e0d814b7697f6",
                      "WBTC/WETH 0.30": "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed",
                      "USDC/USDT 0.01": "0x3416cf6c708da44db2624d63ea0aaef7113527c6",
    }
    adress_to_name = {adress: name for name, adress in name_to_adress.items()}
    price_factor = {
        "USDC/WETH 0.05": 1e-12,
        "USDC/WETH 0.30": 1e-12,
        "WBTC/WETH 0.05": 1e-10,
        "WBTC/WETH 0.30": 1e-10,
        "USDC/USDT 0.05": 1
    }
    if pools is None:
        pools = name_to_adress.keys()
    df = pd.read_parquet(f"data/real/{file}.parquet")
    df = df.sort_values(["transaction.blockNumber", "logIndex"])
    if isinstance(pools, str):
        pools = [pools]
    adresses = [name_to_adress[pool] for pool in pools]
    df = df[(df["pool"].isin(adresses)) & (df["t"]>=start*24*3600) & (df["t"]<end*24*3600)].reset_index(drop=True)

    u_reordered = np.empty(len(df))
    for _, idx in df.groupby("transaction.blockNumber").groups.items():
        idx = np.array(list(idx))
        u_reordered[idx] = np.sort(np.random.rand(len(idx))) # np.sort() pour conserver l'odre intrablock
    df["time"] = df["t"] + 12 * u_reordered
    
    # counts = df.groupby("transaction.blockNumber")["logIndex"].transform("count")
    # ranks = df.groupby("transaction.blockNumber").cumcount()
    # df["time"] = df["t"] + 12 * (ranks + np.random.rand(len(df))) / counts
    
    df["process"] = df["pool"].map(adress_to_name)
    df["price_factor"] = df["process"].map(price_factor)
    df["price"] = (df["sqrtPriceX96"].astype(float) / 2**96) ** 2 * df["price_factor"]
    df["SwapX2Y"] = df["amount1"] > 0
    df["SwapY2X"] = df["amount0"] > 0
    return df