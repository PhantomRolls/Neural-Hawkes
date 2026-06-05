import requests, time
import pandas as pd
import numpy as np
from tqdm import tqdm

API_KEY = "045de66ca74de0f46118b3179625fb75"
SUBGRAPH_ID = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
URL = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}"

POOL_ADDRESSES = [
    "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
    "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
    "0x11b815efB8f581194ae79006d24E0d814B7697F6",
    "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed",
    "0x3416cF6C708Da44DB2624D63ea0AAef7113527C6",
]

DAYS = 365
PAGE_SIZE = 1000
SLEEP_S = 0.03
MAX_PAGES = 5000

def run_query(q: str) -> dict:
    r = requests.post(URL, json={"query": q}, timeout=60)
    r.raise_for_status()
    js = r.json()
    if "errors" in js:
        raise RuntimeError(js["errors"])
    return js["data"]

def get_latest_timestamp() -> int:
    q = "{ swaps(first: 1, orderBy: timestamp, orderDirection: desc) { timestamp } }"
    return int(run_query(q)["swaps"][0]["timestamp"])

def utc_midnight(ts: int) -> int:
    dt = pd.to_datetime(ts, unit="s", utc=True)
    return int(dt.floor("D").timestamp())

def fetch_event(pool_addr: str, event_name: str, fields: str, t_start: int, t_end: int) -> pd.DataFrame:
    pool = pool_addr.lower()
    rows = []
    cursor = t_end - 1

    for _ in tqdm(range(MAX_PAGES), desc=f"{event_name}:{pool[:6]}…", unit="page", leave=False):
        q = f"""
        {{
          {event_name}(
            first:{PAGE_SIZE},
            orderBy:timestamp,
            orderDirection:desc,
            where:{{
              pool:"{pool}",
              timestamp_lte:{cursor},
              timestamp_gte:{t_start}
            }}
          ) {{
            {fields}
          }}
        }}
        """
        batch = run_query(q).get(event_name, [])
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1]["timestamp"]) - 1
        if cursor < t_start:
            break
        time.sleep(SLEEP_S)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["pool"] = pool
    df["event_type"] = event_name[:-1]  
    return df

swap_fields = """
id
timestamp
logIndex
origin
sender
recipient
amount0
amount1
amountUSD
sqrtPriceX96
tick
transaction { id blockNumber timestamp gasUsed gasPrice }
"""

mint_fields = """
id
timestamp
logIndex
origin
sender
owner
amount
amount0
amount1
amountUSD
tickLower
tickUpper
transaction { id blockNumber timestamp }
"""

burn_fields = """
id
timestamp
logIndex
origin
owner
amount
amount0
amount1
amountUSD
tickLower
tickUpper
transaction { id blockNumber timestamp }
"""

# --------- MAIN ---------
latest = get_latest_timestamp()
t_end = utc_midnight(latest)
t_start = t_end - DAYS * 24 * 3600
print("Window UTC:", pd.to_datetime(t_start, unit="s", utc=True), "->", pd.to_datetime(t_end, unit="s", utc=True))

parts = []
for pool in tqdm(POOL_ADDRESSES, desc="Pools", unit="pool"):
    parts.append(fetch_event(pool, "swaps", swap_fields, t_start, t_end))
    parts.append(fetch_event(pool, "mints", mint_fields, t_start, t_end))
    parts.append(fetch_event(pool, "burns", burn_fields, t_start, t_end))

df = pd.concat([p for p in parts if len(p) > 0], ignore_index=True)


tx = pd.json_normalize(df["transaction"]).add_prefix("transaction.")
df = pd.concat([df.drop(columns=["transaction"]), tx], axis=1)

df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
for c in ["logIndex", "tick", "tickLower", "tickUpper", "transaction.blockNumber"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

for c in ["amount0","amount1","amountUSD","amount","transaction.gasUsed","transaction.gasPrice"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df["price_1_per_0"] = np.where(df["tick"].notna(), np.power(1.0001, df["tick"].astype(float)), np.nan)

t0 = pd.to_datetime(t_start, unit="s", utc=True)
df["t"] = (df["timestamp"] - t0).dt.total_seconds()

sort_cols = [c for c in ["transaction.blockNumber", "timestamp", "logIndex"] if c in df.columns]
df = df.sort_values(sort_cols).reset_index(drop=True)

out = "data/real/365days.csv"
df.to_csv(out, index=False)

print("Saved:", out, "rows:", len(df))
print("event_type counts:\n", df["event_type"].value_counts(dropna=False))
print("pools:\n", df["pool"].value_counts().head())
