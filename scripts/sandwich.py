import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SandwichAttack:
    """Représente une attaque sandwich détectée."""
    block_number: int
    front_run_idx: int          # index pandas de la ligne front-run
    back_run_idx: int           # index pandas de la ligne back-run
    victim_idxs: List[int]      # index pandas des victimes
    attacker_wallet: str
    front_log_index: int
    back_log_index: int
    n_victims: int
    volume_usd_front: Optional[float] = None
    volume_usd_back: Optional[float] = None
    volume_usd_victims: Optional[float] = None

    def to_dict(self):
        return {
            "block_number": self.block_number,
            "attacker_wallet": self.attacker_wallet,
            "front_run_idx": self.front_run_idx,
            "back_run_idx": self.back_run_idx,
            "victim_idxs": self.victim_idxs,
            "n_victims": self.n_victims,
            "front_log_index": self.front_log_index,
            "back_log_index": self.back_log_index,
            "volume_usd_front": self.volume_usd_front,
            "volume_usd_back": self.volume_usd_back,
            "volume_usd_victims": self.volume_usd_victims,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Détection des attaques sandwich  (Algorithm 2 du papier)
# ─────────────────────────────────────────────────────────────────────────────

def detect_sandwiches(
    df: pd.DataFrame,
    process: str,
    wallet_col: str = "origin",
) -> pd.DataFrame:
    
    # ── 1. Filtrer la pool et les swaps uniquement ──────────────────────────
    pool_df = df[df["process"] == process].copy()

    if pool_df.empty:
        raise ValueError(f"Pool '{process}' introuvable dans le dataframe.")

    # Convertir les colonnes utiles
    pool_df["_block"] = pd.to_numeric(pool_df["transaction.blockNumber"], errors="coerce")
    pool_df["_log"]   = pd.to_numeric(pool_df["logIndex"], errors="coerce")

    # Normaliser SwapX2Y / SwapY2X en booléens
    def _to_bool(col):
        if col.dtype == bool:
            return col
        return col.map(lambda x: str(x).strip().lower() in ("true", "1", "yes"))

    pool_df["_x2y"] = _to_bool(pool_df["SwapX2Y"])
    pool_df["_y2x"] = _to_bool(pool_df["SwapY2X"])

    # Signe du swap : +1 = X→Y, -1 = Y→X, 0 = pas un swap
    def _sign(row):
        if row["_x2y"]:
            return 1
        if row["_y2x"]:
            return -1
        return 0

    pool_df["_sign"] = pool_df.apply(_sign, axis=1)

    # Conserver uniquement les swaps (sign ≠ 0)
    swaps = pool_df[pool_df["_sign"] != 0].copy()
    swaps = swaps.sort_values(["_block", "_log"])

    has_usd = "amountUSD" in swaps.columns

    # ── 2. Itérer bloc par bloc ─────────────────────────────────────────────
    attacks: List[SandwichAttack] = []

    for block_num, block_swaps in swaps.groupby("_block"):
        block_swaps = block_swaps.sort_values("_log").reset_index()
        # reset_index conserve l'ancien index dans la colonne "index"
        n = len(block_swaps)

        used_as_back = set()   # évite qu'un back-run soit réutilisé

        for i in range(n):
            front = block_swaps.iloc[i]
            if front.name in used_as_back:
                continue

            front_sign   = front["_sign"]
            front_wallet = front[wallet_col]
            front_log    = front["_log"]

            # Cherche le premier swap de signe opposé avec le même wallet
            back_candidate = None
            for j in range(i + 1, n):
                cand = block_swaps.iloc[j]
                if (
                    cand["_sign"] == -front_sign
                    and cand[wallet_col] == front_wallet
                    and j not in used_as_back
                ):
                    back_candidate = cand
                    back_j = j
                    break

            if back_candidate is None:
                continue

            back_log = back_candidate["_log"]

            victims = block_swaps.iloc[i + 1 : back_j]
            victims = victims[victims["_sign"] == front_sign]

            if victims.empty:
                continue

            victim_orig_idxs = list(victims["index"])
            front_orig_idx   = front["index"]
            back_orig_idx    = back_candidate["index"]

            vol_front   = None
            vol_back    = None
            vol_victims = None
            if has_usd:
                try:
                    vol_front   = float(pool_df.loc[front_orig_idx, "amountUSD"])
                    vol_back    = float(pool_df.loc[back_orig_idx, "amountUSD"])
                    vol_victims = float(pool_df.loc[victim_orig_idxs, "amountUSD"].sum())
                except Exception:
                    pass

            attacks.append(SandwichAttack(
                block_number     = int(block_num),
                front_run_idx    = front_orig_idx,
                back_run_idx     = back_orig_idx,
                victim_idxs      = victim_orig_idxs,
                attacker_wallet  = front_wallet,
                front_log_index  = int(front_log),
                back_log_index   = int(back_log),
                n_victims        = len(victim_orig_idxs),
                volume_usd_front   = vol_front,
                volume_usd_back    = vol_back,
                volume_usd_victims = vol_victims,
            ))

            used_as_back.add(back_j)

    if not attacks:
        print(f"[detect_sandwiches] Aucune attaque sandwich détectée dans '{process}'.")
        return pd.DataFrame()

    result = pd.DataFrame([a.to_dict() for a in attacks])
    print(
        f"[detect_sandwiches] {len(result)} attaques détectées dans '{process}' "
        f"({result['n_victims'].sum()} victimes au total)."
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Suppression des événements MEV du dataframe
# ─────────────────────────────────────────────────────────────────────────────

def remove_sandwiches(
    df: pd.DataFrame,
    sandwiches: pd.DataFrame,
    remove_victims: bool = False,
) -> pd.DataFrame:
    
    if sandwiches.empty:
        return df.copy()

    idxs_to_remove = set()

    for _, row in sandwiches.iterrows():
        idxs_to_remove.add(row["front_run_idx"])
        idxs_to_remove.add(row["back_run_idx"])
        if remove_victims:
            idxs_to_remove.update(row["victim_idxs"])

    df_clean = df.drop(index=list(idxs_to_remove & set(df.index)))
    n_removed = len(df) - len(df_clean)
    print(f"[remove_sandwiches] {n_removed} lignes supprimées "
          f"({'victimes incluses' if remove_victims else 'victimes conservées'}).")
    return df_clean


# ─────────────────────────────────────────────────────────────────────────────
# Statistiques rapides
# ─────────────────────────────────────────────────────────────────────────────

def sandwich_stats(df: pd.DataFrame, sandwiches: pd.DataFrame, process: str) -> dict:
    if sandwiches.empty:
        return {}

    pool_swaps = df[(df["process"] == process) & (
        df["SwapX2Y"].map(lambda x: str(x).lower() in ("true","1")) |
        df["SwapY2X"].map(lambda x: str(x).lower() in ("true","1"))
    )]

    total_vol = pd.to_numeric(pool_swaps["amountUSD"], errors="coerce").sum()

    attacker_idxs = set(sandwiches["front_run_idx"]) | set(sandwiches["back_run_idx"])
    attacker_vol  = pd.to_numeric(
        df.loc[list(attacker_idxs & set(df.index)), "amountUSD"], errors="coerce"
    ).sum()

    victim_idxs_all = [idx for lst in sandwiches["victim_idxs"] for idx in lst]
    victim_vol = pd.to_numeric(
        df.loc[list(set(victim_idxs_all) & set(df.index)), "amountUSD"], errors="coerce"
    ).sum()

    top_attackers = (
        sandwiches.groupby("attacker_wallet")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .rename("n_attacks")
    )

    stats = {
        "process": process,
        "n_attacks": len(sandwiches),
        "n_victims_total": int(sandwiches["n_victims"].sum()),
        "pct_sandwich_volume": round(100 * attacker_vol / total_vol, 2) if total_vol else None,
        "pct_victim_volume":   round(100 * victim_vol  / total_vol, 2) if total_vol else None,
        "top_attackers": top_attackers,
    }

    print(f"\n{'═'*55}")
    print(f"  Sandwich stats — {process}")
    print(f"{'═'*55}")
    print(f"  Attaques détectées    : {stats['n_attacks']}")
    print(f"  Victimes totales      : {stats['n_victims_total']}")
    if stats['pct_sandwich_volume'] is not None:
        print(f"  Volume attaquant (%)  : {stats['pct_sandwich_volume']} %")
        print(f"  Volume victimes  (%)  : {stats['pct_victim_volume']} %")
    print(f"\n  Top 10 wallets attaquants :")
    print(top_attackers.to_string())
    print(f"{'═'*55}\n")

    return stats



if __name__ == "__main__":
    from src.load_data import load_real_data
    process = "USDC/USDT 0.01"
    df = load_real_data(process)
    sandwiches = detect_sandwiches(df, process=process)
    print(sandwiches[["block_number", "attacker_wallet", "n_victims",
                       "volume_usd_front", "volume_usd_victims"]].to_string())

    df_clean = remove_sandwiches(df, sandwiches)
    df_clean.to_parquet("data/real/30days_no_sandwich.parquet")
    stats    = sandwich_stats(df, sandwiches, process=process)

