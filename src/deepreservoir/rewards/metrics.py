# navajo_rl/eval/metrics.py
import pandas as pd

def compute_basic_metrics(traj: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row table of simple KPIs from a rollout trajectory."""
    out = {}

    if "reward" in traj:
        out["reward_sum"] = float(traj["reward"].sum())
        out["reward_mean"] = float(traj["reward"].mean())

    if {"niip_demand_m3_d","release_total_m3_d"} <= set(traj.columns):
        demand = traj["niip_demand_m3_d"].fillna(0.0)
        release = traj["release_total_m3_d"].fillna(0.0)
        mask = demand > 0
        out["niip_days_with_demand"] = int(mask.sum())
        out["niip_days_met"] = int((release[mask] >= demand[mask]).sum())
        out["niip_pct_days_met"] = 100.0 * out["niip_days_met"] / max(1, out["niip_days_with_demand"])
        out["niip_vol_met_pct"] = 100.0 * release[mask].sum() / max(1.0, demand[mask].sum())

    return pd.DataFrame([out])
