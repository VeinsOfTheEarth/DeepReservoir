# deepreservoir/define_env/spring_peak_release/opportunity_index.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .swe_helpers import (
    SigmoidRuleParams,
    sigmoid_boundary_y,
    opportunity_index_from_margin,
    assemble_wy_metrics,
    prespring_storage_by_wy,
)

@dataclass
class OIParams:
    boundary: SigmoidRuleParams
    m0: float = 40.0
    beta: float = 0.916290731874155  # ~beta_from_target(omega=0.75, oi_at_pos_m0=0.90)
    omega_on_line: float = 0.75
    storage_method: str = "feb_mean"            # Feb mean storage
    swe_metric: str = "SWE_peak_by_Mar1_mm"     # or "SWE_Feb_max_mm"

    @staticmethod
    def load(path: Path | str) -> "OIParams":
        with open(path, "r") as f:
            cfg = json.load(f)
        b = cfg["boundary"]
        return OIParams(
            boundary=SigmoidRuleParams(**b),
            m0=cfg.get("m0", 40.0),
            beta=cfg.get("beta", 0.916290731874155),
            omega_on_line=cfg.get("omega_on_line", 0.75),
            storage_method=cfg.get("storage_method", "feb_mean"),
            swe_metric=cfg.get("swe_metric", "SWE_peak_by_Mar1_mm"),
        )

def precompute_oi_by_wy(model_df: pd.DataFrame, oi: OIParams) -> pd.DataFrame:
    """
    Returns DataFrame indexed by water year with:
        ['storage_x', 'swe_y', 'margin_mm', 'oi', 'go']
    """
    # x = pre-spring storage (e.g. Feb mean)
    x = prespring_storage_by_wy(model_df, method=oi.storage_method)

    # y = SWE metric by WY (computed from daily animas SWE)
    swe_daily = model_df["animas_swe_m"]
    # an_daily not needed for this y metric; pass a placeholder series with same index
    dummy_q = model_df.get("animas_farmington_q_cfs", pd.Series(index=model_df.index, dtype=float))
    wy = assemble_wy_metrics(swe_daily, dummy_q)
    if oi.swe_metric not in wy.columns:
        raise KeyError(f"Unknown SWE metric {oi.swe_metric!r}. Have: {list(wy.columns)}")
    y = wy[oi.swe_metric]

    df = pd.concat([x.rename("storage_x"), y.rename("swe_y")], axis=1).dropna()

    margin = df["swe_y"].values - sigmoid_boundary_y(df["storage_x"].values, oi.boundary)
    oi_vals, _ = opportunity_index_from_margin(margin, m0=oi.m0, beta=oi.beta, omega_on_line=oi.omega_on_line)

    df["margin_mm"] = margin
    df["oi"] = oi_vals
    df["go"] = df["oi"] >= oi.omega_on_line
    return df
