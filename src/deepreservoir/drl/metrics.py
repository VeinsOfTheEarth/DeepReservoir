"""deepreservoir.drl.metrics

Registry-driven metrics computation for DeepReservoir experiment runs.

Design goals
------------
- **Low-touch integration**: compute everything from the test rollout dataframe
  produced by :func:`deepreservoir.drl.model.run_test_rollout`.
- **Composable**: metrics are registered by name and can be grouped.
- **Batch-friendly**: metrics return a single-row DataFrame so many runs can be
  concatenated for quick comparisons.

This module intentionally focuses on wiring and extensibility; metric
definitions can be expanded over time.

How to add a new metric
-----------------------
1) Write a function that accepts ``df_test: pd.DataFrame`` (and optional kwargs)
   and returns ``dict[str, float]``.
2) Register it in ``METRIC_REGISTRY`` with a short name.
3) Add it to one or more groups in ``METRIC_GROUPS`` (e.g. "core").
4) Document any emitted output keys in ``METRIC_DEFINITIONS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import json
import numpy as np
import pandas as pd


_CFS_DAY_TO_ACRE_FEET = 86400.0 / 43560.0  # 1 cfs sustained for 1 day -> acre-feet


# -----------------------------------------------------------------------------
# Metric definitions (for documentation / CSV headers)
# -----------------------------------------------------------------------------
#
# Notes
# - All "fraction of days" metrics (e.g., *_pct_days_*) are fractions in [0, 1] (not 0–100).
# - `hydropower_pct_of_historic` and `niip_annual_volume_pct_of_contract` are in percent (0–100).
# - Objective-aligned metrics return NaN if that objective was not active in the
#   experiment (detected by absence of `rc_<objective>.*` columns in df_test).
# - Reward-component summaries are dynamic because reward specs vary by run.

METRIC_DEFINITIONS: dict[str, str] = {
    # --- Rewards ---
    "total_reward": "Sum of per-timestep total reward over the test rollout (unitless).",
    "mean_reward": "Mean per-timestep total reward over the test rollout (unitless).",

    # --- Dam safety (storage) ---
    "dam_safety_pct_days_within_storage_bounds": (
        "Fraction of days storage is within [min_storage_af, max_storage_af]. Uses storage_agent_af_end if "
        "available, else storage_agent_af. Bounds come from min_storage_af/max_storage_af if present; otherwise "
        "defaults to the dam_safety storage-band bounds used in training."
    ),
    "dam_safety_pct_days_below_min_storage": "Fraction of days storage is below the minimum storage bound.",
    "dam_safety_pct_days_above_max_storage": "Fraction of days storage is above the maximum storage bound.",
    "dam_safety_max_storage_range_water_year_af": "Maximum (max-min) storage range within any water year (AF). Water year starts Oct 1.",

    # --- ESA minimum flow ---
    "esa_min_flow_pct_days_met": (
        "Fraction of days ESA minimum flow is met. Preferred computation: (animas_farmington_q_cfs + "
        "sanjuan_release_cfs) >= threshold (default 500 cfs). Fallback: uses rc_esa_min_flow.baseline when physical "
        "columns are unavailable."
    ),

    # --- Flooding ---
    "flooding_pct_days_met": (
        "Fraction of days flooding constraints are satisfied. Preferred computation mirrors flooding_baseline: "
        "sj_at_farmington_cfs < 5000 AND (sj_at_farmington_lag2_cfs is NaN OR < 12000). Fallback: uses "
        "rc_flooding.baseline when physical columns are unavailable."
    ),

    # --- Hydropower ---
    "hydropower_pct_of_historic": (
        "Total agent generation as a percent of historic generation over the test rollout: "
        "100 * sum(hydro_agent_mwh) / sum(hydro_hist_mwh). Returns NaN if historic sum is 0 or columns missing."
    ),

    # --- NIIP ---
    "niip_pct_days_demand_met_in_window": (
        "Fraction of days within the NIIP active window that the daily NIIP demand is met (delivery >= demand). "
        "By default, the window is DOY 50–300 (inclusive) with demand>0. Demand is computed from the NIIP demand "
        "curve (or from a niip_demand_cfs column if present)."
    ),
    "niip_annual_volume_pct_of_contract": (
        "Mean across calendar years of 100*(delivered seasonal volume / contractual seasonal volume), where volumes are "
        "computed by integrating daily CFS values over the NIIP window (cfs-day -> acre-feet). Contract volume is the "
        "integral under the NIIP demand curve for the days included in each year."
    ),

    # --- Dynamic reward-component summaries (pattern) ---
    "sum_rc_*": (
        "Pattern: sum_rc_<objective>.<variant> = sum of that reward component over the test rollout. Emitted for every "
        "column in df_test starting with 'rc_'."
    ),
    "mean_rc_*": (
        "Pattern: mean_rc_<objective>.<variant> = mean of that reward component over the test rollout. Emitted for every "
        "column in df_test starting with 'rc_'."
    ),
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def compute_metrics(
    df_test: pd.DataFrame,
    *,
    which: str | Sequence[str] | None = "core",
    metric_kwargs: Mapping[str, Mapping[str, object]] | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Compute one or more metrics from a test rollout dataframe.

    Parameters
    ----------
    df_test
        Test-period rollout dataframe (from DRLModel.evaluate_test()).
    which
        - "all": all metrics in METRIC_REGISTRY
        - name of a single metric (e.g. "rewards_summary")
        - name of a group (e.g. "core", "storage")
        - list of metric and/or group names
    metric_kwargs
        Optional dict mapping metric-name -> kwargs dict passed to that metric
        function.
    validate
        If True, run basic checks on df_test and required columns.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame of scalar metrics.
    """
    if metric_kwargs is None:
        metric_kwargs = {}

    keys = _resolve_metric_keys(which)
    if validate:
        validate_rollout_df(df_test)
        _validate_required_columns(df_test, keys)

    out: dict[str, float] = {}
    for name in keys:
        spec = METRIC_REGISTRY[name]
        kw = dict(metric_kwargs.get(name, {}))
        res = spec.func(df_test, **kw)
        if not isinstance(res, dict):
            raise TypeError(f"Metric {name!r} must return dict[str, float], got {type(res)}")

        # Normalize to floats where possible (leave NaN as float)
        for k, v in res.items():
            try:
                out[k] = float(v)  # type: ignore[arg-type]
            except Exception:
                out[k] = float("nan")

    return pd.DataFrame([out])


def save_metrics(
    *,
    df_test: pd.DataFrame,
    outdir: Path | str,
    which: str | Sequence[str] | None = "core",
    metric_kwargs: Mapping[str, Mapping[str, object]] | None = None,
    stem: str = "eval_metrics",
    validate: bool = True,
) -> dict[str, Path]:
    """Compute and save metrics to CSV (and JSON for convenience)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dfm = compute_metrics(
        df_test,
        which=which,
        metric_kwargs=metric_kwargs,
        validate=validate,
    )

    csv_path = outdir / f"{stem}.csv"
    json_path = outdir / f"{stem}.json"

    dfm.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dfm.iloc[0].to_dict(), f, indent=2, sort_keys=True)

    return {"csv": csv_path, "json": json_path}


def collect_run_metrics(
    run_dirs: Sequence[Path | str],
    *,
    metrics_filename: str = "eval_metrics.csv",
    rollout_filename: str = "eval_test_rollout.parquet",
    which_if_missing: str | Sequence[str] | None = "core",
) -> pd.DataFrame:
    """Collect metrics across many run directories."""
    rows: list[pd.DataFrame] = []
    for rd in run_dirs:
        run_dir = Path(rd)
        p_metrics = run_dir / metrics_filename
        if p_metrics.exists():
            dfm = pd.read_csv(p_metrics)
        else:
            p_roll = run_dir / rollout_filename
            if not p_roll.exists():
                continue
            df_roll = pd.read_parquet(p_roll)
            dfm = compute_metrics(df_roll, which=which_if_missing)

        dfm.insert(0, "run_dir", str(run_dir))
        rows.append(dfm)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def validate_rollout_df(df_test: pd.DataFrame) -> None:
    """Basic sanity checks on the rollout dataframe."""
    if not isinstance(df_test, pd.DataFrame):
        raise TypeError("df_test must be a pandas DataFrame")
    if not isinstance(df_test.index, pd.DatetimeIndex):
        raise ValueError("df_test.index must be a pandas.DatetimeIndex")
    if df_test.index.has_duplicates:
        raise ValueError("df_test.index contains duplicate timestamps")
    if not df_test.index.is_monotonic_increasing:
        raise ValueError("df_test.index must be monotonically increasing")


# -----------------------------------------------------------------------------
# Metric registry
# -----------------------------------------------------------------------------


MetricFunc = Callable[..., dict[str, float]]


@dataclass(frozen=True)
class MetricSpec:
    func: MetricFunc
    requires: tuple[str, ...]


def _resolve_metric_keys(which: str | Sequence[str] | None) -> list[str]:
    if which is None:
        which = "core"

    if isinstance(which, str):
        if which == "all":
            return list(METRIC_REGISTRY.keys())
        items = [w.strip() for w in which.split(",") if w.strip()]
    else:
        items = list(which)

    selected: list[str] = []
    for item in items:
        if item == "all":
            selected.extend(list(METRIC_REGISTRY.keys()))
        elif item in METRIC_GROUPS:
            selected.extend(list(METRIC_GROUPS[item]))
        elif item in METRIC_REGISTRY:
            selected.append(item)
        else:
            raise KeyError(f"Unknown metric or group: {item!r}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for k in selected:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _validate_required_columns(df_test: pd.DataFrame, metric_keys: Sequence[str]) -> None:
    missing: dict[str, list[str]] = {}
    for name in metric_keys:
        req = METRIC_REGISTRY[name].requires
        if not req:
            continue
        miss = [c for c in req if c not in df_test.columns]
        if miss:
            missing[name] = miss
    if missing:
        parts = [f"{k}: {v}" for k, v in missing.items()]
        raise KeyError("Missing required columns for metrics: " + "; ".join(parts))


# -----------------------------------------------------------------------------
# Metric implementations (starter set)
# -----------------------------------------------------------------------------


def _metric_rewards_summary(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if "reward" in df.columns:
        out["total_reward"] = float(df["reward"].sum())
        out["mean_reward"] = float(df["reward"].mean())
    return out


def _metric_reward_components_summary(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    rc_cols = [c for c in df.columns if c.startswith("rc_")]
    for col in rc_cols:
        key = col[len("rc_") :]
        out[f"sum_rc_{key}"] = float(df[col].sum())
        out[f"mean_rc_{key}"] = float(df[col].mean())
    return out


def _metric_action_saturation(df: pd.DataFrame, *, sat: float = 0.99) -> dict[str, float]:
    out: dict[str, float] = {}
    a_cols = [c for c in ("action_0", "action_1") if c in df.columns]
    if not a_cols:
        return out
    for c in a_cols:
        s = df[c].astype(float)
        out[f"pct_{c}_saturated"] = float((np.abs(s) >= float(sat)).mean())
    if len(a_cols) == 2:
        s0 = df[a_cols[0]].astype(float)
        s1 = df[a_cols[1]].astype(float)
        out["pct_any_action_saturated"] = float(((np.abs(s0) >= sat) | (np.abs(s1) >= sat)).mean())
    return out


def _metric_release_constraint_binding(df: pd.DataFrame, *, eps: float = 1e-6) -> dict[str, float]:
    """How often the requested release was capped/limited."""
    out: dict[str, float] = {}

    if "requested_total_release_cfs" in df.columns and "release_agent_cfs" in df.columns:
        req = df["requested_total_release_cfs"].astype(float)
        act = df["release_agent_cfs"].astype(float)
        out["pct_release_capped_or_limited"] = float((req > (act + eps)).mean())

    if "release_cap_penalty" in df.columns:
        out["pct_release_cap_penalty_pos"] = float((df["release_cap_penalty"].astype(float) > 0.0).mean())
        out["mean_release_cap_penalty"] = float(df["release_cap_penalty"].astype(float).mean())
    if "release_phys_penalty" in df.columns:
        out["pct_release_phys_penalty_pos"] = float((df["release_phys_penalty"].astype(float) > 0.0).mean())
        out["mean_release_phys_penalty"] = float(df["release_phys_penalty"].astype(float).mean())

    return out


def _objective_is_active(df: pd.DataFrame, objective: str) -> bool:
    """Return True if the rollout includes reward-component columns for an objective.

    We use presence of `rc_<objective>.*` columns as the indicator that the
    experiment's reward spec included that objective.
    """
    prefix = f"rc_{objective}."
    return any(str(c).startswith(prefix) for c in df.columns)


# -----------------------------------------------------------------------------
# Objective-aligned metrics (one-per-objective first pass)
# -----------------------------------------------------------------------------


def _metric_dam_safety_pct_days_within_storage_bounds(
    df: pd.DataFrame,
    *,
    low_af: float | None = None,
    high_af: float | None = None,
    storage_col: str = "storage_agent_af_end",
) -> dict[str, float]:
    """Dam safety: % of days storage is within [min_storage_af, max_storage_af].

    Defaults:
      - If df contains `min_storage_af` / `max_storage_af`, use those.
      - Else fall back to the band used in `dam_safety_storage_band`.
    """
    out: dict[str, float] = {}

    # If the experiment did not include dam_safety, report NA.
    if not _objective_is_active(df, "dam_safety"):
        out["dam_safety_pct_days_within_storage_bounds"] = float("nan")
        return out

    col = storage_col if storage_col in df.columns else "storage_agent_af"
    if col not in df.columns:
        out["dam_safety_pct_days_within_storage_bounds"] = float("nan")
        return out

    s = df[col].astype(float)

    if low_af is None:
        if "min_storage_af" in df.columns:
            try:
                low_af = float(df["min_storage_af"].iloc[0])
            except Exception:
                low_af = None
        if low_af is None:
            low_af = 500_000.0

    if high_af is None:
        if "max_storage_af" in df.columns:
            try:
                high_af = float(df["max_storage_af"].iloc[0])
            except Exception:
                high_af = None
        if high_af is None:
            high_af = 1_731_750.0

    lo = float(low_af)
    hi = float(high_af)
    within = (s >= lo) & (s <= hi)
    out["dam_safety_pct_days_within_storage_bounds"] = float(within.mean())
    return out


def _metric_dam_safety_storage_detail(
    df: pd.DataFrame,
    *,
    low_af: float | None = None,
    high_af: float | None = None,
    storage_col: str = "storage_agent_af_end",
) -> dict[str, float]:
    """Extra dam-safety diagnostics (kept out of the default core set)."""
    out: dict[str, float] = {}

    # If the experiment did not include dam_safety, report NA.
    if not _objective_is_active(df, "dam_safety"):
        return {
            "dam_safety_pct_days_below_min_storage": float("nan"),
            "dam_safety_pct_days_above_max_storage": float("nan"),
            "dam_safety_max_storage_range_water_year_af": float("nan"),
        }

    col = storage_col if storage_col in df.columns else "storage_agent_af"
    if col not in df.columns:
        return out

    s = df[col].astype(float)

    if low_af is None and "min_storage_af" in df.columns:
        try:
            low_af = float(df["min_storage_af"].iloc[0])
        except Exception:
            low_af = None
    if high_af is None and "max_storage_af" in df.columns:
        try:
            high_af = float(df["max_storage_af"].iloc[0])
        except Exception:
            high_af = None

    if low_af is None:
        low_af = 500_000.0
    if high_af is None:
        high_af = 1_731_750.0

    lo = float(low_af)
    hi = float(high_af)
    out["dam_safety_pct_days_below_min_storage"] = float((s < lo).mean())
    out["dam_safety_pct_days_above_max_storage"] = float((s > hi).mean())

    # Max within-water-year storage range (AF)
    idx = df.index
    wy = idx.year + (idx.month >= 10).astype(int)
    ranges = s.groupby(wy).max() - s.groupby(wy).min()
    out["dam_safety_max_storage_range_water_year_af"] = float(ranges.max()) if len(ranges) else float("nan")

    return out


def _metric_esa_min_flow_pct_days_met(
    df: pd.DataFrame,
    *,
    threshold_cfs: float = 500.0,
    animas_col: str = "animas_farmington_q_cfs",
    release_col: str = "sanjuan_release_cfs",
) -> dict[str, float]:
    """ESA minimum flow: % of days (Animas @ Farmington + San Juan release) >= threshold."""
    out: dict[str, float] = {}

    # If the experiment did not include esa_min_flow, report NA.
    if not _objective_is_active(df, "esa_min_flow"):
        out["esa_min_flow_pct_days_met"] = float("nan")
        return out

    # Prefer the physical columns; fall back to reward component if needed.
    if (animas_col in df.columns) and (release_col in df.columns):
        animas = df[animas_col].astype(float)
        release = df[release_col].astype(float)
        met = (animas.fillna(0.0) + release.fillna(0.0)) >= float(threshold_cfs)
        out["esa_min_flow_pct_days_met"] = float(met.mean())
        return out

    # Fallback: if reward component exists and is binary, mean corresponds to % met.
    for rc_col in ("rc_esa_min_flow.baseline", "rc_esa_min_flow"):
        if rc_col in df.columns:
            s = df[rc_col].astype(float)
            out["esa_min_flow_pct_days_met"] = float((s > 0.5).mean())
            return out

    out["esa_min_flow_pct_days_met"] = float("nan")
    return out


def _metric_flooding_pct_days_met(
    df: pd.DataFrame,
    *,
    same_day_thresh_cfs: float = 5000.0,
    lag2_thresh_cfs: float = 12000.0,
    q0_col: str = "sj_at_farmington_cfs",
    qlag2_col: str = "sj_at_farmington_lag2_cfs",
) -> dict[str, float]:
    """Flooding: % of days both flood constraints are satisfied.

    Mirrors `flooding_baseline` logic:
      - q0 < 5000
      - qlag2 < 12000 (or treated as safe if missing/NaN early)
    """
    out: dict[str, float] = {}

    # If the experiment did not include flooding, report NA.
    if not _objective_is_active(df, "flooding"):
        out["flooding_pct_days_met"] = float("nan")
        return out

    if q0_col not in df.columns:
        # Fallback to reward component if available
        for rc_col in ("rc_flooding.baseline", "rc_flooding"):
            if rc_col in df.columns:
                s = df[rc_col].astype(float)
                out["flooding_pct_days_met"] = float((s >= 0.999).mean())
                return out
        out["flooding_pct_days_met"] = float("nan")
        return out

    q0 = df[q0_col].astype(float)
    if qlag2_col in df.columns:
        q2 = df[qlag2_col].astype(float)
        lag2_safe = q2.isna() | (q2 < float(lag2_thresh_cfs))
    else:
        lag2_safe = pd.Series(True, index=df.index)

    met = (q0 < float(same_day_thresh_cfs)) & lag2_safe
    out["flooding_pct_days_met"] = float(met.mean())
    return out


def _metric_hydropower_pct_of_historic(
    df: pd.DataFrame,
    *,
    agent_col: str = "hydro_agent_mwh",
    hist_col: str = "hydro_hist_mwh",
) -> dict[str, float]:
    """Hydropower: total generation relative to historic (percent).

    Computed as 100 * sum(agent) / sum(historic) over the test period.
    """
    out: dict[str, float] = {}

    # If the experiment did not include hydropower, report NA.
    if not _objective_is_active(df, "hydropower"):
        out["hydropower_pct_of_historic"] = float("nan")
        return out
    if agent_col not in df.columns or hist_col not in df.columns:
        out["hydropower_pct_of_historic"] = float("nan")
        return out

    a = df[agent_col].astype(float)
    h = df[hist_col].astype(float)
    denom = float(np.nansum(h.values))
    if denom == 0.0:
        out["hydropower_pct_of_historic"] = float("nan")
        return out

    out["hydropower_pct_of_historic"] = float(100.0 * np.nansum(a.values) / denom)
    return out


def _niip_get_delivery_series(
    df: pd.DataFrame,
    *,
    delivery_col: str = "niip_release_cfs",
    fallback_cols: tuple[str, ...] = ("release_comp2_cfs",),
) -> pd.Series | None:
    """Return the NIIP delivery series in CFS, or None if not available."""
    if delivery_col in df.columns:
        return df[delivery_col].astype(float)
    for c in fallback_cols:
        if c in df.columns:
            return df[c].astype(float)
    return None


def _niip_get_demand_series(
    df: pd.DataFrame,
    *,
    demand_col: str = "niip_demand_cfs",
) -> pd.Series | None:
    """Return the NIIP demand series in CFS.

    Preference order:
      1) Use a precomputed column (default: niip_demand_cfs) if present.
      2) Compute from the NIIP demand curve (niip_daily_demand(doy)) using df.index.

    Returns None if demand cannot be obtained.
    """
    if demand_col in df.columns:
        try:
            return df[demand_col].astype(float)
        except Exception:
            return None

    # Compute from demand curve (import lazily; may fail if data files missing).
    try:
        from deepreservoir.define_env.niip.niip_demand import niip_daily_demand
    except Exception:
        return None

    doys = df.index.dayofyear.to_numpy()
    try:
        vals = niip_daily_demand(doys)  # supports ndarray in this repo
        arr = np.asarray(vals, dtype=float)
    except Exception:
        # fallback: scalar calls
        try:
            arr = np.asarray([float(niip_daily_demand(int(d))) for d in doys], dtype=float)
        except Exception:
            return None

    arr = np.clip(arr, 0.0, None)
    return pd.Series(arr, index=df.index, name=demand_col)


def _metric_niip_delivery_and_volume(
    df: pd.DataFrame,
    *,
    doy_start: int = 50,
    doy_end: int = 300,
    demand_col: str = "niip_demand_cfs",
    delivery_col: str = "niip_release_cfs",
    demand_positive_eps: float = 1e-9,
    tol_cfs: float = 0.0,
) -> dict[str, float]:
    """NIIP metrics.

    Emits:
      - niip_pct_days_demand_met_in_window : fraction of active-window days demand is met (delivery >= demand)
      - niip_annual_volume_pct_of_contract : mean across calendar years of 100*(delivered volume / contractual volume)

    The NIIP active window is defined as DOY in [doy_start, doy_end] (inclusive) and demand > 0.
    Demand is taken from `demand_col` if present; otherwise computed from the NIIP demand curve.
    """
    out = {
        "niip_pct_days_demand_met_in_window": float("nan"),
        "niip_annual_volume_pct_of_contract": float("nan"),
    }

    # If the experiment did not include niip, report NA.
    if not _objective_is_active(df, "niip"):
        return out

    demand = _niip_get_demand_series(df, demand_col=demand_col)
    delivery = _niip_get_delivery_series(df, delivery_col=delivery_col)
    if demand is None or delivery is None:
        return out

    doys = df.index.dayofyear
    in_window = (doys >= int(doy_start)) & (doys <= int(doy_end))
    active = in_window & demand.notna() & (demand.astype(float) > float(demand_positive_eps))
    n_active = int(active.sum())
    if n_active == 0:
        return out

    # 1) % of time within the active window that daily demand is met
    met = delivery.astype(float) >= (demand.astype(float) - float(tol_cfs))
    out["niip_pct_days_demand_met_in_window"] = float(met[active].mean())

    # 2) Annual volume (delivered / contract), averaged across calendar years
    years = df.index.year
    ratios: list[float] = []
    for y in np.unique(years):
        m = active & (years == y)
        if not bool(m.any()):
            continue

        delivered_af = float(np.nansum(delivery[m].astype(float).to_numpy()) * _CFS_DAY_TO_ACRE_FEET)
        contract_af = float(np.nansum(demand[m].astype(float).to_numpy()) * _CFS_DAY_TO_ACRE_FEET)
        if contract_af <= 0.0:
            continue
        ratios.append(delivered_af / contract_af)

    if ratios:
        out["niip_annual_volume_pct_of_contract"] = float(100.0 * float(np.mean(ratios)))

    return out


# -----------------------------------------------------------------------------
# Registry + groups
# -----------------------------------------------------------------------------


METRIC_REGISTRY: dict[str, MetricSpec] = {
    # Rewards
    "rewards_summary": MetricSpec(func=_metric_rewards_summary, requires=("reward",)),
    "reward_components_summary": MetricSpec(func=_metric_reward_components_summary, requires=()),

    # Objective-aligned summary metrics
    "dam_safety": MetricSpec(func=_metric_dam_safety_pct_days_within_storage_bounds, requires=()),
    "esa_min_flow": MetricSpec(func=_metric_esa_min_flow_pct_days_met, requires=()),
    "flooding": MetricSpec(func=_metric_flooding_pct_days_met, requires=()),
    "hydropower": MetricSpec(func=_metric_hydropower_pct_of_historic, requires=()),
    "niip": MetricSpec(func=_metric_niip_delivery_and_volume, requires=()),

    # Optional extra objective diagnostics
    "dam_safety_detail": MetricSpec(func=_metric_dam_safety_storage_detail, requires=()),

    # Actions / constraints (diagnostic)
    "action_saturation": MetricSpec(func=_metric_action_saturation, requires=()),
    "release_constraint_binding": MetricSpec(func=_metric_release_constraint_binding, requires=()),
}


METRIC_GROUPS: dict[str, tuple[str, ...]] = {
    # Minimal experiment scoreboard: one metric per objective (+ total/mean reward).
    "core": (
        "rewards_summary",
        "dam_safety",
        "esa_min_flow",
        "flooding",
        "hydropower",
        "niip",
    ),
    "rewards": ("rewards_summary", "reward_components_summary"),
    "objectives": ("dam_safety", "esa_min_flow", "flooding", "hydropower", "niip"),
    "dam_safety_detail": ("dam_safety_detail",),
    "actions": ("action_saturation", "release_constraint_binding"),
}
