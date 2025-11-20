# deepreservoir/drl/rewards.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple, Optional, Mapping

import numpy as np
import pandas as pd

from deepreservoir.define_env.niip.niip_demand import niip_daily_demand


# ---------------------------------------------------------------------
# Types & context
# ---------------------------------------------------------------------

# You can adjust this later as your env API solidifies.
@dataclass
class RewardContext:
    """
    Information needed to compute reward for a single step.

    Feel free to extend this with whatever you want:
    - raw storage/release/inflows
    - normalized obs/next_obs
    - date, water year, etc.
    """
    t: int                       # step index
    date: pd.Timestamp           # current date
    obs: np.ndarray              # current observation (normalized)
    action: np.ndarray           # current action
    next_obs: np.ndarray         # next observation (normalized)
    info: Dict[str, Any]         # extra per-step info (raw series, flags, etc.)


RewardFn = Callable[[RewardContext], float]


# ---------------------------------------------------------------------
# Registry plumbing
# ---------------------------------------------------------------------

OBJECTIVES = [
    "dam_safety",
    "esa_min_flow",
    "esa_spring_peak_release",
    "flooding",
    "hydropower",
    "niip",
    "physics",
]

# OBJECTIVE -> { variant_name -> reward_fn }
REWARD_REGISTRY: Dict[str, Dict[str, RewardFn]] = {obj: {} for obj in OBJECTIVES}


def register_reward(objective: str, variant: str) -> Callable[[RewardFn], RewardFn]:
    """
    Decorator to register a reward function under an objective + variant name.

    Example:
        @register_reward("hydropower", "baseline")
        def hydropower_baseline(ctx: RewardContext) -> float:
            ...
    """
    if objective not in REWARD_REGISTRY:
        raise KeyError(f"Unknown objective {objective!r}. Expected one of {OBJECTIVES}.")

    def decorator(fn: RewardFn) -> RewardFn:
        if variant in REWARD_REGISTRY[objective]:
            raise ValueError(f"Reward already registered for {objective}:{variant}")
        REWARD_REGISTRY[objective][variant] = fn
        return fn

    return decorator


@dataclass
class RewardComponent:
    objective: str
    variant: str
    weight: float
    fn: RewardFn

    @property
    def key(self) -> str:
        # unique identifier, useful for logging
        return f"{self.objective}.{self.variant}"

@dataclass
class RewardDetail:
    total: float
    components: Dict[str, float]   # e.g. {"niip": -0.3, "esa_min_flow": 1.0}


class MultiObjectiveReward:
    def __init__(
        self,
        rewards: Mapping[str, RewardFn],
        weights: Mapping[str, float] | None = None,
    ):
        # names are human labels like "niip", "esa_min_flow"
        self.rewards = dict(rewards)
        self.weights = weights or {name: 1.0 for name in rewards}

    def __call__(self, ctx: "RewardContext") -> RewardDetail:
        components: dict[str, float] = {}
        total = 0.0

        for name, fn in self.rewards.items():
            val = float(fn(ctx))
            components[name] = val
            total += self.weights.get(name, 1.0) * val

        return RewardDetail(total=total, components=components)

class CompositeReward:
    """
    Combines multiple RewardComponents into a single scalar reward.

    Call with a RewardContext; returns (total_reward, component_breakdown_dict).
    """

    def __init__(self, components: List[RewardComponent]):
        self.components = components

    def __call__(self, ctx: RewardContext) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        breakdown: Dict[str, float] = {}

        for comp in self.components:
            r = comp.fn(ctx)
            weighted = comp.weight * r
            breakdown[comp.key] = weighted
            total += weighted

        return total, breakdown

# ---------------------------------------------------------------------
# Building a composite from a user spec
# ---------------------------------------------------------------------

def build_composite_reward(
    spec: Dict[str, str],
    weights: Optional[Dict[str, float]] = None,
) -> CompositeReward:
    """
    Parameters
    ----------
    spec
        Mapping objective -> variant, e.g.
        {
            "dam_safety": "band_penalty",
            "esa_min_flow": "shortage",
            "hydropower": "baseline",
            "niip": "deficit",
        }

    weights
        Optional mapping objective -> scalar weight. If None, all weights = 1.0.

    Returns
    -------
    CompositeReward
        Callable that takes RewardContext and returns (total_reward, breakdown_dict)
    """
    components: List[RewardComponent] = []
    weights = weights or {}

    for objective, variant in spec.items():
        if objective not in REWARD_REGISTRY:
            raise KeyError(
                f"Unknown objective {objective!r}. Known: {list(REWARD_REGISTRY.keys())}"
            )
        variants = REWARD_REGISTRY[objective]
        if variant not in variants:
            raise KeyError(
                f"Unknown variant {variant!r} for objective {objective!r}. "
                f"Known: {list(variants.keys())}"
            )

        fn = variants[variant]
        w = float(weights.get(objective, 1.0))
        components.append(RewardComponent(objective, variant, w, fn))

    return CompositeReward(components)


def parse_objective_spec(spec_str: str) -> Dict[str, str]:
    """
    Parse a CLI-style string into a {objective -> variant} mapping.

    Format:
        "dam_safety:band,esa_min_flow:shortage,hydropower:baseline,niip:deficit"

    No weights here (can add a separate `--weights` arg later if wanted).
    """
    spec: Dict[str, str] = {}
    if not spec_str:
        return spec

    pairs = [s.strip() for s in spec_str.split(",") if s.strip()]
    for token in pairs:
        obj, variant = [p.strip() for p in token.split(":", 1)]
        spec[obj] = variant
    return spec


# ---------------------------------------------------------------------
# Stub reward functions (fill these in as you go)
# ---------------------------------------------------------------------

# Each objective gets at least one variant ("baseline" here).
# Add more variants later (e.g., "soft", "quadratic", "log", etc.)


@register_reward("dam_safety", "baseline")
def dam_safety_baseline(ctx: RewardContext) -> float:
    """
    Reward = 1 if elevation < 6100 ft, else 0.
    """
    elev_ft = ctx.info["elev_ft"] 
    return 1.0 if elev_ft < 6100.0 else 0.0


@register_reward("dam_safety", "storage_band")
def dam_safety_storage_band(ctx: RewardContext) -> float:
    storage = ctx.info["storage_af"]
    s_min   = 500_000
    s_max   = 1_731_750
    target  = 0.5 * (s_min + s_max)

    # Define a symmetric scale so "far" from target is -1, near target is +1
    span = (s_max - s_min) / 2  # half-width of band

    # Relative deviation from target; 0 at target, ±1 at the band edges
    x = (storage - target) / span

    # Parabolic shape: 1 at target, 0 at band edges, negative outside
    r = 1.0 - x**2

    # Clip to [-1, 1] so super-extreme values don't blow up
    r = float(np.clip(r, -1.0, 1.0))

    # Make it matter
    r = 10.0 * r
    return r


@register_reward("niip", "baseline")
def niip_baseline(ctx: RewardContext) -> float:
    """
    NIIP baseline reward: penalize irrigation shortage.

    Logic:
      - Only active during DOY 50–300 (NIIP demand season).
      - Let D = daily NIIP demand [cfs], R = NIIP diversion release [cfs].
      - If R >= D  → reward = 1.0
      - Else       → reward = R / D  (in [0, 1))
      - Outside doy 50–300 → reward = 0.0
    """
    # Use the context date to get day-of-year
    doy = ctx.date.timetuple().tm_yday

    # Outside demand period: no NIIP reward/penalty
    if not (50 <= doy <= 300):
        return 0.0

    # Demand in cfs from spline
    demand_cfs = float(niip_daily_demand(doy))
    if demand_cfs <= 0.0:
        # No demand today → nothing to reward
        return 0.0

    # NIIP release in cfs (from env.step info)
    niip_release_cfs = float(ctx.info["niip_release_cfs"])

    if niip_release_cfs >= demand_cfs:
        return 1.0
    else:
        # Fraction of demand that was actually met
        return max(0.0, niip_release_cfs / demand_cfs)


@register_reward("esa_min_flow", "baseline")
def esa_min_flow_baseline(ctx: RewardContext) -> float:
    """
    ESA minimum flow reward.

    +1 if (Animas at Farmington + San Juan release) >= 500 cfs, else 0.
    """
    row = ctx.info["raw_forcings"]  # pd.Series from data_raw
    animas_cfs = float(row["animas_farmington_q_cfs"])

    sanjuan_release_cfs = float(ctx.info["sanjuan_release_cfs"])

    total_flow_cfs = animas_cfs + sanjuan_release_cfs # SJ @ Farmington is approximated by this summation that's based on Agent's actions

    return 1.0 if total_flow_cfs >= 500.0 else 0.0


@register_reward("flooding", "baseline")
def flooding_baseline(ctx: RewardContext) -> float:
    """
    Flooding baseline reward with two components:

      1. Same-day San Juan @ Farmington flow:
         Q0 = sj_at_farmington_cfs = Animas + San Juan release
         -> +1 if Q0 < 5,000 cfs, else 0.

      2. Two-day-lagged flow:
         Qlag2 = sj_at_farmington_lag2_cfs (if available)
         -> +1 if Qlag2 < 12,000 cfs, else 0.
         (If lag-2 is not yet available early in the episode, we treat it as safe.)

    Total reward = average of the two components (0, 0.5, or 1.0).
    """
    q0 = float(ctx.info["sj_at_farmington_cfs"])
    qlag2 = ctx.info.get("sj_at_farmington_lag2_cfs", None)

    # Component 1: same-day threshold
    c1 = 1.0 if q0 < 5000.0 else 0.0

    # Component 2: lag-2 threshold; if we don't have lag-2 yet, treat as safe
    if qlag2 is None:
        c2 = 1.0
    else:
        c2 = 1.0 if float(qlag2) < 12000.0 else 0.0

    # Average so the objective stays in [0, 1]
    return 0.5 * (c1 + c2)


@register_reward("hydropower", "baseline")
def hydropower_baseline(ctx: RewardContext) -> float:
    """
    Hydropower reward based on daily generation.

    Uses:
        ctx.info["hydropower_mwh"]  : hydropower generation [MWh/day]

    Logic (mirrors old code):
        - If generation < min_mwh     → -0.5  (not enough generation)
        - Else                       → linear bonus from 0 to 1 over [min_mwh, max_mwh]
    """
    hydropower_mwh = float(ctx.info["hydropower_mwh"])

    min_mwh = 288.0
    max_mwh = 768.0

    if hydropower_mwh < min_mwh:
        return -0.5  # not enough generation
    else:
        return (hydropower_mwh - min_mwh) / (max_mwh - min_mwh)

@register_reward("physics", "scale_penalty")
def physics_scale_penalty(ctx: RewardContext) -> float:
    # Punish asking for more water than exists
    penalty = float(ctx.info.get("release_scale_penalty", 0.0))
    return -1 * penalty  # tune weight as you like


@register_reward("esa_spring_peak_release", "baseline")
def esa_spring_peak_baseline(ctx: RewardContext) -> float:
    """
    Reward/penalize how well releases match ESA spring peak hydrograph.

    TODO: implement (likely uses multi-day context, but start with per-step).
    """
    raise NotImplementedError
