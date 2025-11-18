# deepreservoir/drl/rewards.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd


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

    No weights here (you can add a separate `--weights` arg later if you want).
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
    """
    Storage-based dam safety reward.

    - If storage is within [min_storage_af, max_storage_af]:
        r = 1 - |S - S_target| / (S_max - S_min)
      where S_target is the midpoint of the band.
    - Otherwise:
        r = -1
    """
    storage = ctx.info["storage_af"]         # current storage [AF]
    s_min   = 500000     # safe min [AF]
    s_max   = 1731750    # safe max [AF]
    target_storage = (s_min + s_max)/2

    if s_min <= storage <= s_max:
        target_storage = 0.5 * (s_max + s_min)
        r = 1.0 - abs(storage - target_storage) / (s_max - s_min)
    else:
        r = -1.0

    return float(r)

@register_reward("esa_min_flow", "baseline")
def esa_min_flow_baseline(ctx: RewardContext) -> float:
    """
    Penalize violation of ESA minimum flow targets.

    TODO: implement using flow at critical gages in ctx.info.
    """
    raise NotImplementedError


@register_reward("esa_spring_peak_release", "baseline")
def esa_spring_peak_baseline(ctx: RewardContext) -> float:
    """
    Reward/penalize how well releases match ESA spring peak hydrograph.

    TODO: implement (likely uses multi-day context, but start with per-step).
    """
    raise NotImplementedError


@register_reward("flooding", "baseline")
def flooding_baseline(ctx: RewardContext) -> float:
    """
    Penalize flows above bankfull or flood thresholds at downstream gages.
    """
    raise NotImplementedError


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


@register_reward("niip", "baseline")
def niip_baseline(ctx: RewardContext) -> float:
    """
    Penalize irrigation shortage (NIIP demand not met).
    """
    raise NotImplementedError
