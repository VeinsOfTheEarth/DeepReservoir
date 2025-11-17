# navajo_rl/rewards/components.py
from __future__ import annotations
from dataclasses import dataclass
from .base import RewardComponent

@dataclass
class StorageBandReward(RewardComponent):
    min_m3: float
    max_m3: float
    prefer_mid: bool = True

    def __call__(self, state, action, next_state, info) -> float:
        s = float(info["next_storage_m3"])
        if s < self.min_m3 or s > self.max_m3:
            return -1.0
        if not self.prefer_mid:
            return 0.5
        mid = 0.5 * (self.min_m3 + self.max_m3)
        span = max(1.0, self.max_m3 - self.min_m3)
        return 1.0 - abs(s - mid) / span

@dataclass
class MinFlowReward(RewardComponent):
    key_release_m3_d: str
    min_m3_d: float
    ok_reward: float = 0.5
    penalty: float = -0.2

    def __call__(self, state, action, next_state, info) -> float:
        r = float(info[self.key_release_m3_d])
        return self.ok_reward if r >= self.min_m3_d else self.penalty

@dataclass
class HydropowerReward(RewardComponent):
    min_mwh: float = 0.0
    max_mwh: float = 800.0

    def __call__(self, state, action, next_state, info) -> float:
        mwh = info.get("hydropower_mwh", None)
        if mwh is None:
            return 0.0
        if mwh < self.min_mwh:
            return -0.5
        return (float(mwh) - self.min_mwh) / max(1.0, self.max_mwh - self.min_mwh)

@dataclass
class NIIPDemandReward(RewardComponent):
    key_release_m3_d: str
    key_demand_m3_d: str

    def __call__(self, state, action, next_state, info) -> float:
        d = info.get(self.key_demand_m3_d, None)
        if d is None or d <= 0:
            return 0.0
        r = float(info[self.key_release_m3_d])
        return 1.0 if r >= d else r / d
