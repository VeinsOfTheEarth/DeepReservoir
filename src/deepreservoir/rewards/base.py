# navajo_rl/rewards/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple

class RewardComponent:
    """Contract: __call__(state, action, next_state, info) -> float"""
    def __call__(self, state, action, next_state, info) -> float:
        raise NotImplementedError

@dataclass
class Weighted:
    name: str
    comp: RewardComponent
    weight: float = 1.0

class CompositeReward(RewardComponent):
    def __init__(self, components: List[Weighted]):
        self.components = components

    def __call__(self, state, action, next_state, info) -> float:
        total = 0.0
        for w in self.components:
            total += w.weight * float(w.comp(state, action, next_state, info))
        return total
