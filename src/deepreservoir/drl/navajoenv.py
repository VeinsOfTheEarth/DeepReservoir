# navajo_rl/env/navajo_env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from loader import NavajoData

HydroFn = Callable[[pd.Timestamp, float, float], float]   # (date, release_cms, elev_m) -> MWh
NIIPFn  = Callable[[int], float]                          # (doy) -> demand_m3_d

@dataclass
class ActionSpec:
    names: Sequence[str]
    low: Sequence[float]
    high: Sequence[float]

class NavajoEnv(gym.Env):
    """
    A minimal, testable reservoir env:
      - Observations built via NavajoData.build_state()
      - Actions are physical releases (m3/day) within bounds
      - Mass balance in SI units
      - Rewards delegated to a CompositeReward object
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        data: NavajoData,
        table: str,
        state_sources: Mapping[str, Iterable[str]],
        action_spec: Mapping[str, Sequence[float]],
        episode_length: int,
        reward,
        use_normalized_obs: bool = True,
        hydropower_fn: Optional[HydroFn] = None,
        niip_demand_fn: Optional[NIIPFn] = None,
    ):
        super().__init__()
        self.data = data
        self.table = table
        self.state_sources = state_sources
        self.episode_length = episode_length
        self.reward_obj = reward
        self.use_normalized_obs = use_normalized_obs
        self.hydropower_fn = hydropower_fn
        self.niip_demand_fn = niip_demand_fn

        self.df = self.data.get(self.table)
        self.index = self.df.index

        self.action_names = list(action_spec["names"])
        self.action_low = np.array(action_spec["low"], dtype=np.float32)
        self.action_high = np.array(action_spec["high"], dtype=np.float32)
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)

        # Obs dims from one built state row
        probe_t = self.index[0]
        obs0 = self._build_obs(probe_t)
        self.obs_columns = list(obs0.index)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.obs_columns),), dtype=np.float32
        )

        # Runtime
        self._t_idx = 0
        self._t_ep = 0
        self._storage_m3 = float(self.df["storage_m3"].iloc[0])

        # Logging
        self.history = {"episode_rewards": []}
        self._episode_rewards: List[float] = []
        self._traj_rows: List[dict] = []

    # ---------- SB3 helpers ---------- #

    def seed(self, seed: Optional[int] = None):
        np.random.seed(seed)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        # start episodes anywhere within the train split window
        train = self.data.get_split("train", self.table)
        start_idx = np.random.randint(0, len(train) - self.episode_length)
        self._t_idx = self.df.index.get_indexer([train.index[start_idx]])[0]
        self._t_ep = 0
        self._episode_rewards.clear()
        self._traj_rows.clear()

        self._storage_m3 = float(self.df["storage_m3"].iloc[self._t_idx])
        obs = self._build_obs(self.index[self._t_idx])
        return obs.values.astype(np.float32), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_low, self.action_high).astype(float)
        t = self.index[self._t_idx]
        row = self.df.iloc[self._t_idx]

        # Mass balance (SI volumes per day)
        release_total_m3_d = float(action.sum())
        inflow_m3_d = float(row.get("inflow_cms", 0.0)) * 86400.0
        evap_m3_d = float(row.get("evap_cms", 0.0)) * 86400.0
        next_storage_m3 = max(0.0, self._storage_m3 + inflow_m3_d - evap_m3_d - release_total_m3_d)

        # Elevation (optional; use hydrology models if you have them)
        elev_m = float(row.get("elev_m", np.nan))

        # Optional domain callbacks
        hydropower_mwh = None
        if self.hydropower_fn is not None and np.isfinite(elev_m):
            hydropower_mwh = float(self.hydropower_fn(t, release_total_m3_d / 86400.0, elev_m))  # cms

        niip_demand_m3_d = None
        if self.niip_demand_fn is not None:
            doy = int(row.get("doy", pd.Timestamp(t).dayofyear))
            niip_demand_m3_d = float(self.niip_demand_fn(doy))

        # Reward
        info = {
            "t": t,
            "storage_m3": self._storage_m3,
            "next_storage_m3": next_storage_m3,
            "inflow_m3_d": inflow_m3_d,
            "evap_m3_d": evap_m3_d,
            "release_total_m3_d": release_total_m3_d,
            "elev_m": elev_m,
            "hydropower_mwh": hydropower_mwh,
            "niip_demand_m3_d": niip_demand_m3_d,
        }
        r = float(self.reward_obj(state=None, action=action, next_state=None, info=info))
        self._episode_rewards.append(r)

        # Log one trajectory row
        self._traj_rows.append({**info, "reward": r})

        # Advance
        self._storage_m3 = next_storage_m3
        self._t_idx += 1
        self._t_ep += 1

        terminated = self._t_idx >= len(self.index) - 1 or self._t_ep >= self.episode_length
        truncated = False

        if terminated:
            self.history["episode_rewards"].append(sum(self._episode_rewards))

        obs = self._build_obs(self.index[self._t_idx]) if not terminated else np.zeros(len(self.obs_columns))
        return obs.astype(np.float32), r, terminated, truncated, {}

    # ---------- Helpers ---------- #

    def _build_obs(self, t: pd.Timestamp) -> pd.Series:
        return self.data.build_state(
            at=t,
            sources=self.state_sources,
            window=None,
            agg="last",
            normalize=self.use_normalized_obs,
        )

    def rollout_over_split(self, split: str, policy) -> pd.DataFrame:
        """Rollout a deterministic policy over an entire split; return trajectory DataFrame."""
        df = self.data.get_split(split, self.table)
        # Hard reset at the start of the split
        self._t_idx = self.df.index.get_indexer([df.index[0]])[0]
        self._t_ep = 0
        self._episode_rewards.clear()
        self._traj_rows.clear()
        self._storage_m3 = float(self.df["storage_m3"].iloc[self._t_idx])

        for _ in range(len(df)):
            obs = self._build_obs(self.index[self._t_idx]).values.astype(np.float32)
            action, _ = policy.predict(obs, deterministic=True)
            self.step(action)[0]  # ignore returned obs here

            if self._t_idx >= self.df.index.get_indexer([df.index[-1]])[0]:
                break

        return pd.DataFrame(self._traj_rows)
