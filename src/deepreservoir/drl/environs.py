import pickle
import numpy as np
import pandas as pd
from gymnasium import Env
from collections import deque, defaultdict
from gymnasium.spaces import Box

from deepreservoir.drl.rewards import RewardContext
from deepreservoir.data.metadata import project_metadata
from deepreservoir.define_env.hydropower_model import navajo_power_generation_model
from deepreservoir.define_env.spring_peak_release.opportunity_index import OIParams, precompute_oi_by_wy

m = project_metadata()

# 1 cfs sustained over a day to acre-feet
CFS_TO_AF_PER_DAY = 1.98211


class NavajoReservoirEnv(Env):
    """
    Navajo Reservoir environment (daily timestep).

    ACTION (daily releases)
    -----------------------
    action: np.array shape (2,), each in [-1, 1]
        action[0]  → San Juan mainstem release, mapped to sanjuan_release_cfs [cfs]
        action[1]  → NIIP diversion release, mapped to niip_release_cfs [cfs]

    OBSERVATION (normalized)
    ------------------------
    obs_cols (in order):
        storage_af          : normalized storage (from simulated storage_af)
        inflow_cfs          : normalized inflow
        evap_af             : normalized evaporation (if present)   <-- added
        sj_farmington_q_cfs : normalized flow at Farmington (if present)
        sj_bluff_q_cfs      : normalized flow at Bluff (if present)
        doy                 : day-of-year / 366
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_raw: pd.DataFrame,
        data_norm: pd.DataFrame,
        norm_stats: pd.DataFrame,
        reward_fn,
        episode_length: int | None = None,
        min_release_cfs: float = 0.0,
        max_release_cfs: float = 5000.0,
        is_eval: bool = False,
    ):
        super().__init__()
        assert data_raw.index.equals(data_norm.index)

        self.data_raw = data_raw
        self.data_norm = data_norm
        self.norm_stats = norm_stats
        self.reward_fn = reward_fn
        self.is_eval = is_eval

        self.dates = self.data_raw.index.to_list()
        self.n_steps = len(self.dates)

        # Episode length (in steps); default is full series
        self.episode_length = episode_length or self.n_steps

        # Release limits (per outlet) in cfs
        self.min_release_cfs = float(min_release_cfs)
        self.max_release_cfs = float(max_release_cfs)

        # Storage safety band (raw AF) – used by dam_safety rewards
        self.min_storage_af = 500_000.0
        self.max_storage_af = 1_731_750.0

        # ACTION SPACE: 2 releases in [-1, 1]
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Create storage for computing 2-day lagged discharge at Farmington proxy
        self.sj_at_farmington_history = deque(maxlen=3)

        # OBSERVATION SPACE: choose normalized columns available to agent
        # NOTE: storage_af is recomputed from simulated storage, not taken from data_norm
        obs_cols = []

        # include these if present in normalized data
        for col in [
            "storage_af",
            "inflow_cfs",
            "evap_af",        # <-- added
            # optionally support evap_cfs if someone later changes preprocessing
            "evap_cfs",
            "sj_farmington_q_cfs",
            "sj_bluff_q_cfs",
            "doy",
        ]:
            if col in self.data_norm.columns and col not in obs_cols:
                obs_cols.append(col)

        # If both evap_af and evap_cfs exist, prefer evap_af (drop evap_cfs)
        if "evap_af" in obs_cols and "evap_cfs" in obs_cols:
            obs_cols.remove("evap_cfs")

        self.obs_cols = obs_cols
        self.obs_dim = len(self.obs_cols)

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Load elevation–area–capacity models built within build_interpolating_models.py
        with open(m.path("elev_area_storage_pickle"), "rb") as f:
            elev_models = pickle.load(f)
        self.capacity_to_elev = elev_models["capacity_to_elevation"]

        # --- SPR Opportunity Index: load params from metadata and precompute ---
        pm = project_metadata()
        params_path = pm.path("params.spr_oi_params_json")

        # Load tuned boundary + OI mapping parameters
        self.spring_oi_params: OIParams = OIParams.load(params_path)

        # Compute per-water-year OI/GO using current model data
        _df_wy = precompute_oi_by_wy(self.data_raw, self.spring_oi_params)
        self._spring_oi_by_wy = _df_wy["oi"]
        self._spring_go_by_wy = _df_wy["go"]

        # Build daily-aligned series so step() lookup is trivial
        _wy_by_day = (self.data_raw.index.year + (self.data_raw.index.month >= 10)).astype(int)
        _oi_map = self._spring_oi_by_wy.to_dict()
        _go_map = self._spring_go_by_wy.to_dict()

        self.spring_oi_daily = pd.Series(_wy_by_day).map(_oi_map).astype(float)
        self.spring_go_daily = pd.Series(_wy_by_day).map(_go_map).astype("boolean")
        self.spring_oi_daily.index = self.data_raw.index
        self.spring_go_daily.index = self.data_raw.index


        # Internal state
        self.t = 0
        self.start_idx = 0
        self.storage_af: float | None = None
        self.episode_step_count = 0

        # Optional bookkeeping
        self.last_reward_breakdown: dict[str, float] | None = None

        # Episode-level reward tracking (for diagnostics)
        self._episode_reward_sums: dict[str, float] = defaultdict(float)
        self._episode_total_reward: float = 0.0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _current_global_idx(self) -> int:
        return self.start_idx + self.t

    def _current_date(self) -> pd.Timestamp:
        return self.dates[self._current_global_idx()]

    def _build_obs(self) -> np.ndarray:
        """
        Build observation vector at CURRENT index.

        - storage_af: derived from simulated storage_af (normalized)
        - other forcings: read from data_norm for that day
        - doy: date.dayofyear / 366
        """
        idx = self._current_global_idx()
        date = self.dates[idx]

        # Storage normalized from simulated storage_af
        storage_mean = float(self.norm_stats.loc["storage_af", "mean"])
        storage_std = float(self.norm_stats.loc["storage_af", "std"])
        # guard against weird std
        if storage_std == 0.0:
            storage_std = 1.0
        storage_norm = (float(self.storage_af) - storage_mean) / storage_std

        row_norm = self.data_norm.iloc[idx]

        vals: list[float] = []
        for col in self.obs_cols:
            if col == "storage_af":
                vals.append(float(storage_norm))
            elif col == "doy":
                vals.append(float(date.dayofyear) / 366.0)
            else:
                vals.append(float(row_norm[col]))

        return np.asarray(vals, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Decide where this episode starts
        # NOTE: if episode_length is full series, random windows don't matter much,
        # but we keep the logic consistent.
        if self.is_eval:
            self.start_idx = 0
            self.max_steps = self.n_steps
        else:
            # Training: random window if episode_length < n_steps
            max_start = max(self.n_steps - self.episode_length, 0)
            self.start_idx = int(self.np_random.integers(0, max_start + 1))
            self.max_steps = self.episode_length

        self.t = 0
        self.episode_step_count = 0

        # Initialize storage at the chosen start index
        self.storage_af = float(self.data_raw.iloc[self.start_idx]["storage_af"])

        # Reset per-episode caches
        self.last_reward_breakdown = None
        self.sj_at_farmington_history.clear()

        # IMPORTANT: reset episode reward accumulators
        self._episode_reward_sums = defaultdict(float)
        self._episode_total_reward = 0.0

        obs = self._build_obs()
        return obs, {}

    def step(self, action):
        # --- robust action shaping ---
        action = np.asarray(action, dtype=np.float32).squeeze()
        if action.ndim != 1 or action.shape[0] != 2:
            raise ValueError(f"Expected action shape (2,), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        # Current observation BEFORE applying action
        obs = self._build_obs()
        global_idx = self._current_global_idx()
        date = self.dates[global_idx]

        # Map actions to releases [cfs]
        frac_sanjuan = (action[0] + 1.0) / 2.0
        frac_niip = (action[1] + 1.0) / 2.0

        sanjuan_release_cfs = self.min_release_cfs + frac_sanjuan * (
            self.max_release_cfs - self.min_release_cfs
        )
        niip_release_cfs = self.min_release_cfs + frac_niip * (
            self.max_release_cfs - self.min_release_cfs
        )

        requested_total_release_cfs = sanjuan_release_cfs + niip_release_cfs

        # --- MASS BALANCE PIECES ---
        row_raw = self.data_raw.iloc[global_idx]
        inflow_cfs = float(row_raw["inflow_cfs"])
        inflow_af = inflow_cfs * CFS_TO_AF_PER_DAY
        evap_af = float(row_raw["evap_af"])

        # Physically available volume this day
        available_af = max(float(self.storage_af) + inflow_af - evap_af, 0.0)
        max_total_release_cfs_phys = available_af / CFS_TO_AF_PER_DAY

        # --- PROJECT REQUESTED RELEASE INTO FEASIBLE SET ---
        scale_penalty = 0.0
        if requested_total_release_cfs > max_total_release_cfs_phys:
            scale = (
                max_total_release_cfs_phys / requested_total_release_cfs
                if requested_total_release_cfs > 0.0
                else 0.0
            )
            sanjuan_release_cfs *= scale
            niip_release_cfs *= scale
            total_release_cfs = sanjuan_release_cfs + niip_release_cfs

            # Optional: how “bad” was the request?
            scale_penalty = (requested_total_release_cfs - total_release_cfs) / (
                2.0 * self.max_release_cfs + 1e-6
            )
        else:
            total_release_cfs = requested_total_release_cfs

        # --- Mass balance update in AF ---
        total_release_af = total_release_cfs * CFS_TO_AF_PER_DAY
        new_storage_af = float(self.storage_af) + inflow_af - evap_af - total_release_af
        new_storage_af = max(new_storage_af, 0.0)

        # Farmington proxy and lag-2
        animas_cfs = float(row_raw["animas_farmington_q_cfs"])
        sj_at_farm_cfs = animas_cfs + sanjuan_release_cfs

        self.sj_at_farmington_history.append(sj_at_farm_cfs)
        if len(self.sj_at_farmington_history) >= 3:
            sj_at_farm_lag2_cfs = self.sj_at_farmington_history[-3]
        else:
            sj_at_farm_lag2_cfs = None

        # Elevation based on new storage using capacity_to_elevation model
        new_elev_ft = float(self.capacity_to_elev(new_storage_af))

        # Hydropower generation [MWh/day] from mainstem release + elevation
        hydropower_mwh = navajo_power_generation_model(
            cfs_values=float(sanjuan_release_cfs),
            elevation_ft=float(new_elev_ft),
        )

        # Current simulation timestamp. Replace `date` with your step's timestamp variable if different.
        # For example, if you track a pointer `self.t`, you might use: date = self.data_raw.index[self.t]
        date = date  # <-- use the same variable you already use to read today's row

        wy = int(date.year + (date.month >= 10))
        oi_val = float(self.spring_oi_daily.get(date, np.nan))
        go_val = bool(self.spring_go_daily.get(date, False))

        # Build info dict for rewards/logging
        info = {
            "date": date,
            "storage_af": new_storage_af,
            "prev_storage_af": float(self.storage_af),
            "inflow_cfs": inflow_cfs,
            "inflow_af": inflow_af,
            "evap_af": evap_af,
            "sanjuan_release_cfs": float(sanjuan_release_cfs),
            "niip_release_cfs": float(niip_release_cfs),
            "total_release_cfs": float(total_release_cfs),
            "total_release_af": float(total_release_af),
            "elev_ft": float(new_elev_ft),
            "sj_at_farmington_cfs": float(sj_at_farm_cfs),
            "sj_at_farmington_lag2_cfs": None if sj_at_farm_lag2_cfs is None else float(sj_at_farm_lag2_cfs),
            "min_storage_af": float(self.min_storage_af),
            "max_storage_af": float(self.max_storage_af),
            "raw_forcings": row_raw,
            "hydropower_mwh": float(hydropower_mwh),
            "release_scale_penalty": float(scale_penalty),
            "spring_wy": wy,       # SPR
            "spring_oi": oi_val,   # SPR opportunity index: ∈ [0,1] or NaN if out of range
            "spring_go": go_val,   # SPR opportunity index - binary: bool
        }

        # Advance internal state
        self.storage_af = new_storage_af
        self.t += 1
        self.episode_step_count += 1

        # Done logic: stop if we hit end-of-series or episode length
        global_idx_next = self._current_global_idx()
        done = (global_idx_next >= self.n_steps) or (self.episode_step_count >= self.episode_length)

        terminated = bool(done)
        truncated = False

        # Next observation (SB3 doesn’t care if obs is dummy at done)
        if not done:
            next_obs = self._build_obs()
        else:
            next_obs = np.zeros_like(obs, dtype=np.float32)

        # Reward via composite reward function
        ctx = RewardContext(
            t=global_idx,
            date=date,
            obs=obs,
            action=action,
            next_obs=next_obs,
            info=info,
        )
        total_reward, breakdown = self.reward_fn(ctx)
        self.last_reward_breakdown = breakdown

        # --- Episode-level reward bookkeeping ---
        self._episode_total_reward += float(total_reward)
        for key, val in breakdown.items():
            self._episode_reward_sums[key] += float(val)

        # Expose everything via info for logging/diagnostics
        info["reward_components_step"] = breakdown
        info["reward_components_episode"] = dict(self._episode_reward_sums)
        info["episode_total_reward"] = float(self._episode_total_reward)

        # Backwards-compatible alias
        info["reward_components"] = breakdown

        return next_obs, float(total_reward), terminated, truncated, info
