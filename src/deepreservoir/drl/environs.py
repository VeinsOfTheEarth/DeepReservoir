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

m  = project_metadata()

# 1 cfs sustained over a day to acre-feet
CFS_TO_AF_PER_DAY = 1.98211


class NavajoReservoirEnv(Env):
    """
    Navajo Reservoir environment (daily timestep).

    DATA / STATE (raw units)
    ------------------------
    Internal state:
        storage_af          : current simulated storage [acre-feet]

    Required columns in data_raw (index = datetime):
        storage_af          : historical storage [acre-feet] (used for initialization only)
        inflow_cfs          : inflow to reservoir [cubic feet per second]
        evap_af             : evaporation loss from reservoir [acre-feet]
        elev_ft (optional)  : historical elevation [feet] (not required; we compute elevation from storage)

    data_norm:
        Same index/columns as data_raw, normalized with norm_stats.
        - storage_af        : standard normal variate (we recompute from simulated storage)
        - other cols        : standard normal variate
        - doy               : day-of-year normalized as doy / 366

    OBSERVATION (normalized)
    ------------------------
    obs_cols (in order):
        storage_af          : normalized storage (from simulated storage_af)
        inflow_cfs          : normalized inflow
        sj_farmington_q_cfs : normalized flow at Farmington (if present)
        sj_bluff_q_cfs      : normalized flow at Bluff (if present)
        doy                 : day-of-year / 366

    ACTION (daily releases)
    -----------------------
    action: np.array shape (2,), each in [-1, 1]
        action[0]  → San Juan mainstem release, mapped to sanjuan_release_cfs [cfs]
        action[1]  → NIIP diversion release, mapped to niip_release_cfs [cfs]

    MASS BALANCE (AF)
    -----------------
    S(t+1) = S(t) + inflow_af - evap_af - total_release_af
        inflow_af         = inflow_cfs * CFS_TO_AF_PER_DAY
        total_release_af  = (sanjuan_release_cfs + niip_release_cfs) * CFS_TO_AF_PER_DAY

    INFO DICT KEYS (per step)
    -------------------------
        date                      : pd.Timestamp, current date
        storage_af                : new simulated storage [acre-feet]
        prev_storage_af           : previous storage [acre-feet]
        inflow_cfs                : inflow [cfs]
        inflow_af                 : inflow [acre-feet/day]
        evap_af                   : evaporation [acre-feet/day]
        sanjuan_release_cfs       : San Juan release [cfs]
        niip_release_cfs          : NIIP release [cfs]
        total_release_cfs         : combined release [cfs]
        total_release_af          : combined release [acre-feet/day]
        elev_ft                   : reservoir elevation [feet] (computed from storage)
        min_storage_af            : lower bound of “safe” storage band [acre-feet]
        max_storage_af            : upper bound of “safe” storage band [acre-feet]
        raw_forcings              : full raw row from data_raw (pd.Series)
    """

    metadata = {"render_modes": []}  # Gymnasium-style; unused for now

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
        #  action[0] -> sanjuan_release_cfs
        #  action[1] -> niip_release_cfs
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Create storage for computing 2-day lagged discharge at Bluff
        self.sj_at_farmington_history = deque(maxlen=3)

        # OBSERVATION SPACE: pick normalized columns you want agent to see
        obs_cols = []
        for col in [
            "storage_af",
            "inflow_cfs",
            "sj_farmington_q_cfs",
            "sj_bluff_q_cfs",
            "doy",
        ]:
            if col in self.data_norm.columns:
                obs_cols.append(col)
        self.obs_cols = obs_cols
        self.obs_dim = len(self.obs_cols)
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Load elevation–area–capacity models built within build_interpolating_models.py
        with open(m.path('elev_area_storage_pickle'), "rb") as f:
            elev_models = pickle.load(f)

        # Interpolators built from CSV:
        #   "capacity_to_elevation": capacity [ac-ft] -> elevation [ft]
        #   others exist if you ever need them: elevation_to_capacity, etc.
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
        self.t = 0                         # step index within episode
        self.start_idx = 0                # where this episode starts in the time series
        self.storage_af: float | None = None  # current simulated storage [AF]
        self.episode_step_count = 0

        # Optional bookkeeping
        self.last_reward_breakdown: dict[str, float] | None = None

        # Episode-level reward tracking (for diagnostics)
        # keys match CompositeReward keys, e.g. "niip.baseline"
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
        Build the observation vector at the CURRENT time index using:
        - simulated storage_af (normalized),
        - exogenous normalized forcings from data_norm,
        - normalized day-of-year.

        obs[i] units (conceptually):
            storage_af          -> standard normal variate of storage [AF]
            inflow_cfs          -> standard normal variate of inflow [cfs]
            sj_*_q_cfs          -> standard normal variate of gage flow [cfs]
            doy                 -> day_of_year / 366 in [0, 1]
        """
        idx = self._current_global_idx()
        date = self.dates[idx]

        # storage_norm from simulated storage_af, not historical
        storage_mean = self.norm_stats.loc["storage_af", "mean"]
        storage_std = self.norm_stats.loc["storage_af", "std"]
        storage_norm = (self.storage_af - storage_mean) / storage_std

        row_norm = self.data_norm.iloc[idx]

        vals: list[float] = []
        for col in self.obs_cols:
            if col == "storage_af":
                vals.append(float(storage_norm))
            elif col == "doy":
                vals.append(date.dayofyear / 366.0)
            else:
                vals.append(float(row_norm[col]))
        return np.asarray(vals, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Decide where this episode starts
        if self.is_eval or self.episode_length is None:
            # Evaluation: always start from the beginning, full series
            self.start_idx = 0
            self.max_steps = self.n_steps
        else:
            # Training: random window of length episode_length
            max_start = self.n_steps - self.episode_length
            # self.np_random is provided by gymnasium after super().reset(seed)
            self.start_idx = int(self.np_random.integers(0, max_start + 1))
            self.max_steps = self.episode_length

        self.t = 0
        self.episode_step_count = 0

        # Initialize storage at the chosen start index
        self.storage_af = float(self.data_raw.iloc[self.start_idx]["storage_af"])

        # any other per-episode state
        self.last_reward_breakdown = None
        self.sj_at_farmington_history.clear()

        obs = self._build_obs()
        return obs, {}

    def step(self, action):
        # --- Clip / unpack action (2 outlets in [-1, 1]) ---
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Expected action shape (2,), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        # Current observation BEFORE applying action
        obs = self._build_obs()
        global_idx = self._current_global_idx()
        date = self.dates[global_idx]

        # Map actions to releases [cfs]
        # action ∈ [-1,1] -> frac ∈ [0,1] -> [min_release_cfs, max_release_cfs]
        frac_sanjuan = (action[0] + 1.0) / 2.0
        frac_niip    = (action[1] + 1.0) / 2.0

        sanjuan_release_cfs = self.min_release_cfs + frac_sanjuan * (
            self.max_release_cfs - self.min_release_cfs
        )
        niip_release_cfs = self.min_release_cfs + frac_niip * (
            self.max_release_cfs - self.min_release_cfs
        )

        requested_total_release_cfs = sanjuan_release_cfs + niip_release_cfs

        # --- MASS BALANCE PIECES ---
        row_raw   = self.data_raw.iloc[global_idx]
        inflow_cfs = float(row_raw["inflow_cfs"])
        inflow_af  = inflow_cfs * CFS_TO_AF_PER_DAY
        evap_af    = float(row_raw["evap_af"])

        # Physically available volume this day
        available_af = max(self.storage_af + inflow_af - evap_af, 0.0)
        max_total_release_cfs_phys = available_af / CFS_TO_AF_PER_DAY

        # # Also respect your engineered cap
        # max_total_release_cfs_phys = min(max_total_release_cfs_phys,
        #                                 2.0 * self.max_release_cfs)

        # --- PROJECT REQUESTED RELEASE INTO FEASIBLE SET ---
        scale_penalty = 0.0
        if requested_total_release_cfs > max_total_release_cfs_phys:
            if requested_total_release_cfs > 0.0:
                scale = max_total_release_cfs_phys / requested_total_release_cfs
            else:
                scale = 0.0
            sanjuan_release_cfs *= scale
            niip_release_cfs    *= scale
            total_release_cfs   = sanjuan_release_cfs + niip_release_cfs

            # Optional: how “bad” was the request?
            scale_penalty = (requested_total_release_cfs - total_release_cfs) / (
                2.0 * self.max_release_cfs + 1e-6
            )
        else:
            total_release_cfs = requested_total_release_cfs

        total_release_af = total_release_cfs * CFS_TO_AF_PER_DAY
        new_storage_af   = self.storage_af + inflow_af - evap_af - total_release_af
        new_storage_af   = max(new_storage_af, 0.0)


        # Animas at Farmington [cfs]
        animas_cfs = float(row_raw["animas_farmington_q_cfs"])

        # San Juan @ Farmington = Animas + mainstem release [cfs]
        sj_at_farm_cfs = animas_cfs + sanjuan_release_cfs

        # Update history for lag-2
        self.sj_at_farmington_history.append(sj_at_farm_cfs)
        if len(self.sj_at_farmington_history) >= 3:
            sj_at_farm_lag2_cfs = self.sj_at_farmington_history[-3]  # two days ago
        else:
            sj_at_farm_lag2_cfs = None


        # --- Mass balance update in AF ---
        new_storage_af = self.storage_af + inflow_af - evap_af - total_release_af
        new_storage_af = max(new_storage_af, 0.0)  # enforce non-negative storage

        # Elevation based on new storage using capacity_to_elevation model
        new_elev_ft = float(self.capacity_to_elev(new_storage_af))

        # Hydropower generation [MWh/day] from mainstem release + elevation
        hydropower_mwh = navajo_power_generation_model(
            cfs_values=sanjuan_release_cfs,
            elevation_ft=new_elev_ft,
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
            "prev_storage_af": self.storage_af,
            "inflow_cfs": inflow_cfs,
            "inflow_af": inflow_af,
            "evap_af": evap_af,
            "sanjuan_release_cfs": sanjuan_release_cfs,
            "niip_release_cfs": niip_release_cfs,
            "total_release_cfs": total_release_cfs,
            "total_release_af": total_release_af,
            "elev_ft": new_elev_ft,
            "sj_at_farmington_cfs": sj_at_farm_cfs,
            "sj_at_farmington_lag2_cfs": sj_at_farm_lag2_cfs,
            "min_storage_af": self.min_storage_af,
            "max_storage_af": self.max_storage_af,
            "raw_forcings": row_raw,
            "hydropower_mwh" : hydropower_mwh,
            "release_scale_penalty": scale_penalty,
            "spring_wy": wy,       # SPR
            "spring_oi": oi_val,   # SPR opportunity index: ∈ [0,1] or NaN if out of range
            "spring_go": go_val,   # SPR opportunity index - binary: bool
        }

        # Advance internal state
        self.storage_af = new_storage_af
        self.t += 1
        self.episode_step_count += 1

        done = (
            self._current_global_idx() >= self.n_steps
            or self.episode_step_count >= self.episode_length
        )
        terminated = done
        truncated = False  # could distinguish time-limit truncation later

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
            # breakdown already includes weights from CompositeReward
            self._episode_reward_sums[key] += float(val)

        # Expose everything via info for logging/diagnostics
        info["reward_components_step"] = breakdown                     # this step
        info["reward_components_episode"] = dict(self._episode_reward_sums)  # cumulative
        info["episode_total_reward"] = self._episode_total_reward

        # Backwards-compatible alias if you were already using this
        info["reward_components"] = breakdown

        return next_obs, float(total_reward), terminated, truncated, info
