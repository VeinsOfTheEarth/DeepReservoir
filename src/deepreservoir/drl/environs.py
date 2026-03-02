# deepreservoir/drl/environs.py
import pickle
import numpy as np
import pandas as pd
from gymnasium import Env
from collections import deque, defaultdict
from gymnasium.spaces import Box

from deepreservoir.drl.rewards import RewardContext
from deepreservoir.data.metadata import project_metadata
from deepreservoir.define_env.hydropower_model import navajo_power_generation_model
from deepreservoir.define_env.spring_peak_release.opportunity_index import (
    OIParams,
    precompute_oi_by_wy,
)

m = project_metadata()

# 1 cfs sustained over a day to acre-feet
CFS_TO_AF_PER_DAY = 1.98211

# -----------------------------------------------------------------------------
# Navajo Reservoir physical thresholds (authoritative elevations)
# -----------------------------------------------------------------------------
# NOTE: These are enforced in the environment physics (deadpool release blocking
# and automatic spill). Storage thresholds are derived from the E–S curve.
NAVAJO_DEADPOOL_ELEV_FT = 5775.0
NAVAJO_SPILL_ELEV_FT = 6085.0


class NavajoReservoirEnv(Env):
    """Navajo Reservoir environment (daily timestep).

    One agent, two continuous actions (Box[-1, 1], shape=(2,)):
      - action[0] -> release_sj_main_cfs
      - action[1] -> release_niip_cfs

    Important physical constraints implemented here:
      - Per-outlet capacity caps:
          release_sj_main_cfs <= max_release_sj_main_cfs (default 5000)
          release_niip_cfs    <= max_release_niip_cfs    (default 2500)
        If the agent requests more than the cap, we hard-cap (no proportional rescaling).

      - Deadpool constraint (elevation-based):
          If starting reservoir elevation is at/below NAVAJO_DEADPOOL_ELEV_FT (5775 ft),
          *no releases are physically possible*, so both outlet releases are forced to 0.

      - Spill constraint (elevation-based):
          If ending reservoir elevation would exceed NAVAJO_SPILL_ELEV_FT (6085 ft),
          the excess storage is automatically spilled (uncontrolled) to the San Juan mainstem.

      - Water-available constraint:
          Even above deadpool, releases cannot exceed the water available that day.
          If total requested release exceeds available water, both outlets are scaled down
          proportionally to satisfy mass balance (this preserves the requested split).

    Observation (fixed order, Colab-aligned):
      [ storage_norm, evap_norm, inflow_norm, doy ]
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
        max_release_sj_main_cfs: float = 5000.0,
        max_release_niip_cfs: float = 2500.0,
        # NOTE: legacy arg retained so callers don't break; it is no longer used
        # to *define* deadpool. Deadpool is defined by NAVAJO_DEADPOOL_ELEV_FT.
        deadpool_storage_af: float = 500_000.0,
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

        # Episode length (in steps); default full series
        self.episode_length = (
            int(episode_length) if episode_length is not None else self.n_steps
        )

        # Release limits (per-outlet) in cfs
        self.min_release_cfs = float(min_release_cfs)
        self.max_release_sj_main_cfs = float(max_release_sj_main_cfs)
        self.max_release_niip_cfs = float(max_release_niip_cfs)

        # Deadpool/spill are defined by *elevation*; storage thresholds are derived
        # from the elevation-storage relationship for convenience/clamping.
        self.deadpool_elev_ft = float(NAVAJO_DEADPOOL_ELEV_FT)
        self.spill_elev_ft = float(NAVAJO_SPILL_ELEV_FT)

        # Keep the provided value only as a legacy/debug field (not authoritative).
        self._deadpool_storage_af_legacy = float(deadpool_storage_af)

        # ACTION SPACE: 2 releases in [-1, 1]
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # For optional lagged discharge proxy
        self.sj_at_farmington_history = deque(maxlen=3)

        # ---- Observation schema checks ----
        # We REQUIRE raw columns for mass balance:
        required_raw = ["storage_af", "inflow_cfs", "evap_af"]
        for c in required_raw:
            if c not in self.data_raw.columns:
                raise KeyError(f"data_raw is missing required column '{c}'")

        # We REQUIRE normalized columns for observations (inflow/evap):
        required_norm = ["inflow_cfs", "evap_af"]
        for c in required_norm:
            if c not in self.data_norm.columns:
                raise KeyError(
                    f"data_norm is missing required column '{c}'. "
                    f"To include evaporation in observation, ensure preprocessing outputs 'evap_af' in data_norm."
                )

        # Fixed schema across train/eval
        self.obs_cols = ["storage_af", "evap_af", "inflow_cfs", "doy"]
        self.obs_dim = len(self.obs_cols)

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Load elevation model
        with open(m.path("elev_area_storage_pickle"), "rb") as f:
            elev_models = pickle.load(f)
        # Elevation-area-storage interpolators (2019 table).
        # We only *require* capacity→elevation, but we also keep elevation→capacity when available.
        self.capacity_to_elev = elev_models["capacity_to_elevation"]
        self.elev_to_capacity = elev_models.get("elevation_to_capacity", None)

        # Ensure we have elevation -> capacity for spill clamping.
        if self.elev_to_capacity is None:
            # Build a numerical inverse from capacity->elevation as a last resort.
            # This is robust and keeps the environment runnable even if the pickle
            # only contains capacity->elevation.
            caps = np.linspace(0.0, 2_000_000.0, 20001)
            elevs = np.asarray(self.capacity_to_elev(caps), dtype=float)
            order = np.argsort(elevs)
            elevs_sorted = elevs[order]
            caps_sorted = caps[order]

            def _elev_to_capacity(elev_ft):
                elev_arr = np.asarray(elev_ft, dtype=float)
                return np.interp(elev_arr, elevs_sorted, caps_sorted)

            self.elev_to_capacity = _elev_to_capacity

        # Storage thresholds derived from the E–S curve (useful for clamping/debug).
        self.deadpool_storage_af = float(self.elev_to_capacity(self.deadpool_elev_ft))
        self.max_storage_af = float(self.elev_to_capacity(self.spill_elev_ft))

        # --- SPR Opportunity Index precompute ---
        pm = project_metadata()
        params_path = pm.path("params.spr_oi_params_json")
        self.spring_oi_params: OIParams = OIParams.load(params_path)

        _df_wy = precompute_oi_by_wy(self.data_raw, self.spring_oi_params)
        self._spring_oi_by_wy = _df_wy["oi"]
        self._spring_go_by_wy = _df_wy["go"]

        _wy_by_day = pd.Series(
            (self.data_raw.index.year + (self.data_raw.index.month >= 10)).astype(int),
            index=self.data_raw.index,
            name="wy",
        )
        _oi_map = self._spring_oi_by_wy.to_dict()
        _go_map = self._spring_go_by_wy.to_dict()

        self.spring_oi_daily = _wy_by_day.map(_oi_map).astype(float)
        _go_daily = _wy_by_day.map(_go_map).astype("boolean").fillna(False)
        self.spring_go_daily = _go_daily.astype(bool)

        # Internal state
        self.t = 0
        self.start_idx = 0
        self.storage_af: float | None = None
        self.episode_step_count = 0

        self.last_reward_breakdown: dict[str, float] | None = None

        # Episode bookkeeping
        self._episode_reward_sums: dict[str, float] = defaultdict(float)
        self._episode_total_reward: float = 0.0

    # ---------------- Helpers ----------------

    def _current_global_idx(self) -> int:
        return self.start_idx + self.t

    def _current_date(self) -> pd.Timestamp:
        return self.dates[self._current_global_idx()]

    def _norm_storage(self, storage_af: float) -> float:
        storage_mean = float(self.norm_stats.loc["storage_af", "mean"])
        storage_std = float(self.norm_stats.loc["storage_af", "std"])
        if storage_std == 0.0:
            storage_std = 1.0
        return (float(storage_af) - storage_mean) / storage_std

    def _build_obs(self) -> np.ndarray:
        idx = self._current_global_idx()
        date = self.dates[idx]
        row_norm = self.data_norm.iloc[idx]

        storage_norm = self._norm_storage(float(self.storage_af))

        # Fixed order: storage, evap, inflow, doy
        obs = np.array(
            [
                storage_norm,
                float(row_norm["evap_af"]),
                float(row_norm["inflow_cfs"]),
                float(date.dayofyear) / 366.0,
            ],
            dtype=np.float32,
        )
        return obs

    # ---------------- Gym API ----------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.is_eval:
            self.start_idx = 0
            self.max_steps = self.n_steps
        else:
            max_start = max(self.n_steps - self.episode_length, 0)
            self.start_idx = int(self.np_random.integers(0, max_start + 1))
            self.max_steps = self.episode_length

        self.t = 0
        self.episode_step_count = 0

        # init storage at episode start
        self.storage_af = float(self.data_raw.iloc[self.start_idx]["storage_af"])
        # Enforce physical spill level at reset as well (historical series should not exceed this,
        # but small inconsistencies/rounding can otherwise start an episode above the spill level).
        self.storage_af = float(min(self.storage_af, self.max_storage_af))


        self.last_reward_breakdown = None
        self.sj_at_farmington_history.clear()

        self._episode_reward_sums = defaultdict(float)
        self._episode_total_reward = 0.0

        obs = self._build_obs()
        return obs, {}

    def step(self, action):
        # ---- action shaping ----
        action = np.asarray(action, dtype=np.float32).squeeze()
        if action.ndim != 1 or action.shape[0] != 2:
            raise ValueError(f"Expected action shape (2,), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        obs = self._build_obs()
        global_idx = self._current_global_idx()
        date = self.dates[global_idx]

        # Map actions to nonnegative per-outlet releases (cfs)
        frac_sj = (action[0] + 1.0) / 2.0
        frac_niip = (action[1] + 1.0) / 2.0

        requested_release_sj_main_cfs = self.min_release_cfs + frac_sj * (
            self.max_release_sj_main_cfs - self.min_release_cfs
        )
        requested_release_niip_cfs = self.min_release_cfs + frac_niip * (
            self.max_release_niip_cfs - self.min_release_cfs
        )

        # --- Deadpool constraint: no releases below deadpool ---
        # We implement this in *elevation* space (outlet intake constraint), using the
        # starting elevation of the step. This is conservative: if inflow during the day
        # would raise the reservoir above deadpool, releases become possible on the next step.
        start_elev_ft = float(self.capacity_to_elev(float(self.storage_af)))
        deadpool_block = start_elev_ft <= float(self.deadpool_elev_ft)

        if deadpool_block:
            release_sj_main_cfs = 0.0
            release_niip_cfs = 0.0
            cap_penalty = 0.0
        else:
            # --- Per-outlet hard caps ---
            release_sj_main_cfs = float(
                min(requested_release_sj_main_cfs, self.max_release_sj_main_cfs)
            )
            release_niip_cfs = float(min(requested_release_niip_cfs, self.max_release_niip_cfs))

            # Aggregate cap-penalty = fraction of requested flow clipped by outlet caps
            clipped = (requested_release_sj_main_cfs - release_sj_main_cfs) + (
                requested_release_niip_cfs - release_niip_cfs
            )
            cap_penalty = float(max(clipped, 0.0)) / (
                (self.max_release_sj_main_cfs + self.max_release_niip_cfs) + 1e-9
            )

        total_cfs = float(release_sj_main_cfs + release_niip_cfs)

        # --- Physical feasibility (water available) ---
        row_raw = self.data_raw.iloc[global_idx]
        inflow_cfs = float(row_raw["inflow_cfs"])
        inflow_af = inflow_cfs * CFS_TO_AF_PER_DAY
        evap_af = float(row_raw["evap_af"])

        available_af = max(float(self.storage_af) + inflow_af - evap_af, 0.0)
        max_total_cfs_phys = available_af / CFS_TO_AF_PER_DAY

        phys_penalty = 0.0
        if total_cfs > max_total_cfs_phys:
            pre_phys_total = float(total_cfs)
            scale = max_total_cfs_phys / (total_cfs + 1e-9)
            release_sj_main_cfs *= scale
            release_niip_cfs *= scale
            total_cfs = float(release_sj_main_cfs + release_niip_cfs)

            phys_penalty = (pre_phys_total - float(total_cfs)) / (
                (self.max_release_sj_main_cfs + self.max_release_niip_cfs) + 1e-9
            )

        # --- Mass balance update ---
        # --- Mass balance update (controlled releases) ---
        controlled_total_cfs = float(total_cfs)
        controlled_total_af = controlled_total_cfs * CFS_TO_AF_PER_DAY
        new_storage_af = float(self.storage_af) + inflow_af - evap_af - controlled_total_af
        new_storage_af = max(new_storage_af, 0.0)

        # --- Automatic spill (physical): any water above the spill level is released ---
        # We model this as an uncontrolled spill that goes to the San Juan mainstem (not NIIP).
        spill_af = 0.0
        spill_cfs = 0.0
        if new_storage_af > float(self.max_storage_af):
            spill_af = float(new_storage_af - float(self.max_storage_af))
            spill_cfs = float(spill_af / CFS_TO_AF_PER_DAY)
            new_storage_af = float(self.max_storage_af)

        # Totals including spill (actual outflow at the dam)
        sj_main_flow_cfs = float(release_sj_main_cfs + spill_cfs)
        total_cfs = float(sj_main_flow_cfs + release_niip_cfs)
        total_af = float(controlled_total_af + spill_af)

        # Optional Farmington proxy
        animas_cfs = (
            float(row_raw["animas_farmington_q_cfs"])
            if "animas_farmington_q_cfs" in row_raw.index
            else 0.0
        )

        # IMPORTANT: mainstem outlet contributes to Farmington; NIIP does not.
        sj_at_farm_cfs = animas_cfs + float(sj_main_flow_cfs)

        self.sj_at_farmington_history.append(sj_at_farm_cfs)
        sj_at_farm_lag2_cfs = (
            self.sj_at_farmington_history[-3]
            if len(self.sj_at_farmington_history) >= 3
            else None
        )

        # Elevation + hydropower
        new_elev_ft = float(self.capacity_to_elev(new_storage_af))
        hydropower_mwh = navajo_power_generation_model(
            cfs_values=float(release_sj_main_cfs),
            elevation_ft=float(new_elev_ft),
        )

        # SPR
        wy = int(date.year + (date.month >= 10))
        oi_val = float(self.spring_oi_daily.get(date, np.nan))
        go_val = bool(self.spring_go_daily.get(date, False))

        info = {
            "date": date,
            "storage_af": float(new_storage_af),
            "prev_storage_af": float(self.storage_af),
            "prev_elev_ft": float(start_elev_ft),
            "inflow_cfs": float(inflow_cfs),
            "inflow_af": float(inflow_af),
            "evap_af": float(evap_af),
            # Releases (unambiguous names + units)
            "release_sj_main_cfs": float(release_sj_main_cfs),
            "release_niip_cfs": float(release_niip_cfs),
            "total_release_cfs": float(total_cfs),
            "total_release_af": float(total_af),
            "spill_cfs": float(spill_cfs),
            "spill_af": float(spill_af),
            "sj_main_flow_cfs": float(sj_main_flow_cfs),
            "total_controlled_release_cfs": float(controlled_total_cfs),
            "total_controlled_release_af": float(controlled_total_af),
            # Helpful debugging context
            "requested_release_sj_main_cfs": float(requested_release_sj_main_cfs),
            "requested_release_niip_cfs": float(requested_release_niip_cfs),
            "max_release_sj_main_cfs": float(self.max_release_sj_main_cfs),
            "max_release_niip_cfs": float(self.max_release_niip_cfs),
            "deadpool_storage_af": float(self.deadpool_storage_af),
            "deadpool_elev_ft": float(self.deadpool_elev_ft),
            "spill_elev_ft": float(self.spill_elev_ft),
            "deadpool_block": bool(deadpool_block),
            "elev_ft": float(new_elev_ft),
            "sj_at_farmington_cfs": float(sj_at_farm_cfs),
            "sj_at_farmington_lag2_cfs": None
            if sj_at_farm_lag2_cfs is None
            else float(sj_at_farm_lag2_cfs),
            "max_storage_af": float(self.max_storage_af),
            "raw_forcings": row_raw,
            "hydropower_mwh": float(hydropower_mwh),
            # Penalties (optional for rewards/diagnostics)
            "release_cap_penalty": float(cap_penalty),
            "release_phys_penalty": float(phys_penalty),
            # SPR
            "spring_wy": wy,
            "spring_oi": oi_val,
            "spring_go": go_val,
        }

        # Advance
        self.storage_af = float(new_storage_af)
        self.t += 1
        self.episode_step_count += 1

        global_idx_next = self._current_global_idx()
        done = (global_idx_next >= self.n_steps) or (
            self.episode_step_count >= self.episode_length
        )

        terminated = bool(done)
        truncated = False

        next_obs = self._build_obs() if not done else np.zeros_like(obs, dtype=np.float32)

        # Reward
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

        # Episode bookkeeping
        self._episode_total_reward += float(total_reward)
        for k, v in breakdown.items():
            self._episode_reward_sums[k] += float(v)

        info["reward_components_step"] = breakdown
        info["reward_components_episode"] = dict(self._episode_reward_sums)
        info["episode_total_reward"] = float(self._episode_total_reward)
        info["reward_components"] = breakdown

        return next_obs, float(total_reward), terminated, truncated, info