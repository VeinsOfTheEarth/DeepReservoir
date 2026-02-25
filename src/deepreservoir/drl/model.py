"""
High-level training and evaluation utilities for DeepReservoir DRL.

- loads data via NavajoData
- splits into train / test
- builds reward (from registry)
- builds Gymnasium envs
- trains & evaluates an SB3 agent
- provides helpers to roll out on the test period and compute metrics

Colab equivalence goals:
- Training uses random-window episodes (episode_length_train=3600 by default)
- Testing rolls out the full test period deterministically
- Multi-action PPO: one agent, two continuous actions (handled inside environs.py)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from deepreservoir.data import loader
from deepreservoir.drl import helpers
from deepreservoir.drl import rewards as drl_rewards
from deepreservoir.drl import metrics as drl_metrics
from deepreservoir.drl.environs import NavajoReservoirEnv
from deepreservoir.define_env.hydropower_model import navajo_power_generation_model


# ---------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------
def load_datasets(n_years_test: int) -> Dict[str, pd.DataFrame]:
    nav_data = loader.NavajoData()
    alldata = nav_data.load_all(include_cont_streamflow=False, model_data=True)

    data = alldata["model_data"]              # raw
    datanorm = alldata["model_data_norm"]     # normalized
    norm_stats = alldata["model_norm_stats"]  # mean/std table

    data_train, data_test = helpers.split_train_test_by_water_year(
        data, n_years_test=n_years_test
    )

    datanorm_train = datanorm.loc[data_train.index]
    datanorm_test = datanorm.loc[data_test.index]

    return {
        "train_raw": data_train,
        "test_raw": data_test,
        "train_norm": datanorm_train,
        "test_norm": datanorm_test,
        "norm_stats": norm_stats,
    }


# ---------------------------------------------------------------------
# Agent / reward / env builders
# ---------------------------------------------------------------------
def build_agent(
    env: gym.Env,
    algo: str = "ppo",
    seed: int | None = None,
    *,
    device: str = "auto",
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_epochs: int | None = None,
    gamma: float | None = None,
) -> PPO:
    algo = algo.lower()
    if algo != "ppo":
        raise ValueError(f"Unsupported algo: {algo}")

    ppo_kwargs: dict = {
        "verbose": 1,
        "seed": seed,
        "device": device,
    }
    if n_steps is not None:
        ppo_kwargs["n_steps"] = n_steps
    if batch_size is not None:
        ppo_kwargs["batch_size"] = batch_size
    if n_epochs is not None:
        ppo_kwargs["n_epochs"] = n_epochs
    if gamma is not None:
        ppo_kwargs["gamma"] = gamma

    return PPO("MlpPolicy", env, **ppo_kwargs)


def build_reward(reward_spec_str: str):
    spec = drl_rewards.parse_objective_spec(reward_spec_str)
    composite = drl_rewards.build_composite_reward(spec, weights=None)
    return composite


def make_env(
    data_raw: pd.DataFrame,
    data_norm: pd.DataFrame,
    norm_stats: pd.DataFrame,
    reward_spec_str: str,
    *,
    episode_length: int | None,
    is_eval: bool = False,
) -> gym.Env:
    """
    Build a single NavajoReservoirEnv.

    - Training: pass episode_length (e.g., 3600) -> random-window episodes inside env.reset
    - Eval: pass episode_length=None -> full series
    """
    reward_fn = build_reward(reward_spec_str)

    env = NavajoReservoirEnv(
        data_raw=data_raw,
        data_norm=data_norm,
        norm_stats=norm_stats,
        reward_fn=reward_fn,
        episode_length=episode_length,
        is_eval=is_eval,
    )
    return env


def make_vec_env(
    *,
    data_raw: pd.DataFrame,
    data_norm: pd.DataFrame,
    norm_stats: pd.DataFrame,
    reward_spec_str: str,
    n_envs: int,
    episode_length: int | None,
    is_eval: bool = False,
    use_subproc: bool = False,
):
    """
    Build a VecEnv of NavajoReservoirEnv instances.
    """
    vec_cls = SubprocVecEnv if use_subproc else DummyVecEnv

    def _make_single_env():
        return make_env(
            data_raw=data_raw,
            data_norm=data_norm,
            norm_stats=norm_stats,
            reward_spec_str=reward_spec_str,
            episode_length=episode_length,
            is_eval=is_eval,
        )

    env_fns = [_make_single_env for _ in range(n_envs)]
    return vec_cls(env_fns)


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------
def run_test_rollout(
    run_dir: Path | str,
    reward_spec: str,
    *,
    n_years_test: int = 8,
    model_name: str = "last_model",
    device: str = "auto",
) -> pd.DataFrame:
    """
    Deterministic rollout over full test period.

    Returns a DataFrame indexed by date with:
      - reward, reward components rc_*
      - storage_agent_af vs storage_hist_af
      - total_release_agent_cfs vs release_cfs (historic)
      - component releases from env info:
          sanjuan_release_cfs (component #1)
          niip_release_cfs    (component #2)
      - hydropower (agent + historic when possible)
      - raw model outputs (action_0/action_1) and requested releases
      - post-constraint penalties and end-of-step state (storage/elevation)
    """
    run_dir = Path(run_dir)

    datasets = load_datasets(n_years_test=n_years_test)
    eval_env = make_env(
        data_raw=datasets["test_raw"],
        data_norm=datasets["test_norm"],
        norm_stats=datasets["norm_stats"],
        reward_spec_str=reward_spec,
        episode_length=None,  # full series
        is_eval=True,
    )

    model_path = run_dir / model_name
    if model_path.suffix == "":
        model_path = model_path.with_suffix(".zip")
    agent = PPO.load(model_path, device=device)

    obs, _ = eval_env.reset()
    step = 0
    records: list[dict] = []

    while True:
        # Date
        if hasattr(eval_env, "_current_date") and callable(eval_env._current_date):  # type: ignore[attr-defined]
            date = eval_env._current_date()  # type: ignore[attr-defined]
        else:
            t = getattr(eval_env, "t", step)
            start_idx = getattr(eval_env, "start_idx", 0)
            global_idx = start_idx + t
            date = eval_env.data_raw.index[global_idx]

        # Agent internal state before step
        storage_agent_af = float(eval_env.storage_af)
        elev_agent_ft = float(eval_env.capacity_to_elev(storage_agent_af))

        # Step
        action, _ = agent.predict(obs, deterministic=True)
        action_arr = np.asarray(action, dtype=float).reshape(-1)
        if action_arr.size >= 2:
            a0, a1 = float(action_arr[0]), float(action_arr[1])
        elif action_arr.size == 1:
            a0, a1 = float(action_arr[0]), float(action_arr[0])
        else:
            a0, a1 = float("nan"), float("nan")

        # Map raw actions to *requested* releases (pre-capping/physics), mirroring env.step()
        frac_0 = (a0 + 1.0) / 2.0
        frac_1 = (a1 + 1.0) / 2.0
        requested_rel0_cfs = float(
            eval_env.min_release_cfs
            + frac_0 * (eval_env.max_total_release_cfs - eval_env.min_release_cfs)
        )
        requested_rel1_cfs = float(
            eval_env.min_release_cfs
            + frac_1 * (eval_env.max_total_release_cfs - eval_env.min_release_cfs)
        )
        requested_total_cfs = float(requested_rel0_cfs + requested_rel1_cfs)

        next_obs, reward, terminated, truncated, info_step = eval_env.step(action)
        done = bool(terminated or truncated)

        rec: dict[str, float | int | pd.Timestamp] = {
            "step": step,
            "date": pd.to_datetime(date),
            "reward": float(reward),
            "storage_agent_af": storage_agent_af,
            "elev_agent_ft": elev_agent_ft,
            "action_0": a0,
            "action_1": a1,
            "requested_rel0_cfs": requested_rel0_cfs,
            "requested_rel1_cfs": requested_rel1_cfs,
            "requested_total_release_cfs": requested_total_cfs,
        }

        # Reward components
        comps = info_step.get("reward_components", {})
        for k, v in comps.items():
            rec[f"rc_{k}"] = float(v)

        # Component releases (use env’s native keys)
        if "sanjuan_release_cfs" in info_step:
            rec["sanjuan_release_cfs"] = float(info_step["sanjuan_release_cfs"])
            rec["release_comp1_cfs"] = float(info_step["sanjuan_release_cfs"])
        if "niip_release_cfs" in info_step:
            rec["niip_release_cfs"] = float(info_step["niip_release_cfs"])
            rec["release_comp2_cfs"] = float(info_step["niip_release_cfs"])
        if "total_release_cfs" in info_step:
            rec["release_agent_cfs"] = float(info_step["total_release_cfs"])

        # Penalties & end-of-step state (from info)
        if "release_cap_penalty" in info_step:
            rec["release_cap_penalty"] = float(info_step["release_cap_penalty"])
        if "release_phys_penalty" in info_step:
            rec["release_phys_penalty"] = float(info_step["release_phys_penalty"])
        if "storage_af" in info_step:
            rec["storage_agent_af_end"] = float(info_step["storage_af"])
        if "elev_ft" in info_step:
            rec["elev_agent_ft_end"] = float(info_step["elev_ft"])
        if "min_storage_af" in info_step:
            rec["min_storage_af"] = float(info_step["min_storage_af"])
        if "max_storage_af" in info_step:
            rec["max_storage_af"] = float(info_step["max_storage_af"])

        # SPR / downstream proxy info
        for k in [
            "spring_wy",
            "spring_oi",
            "spring_go",
            "sj_at_farmington_cfs",
            "sj_at_farmington_lag2_cfs",
        ]:
            if k in info_step:
                v = info_step[k]
                if isinstance(v, (bool, np.bool_)):
                    rec[k] = int(bool(v))
                elif v is None:
                    pass
                else:
                    try:
                        rec[k] = float(v)
                    except Exception:
                        pass

        # Raw forcings row (prefer info to avoid index edge-cases)
        date_row = info_step.get("raw_forcings", None)
        if date_row is None:
            date_row = eval_env.data_raw.loc[rec["date"]]

        if "storage_af" in date_row.index:
            rec["storage_hist_af"] = float(date_row["storage_af"])

        # Include all numeric raw columns (these are the full environment inputs).
        for col in getattr(eval_env.data_raw, "columns", []):
            if col in rec:
                continue
            if col not in date_row.index:
                continue
            val = date_row[col]
            try:
                if pd.isna(val):
                    continue
            except Exception:
                pass
            try:
                rec[col] = float(val)
            except Exception:
                continue

        # Hydropower: prefer env-computed value (uses post-step elevation)
        if "hydropower_mwh" in info_step:
            rec["hydro_agent_mwh"] = float(info_step["hydropower_mwh"])
        elif "release_comp2_cfs" in rec:
            elev_for_hp = float(rec.get("elev_agent_ft_end", elev_agent_ft))
            hp_agent = navajo_power_generation_model(
                cfs_values=float(rec["release_comp2_cfs"]),
                elevation_ft=elev_for_hp,
            )
            rec["hydro_agent_mwh"] = float(hp_agent)

        # Historic hydropower: use historic total release + elevation from historic storage
        if "storage_hist_af" in rec and "release_cfs" in rec:
            elev_hist_ft = float(eval_env.capacity_to_elev(rec["storage_hist_af"]))
            hp_hist = navajo_power_generation_model(
                cfs_values=float(rec["release_cfs"]),
                elevation_ft=elev_hist_ft,
            )
            rec["elev_hist_ft"] = elev_hist_ft
            rec["hydro_hist_mwh"] = float(hp_hist)

        records.append(rec)

        obs = next_obs
        step += 1
        if done:
            break

    df = pd.DataFrame.from_records(records).set_index("date").sort_index()
    return df


def make_standard_plots(df: pd.DataFrame, outdir: Path | str) -> None:
    """
    Original-codeplots:
      - total reward
      - reward components
      - storage actual vs predicted
      - total release actual vs predicted
      - component releases (comp1 vs comp2)
      - hydropower (agent)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Total reward
    fig, ax = plt.subplots(figsize=(10, 4))
    df["reward"].plot(ax=ax)
    ax.set_title("Total reward over test period")
    ax.set_ylabel("reward")
    fig.tight_layout()
    fig.savefig(outdir / "reward_total.png", dpi=150)
    plt.close(fig)

    # Reward components
    rc_cols = [c for c in df.columns if c.startswith("rc_")]
    if rc_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        df[rc_cols].plot(ax=ax)
        ax.set_title("Reward components over test period")
        ax.set_ylabel("component value")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / "reward_components.png", dpi=150)
        plt.close(fig)

    # Storage actual vs predicted
    if "storage_hist_af" in df.columns and "storage_agent_af" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        df["storage_hist_af"].plot(ax=ax, label="Actual storage (AF)")
        df["storage_agent_af"].plot(ax=ax, label="Predicted storage (AF)")
        ax.set_title("Predicted vs Actual Storage (AF)")
        ax.set_ylabel("storage [AF]")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / "storage_pred_vs_actual.png", dpi=150)
        plt.close(fig)

    # Total release actual vs predicted
    if "release_cfs" in df.columns and "release_agent_cfs" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        df["release_cfs"].plot(ax=ax, label="Actual total release (cfs)")
        df["release_agent_cfs"].plot(ax=ax, label="Predicted total release (cfs)")
        ax.set_title("Predicted vs Actual Total Release (cfs)")
        ax.set_ylabel("cfs")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / "release_pred_vs_actual.png", dpi=150)
        plt.close(fig)

    # Component releases (comp1 vs comp2)
    if "release_comp1_cfs" in df.columns and "release_comp2_cfs" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        df["release_comp1_cfs"].plot(ax=ax, label="Release component #1 (cfs)")
        df["release_comp2_cfs"].plot(ax=ax, label="Release component #2 (cfs)")
        ax.set_title("Component Releases (cfs)")
        ax.set_ylabel("cfs")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / "release_comp1_vs_comp2.png", dpi=150)
        plt.close(fig)

    # Hydropower (agent)
    if "hydro_agent_mwh" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        df["hydro_agent_mwh"].plot(ax=ax)
        ax.set_title("Hydropower generation (agent) [MWh/day]")
        ax.set_ylabel("MWh/day")
        fig.tight_layout()
        fig.savefig(outdir / "hydropower_agent.png", dpi=150)
        plt.close(fig)


def compute_test_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper.

    Prefer using deepreservoir.drl.metrics.compute_metrics directly.
    """
    return drl_metrics.compute_metrics(df, which="core")


# ---------------------------------------------------------------------
# DRLModel class: single training entry point
# ---------------------------------------------------------------------
class DRLModel:
    """
    Single entry point to train/eval, Colab-equivalent intent.

    Key Colab behavior:
    - episode_length_train=3600
    - n_episodes=350
    - total_timesteps = n_episodes * episode_length_train
    """

    def __init__(
        self,
        n_years_test: int,
        reward_spec: str,
        *,
        algo: str = "ppo",
        logdir: str | Path = "runs/debug",
        seed: int | None = None,
        device: str = "auto",
        gamma: float | None = None,
        n_envs: int = 1,
        use_subproc_vec: bool = False,
        episode_length_train: int = 3600,
    ) -> None:

        self.n_years_test = int(n_years_test)
        self.reward_spec = str(reward_spec)
        self.algo = algo.lower()
        self.logdir = Path(logdir)
        self.seed = seed
        self.device = device
        self.n_envs = int(n_envs)
        self.use_subproc_vec = bool(use_subproc_vec)
        self.gamma = gamma
        self.episode_length_train = int(episode_length_train)

        self._train_updates_cb: TrainUpdateRewardComponentsCallback | None = None
        self.train_update_metrics_: pd.DataFrame | None = None

        # Best-effort load previous metrics
        try:
            self.load_train_update_metrics()
        except Exception:
            pass

        self.datasets = load_datasets(self.n_years_test)

        # TRAIN ENV (random 3600-step episodes)
        if self.n_envs == 1:
            self.train_env = make_env(
                data_raw=self.datasets["train_raw"],
                data_norm=self.datasets["train_norm"],
                norm_stats=self.datasets["norm_stats"],
                reward_spec_str=self.reward_spec,
                episode_length=self.episode_length_train,
                is_eval=False,
            )
            self.train_env = Monitor(self.train_env)
        else:
            self.train_env = make_vec_env(
                data_raw=self.datasets["train_raw"],
                data_norm=self.datasets["train_norm"],
                norm_stats=self.datasets["norm_stats"],
                reward_spec_str=self.reward_spec,
                n_envs=self.n_envs,
                episode_length=self.episode_length_train,
                is_eval=False,
                use_subproc=self.use_subproc_vec,
            )

        # EVAL ENV (full series)
        self.eval_env = make_env(
            data_raw=self.datasets["test_raw"],
            data_norm=self.datasets["test_norm"],
            norm_stats=self.datasets["norm_stats"],
            reward_spec_str=self.reward_spec,
            episode_length=None,
            is_eval=True,
        )
        self.eval_env = Monitor(self.eval_env)

        self.agent: PPO | None = None

    def train(
        self,
        *,
        n_episodes: int = 350,
        eval_freq: int = 10_000,
        device: str | None = None,
        n_steps: int | None = None,
        batch_size: int | None = None,
        n_epochs: int | None = None,
        gamma: float | None = None,
        track_reward_components: bool = True,
    ) -> None:
        """
        Train PPO.

        Original code equivalence:
          total_timesteps = n_episodes * episode_length_train

        Note: This matches the original code style most closely when n_envs=1.
        If n_envs>1, SB3 will collect more samples per wall-clock step.
        """
        device_eff = device or self.device
        total_timesteps = int(n_episodes * self.episode_length_train)

        self.agent = build_agent(
            self.train_env,
            algo=self.algo,
            seed=self.seed,
            device=device_eff,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
        )

        best_dir = self.logdir / "best"
        eval_log_dir = self.logdir / "eval"
        best_dir.mkdir(parents=True, exist_ok=True)
        eval_log_dir.mkdir(parents=True, exist_ok=True)

        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(eval_log_dir),
            eval_freq=int(eval_freq),
            deterministic=True,
            render=False,
        )

        cb_list: list[BaseCallback] = [eval_callback]
        self._train_updates_cb = None

        if track_reward_components:
            self._train_updates_cb = TrainUpdateRewardComponentsCallback()
            cb_list.append(self._train_updates_cb)

        callback: BaseCallback = cb_list[0] if len(cb_list) == 1 else CallbackList(cb_list)

        try:
            self.agent.learn(total_timesteps=total_timesteps, callback=callback)
            self.save_model("last_model")
        finally:
            if self._train_updates_cb is not None:
                df_upd = pd.DataFrame(self._train_updates_cb.update_history)
                self.train_update_metrics_ = df_upd
                out_path = self.logdir / "train_update_metrics.parquet"
                df_upd.to_parquet(out_path, index=False)
                df_upd.to_csv(out_path.with_suffix(".csv"), index=False)

    def save_model(self, name: str = "last_model") -> Path:
        if self.agent is None:
            raise RuntimeError("No agent to save (train or load a model first).")
        path = self.logdir / name
        if path.suffix == "":
            path = path.with_suffix(".zip")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(path))
        return path

    def load_model(self, name: str = "last_model") -> PPO:
        path = self.logdir / name
        if path.suffix == "":
            path = path.with_suffix(".zip")
        self.agent = PPO.load(path, env=self.train_env, device=self.device)
        return self.agent

    def load_train_update_metrics(self) -> pd.DataFrame | None:
        path = self.logdir / "train_update_metrics.parquet"
        if not path.exists():
            return None
        df_upd = pd.read_parquet(path)
        self.train_update_metrics_ = df_upd
        return df_upd

    def evaluate_test(
        self,
        model_name: str = "last_model",
        *,
        save_rollout: bool = True,
        save_plots: bool = True,
        save_metrics: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = run_test_rollout(
            run_dir=self.logdir,
            reward_spec=self.reward_spec,
            n_years_test=self.n_years_test,
            model_name=model_name,
            device=self.device,
        )

        if save_rollout:
            out_path = self.logdir / "eval_test_rollout.parquet"
            df.to_parquet(out_path)
            df.to_csv(out_path.with_suffix(".csv"), index=True)

        if save_plots:
            make_standard_plots(df, self.logdir / "eval_plots")

        metrics_df = drl_metrics.compute_metrics(df, which="core")
        if save_metrics:
            drl_metrics.save_metrics(df_test=df, outdir=self.logdir, which="core", stem="eval_metrics")

        return df, metrics_df


class TrainUpdateRewardComponentsCallback(BaseCallback):
    """
    Per-rollout (per PPO update) mean reward component tracker.
    Expects env to expose:
      info["reward_components_step"] OR info["reward_components"].
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.rollout_sums: dict[str, float] = {}
        self.rollout_reward_sum: float = 0.0
        self.rollout_count: int = 0
        self.update_history: list[dict] = []
        self._update_idx: int = 0

    def _on_training_start(self) -> None:
        self.rollout_sums = {}
        self.rollout_reward_sum = 0.0
        self.rollout_count = 0
        self.update_history = []
        self._update_idx = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", None)

        if rewards is not None:
            try:
                self.rollout_reward_sum += float(np.sum(rewards))
            except Exception:
                self.rollout_reward_sum += float(rewards)

        if infos:
            for info in infos:
                comps = info.get("reward_components_step", info.get("reward_components", {}))
                if not comps:
                    continue
                for k, v in comps.items():
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    self.rollout_sums[k] = self.rollout_sums.get(k, 0.0) + fv

            self.rollout_count += len(infos)
        else:
            self.rollout_count += 1

        return True

    def _on_rollout_end(self) -> None:
        count = max(int(self.rollout_count), 1)

        rec: dict[str, float | int] = {
            "update_idx": int(self._update_idx),
            "timesteps": int(self.num_timesteps),
            "rollout_steps": int(self.rollout_count),
            "mean_total_reward": float(self.rollout_reward_sum / count),
        }
        for k, total in self.rollout_sums.items():
            rec[f"mean_{k}"] = float(total / count)

        self.update_history.append(rec)
        self._update_idx += 1

        # reset for next rollout
        self.rollout_sums = {}
        self.rollout_reward_sum = 0.0
        self.rollout_count = 0
