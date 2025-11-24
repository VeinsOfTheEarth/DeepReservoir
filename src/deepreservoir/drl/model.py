"""
High-level training and evaluation utilities for DeepReservoir DRL.

- loads data via NavajoData
- splits into train / test
- builds reward (from registry)
- builds Gymnasium envs
- trains & evaluates an SB3 agent
- provides helpers to roll out on the test period and compute metrics
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
from deepreservoir.drl.environs import NavajoReservoirEnv
from deepreservoir.define_env.hydropower_model import navajo_power_generation_model


# ---------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------
def load_datasets(n_years_test: int) -> Dict[str, pd.DataFrame]:
    """
    Load Navajo data and split into train / test sets for model input.

    Returns a dict with keys:
        'train_raw'   : training data (unnormalized)
        'test_raw'    : test data (unnormalized)
        'train_norm'  : training data (normalized)
        'test_norm'   : test data (normalized)
        'norm_stats'  : normalization mean/std table
    """
    nav_data = loader.NavajoData()
    alldata = nav_data.load_all(include_cont_streamflow=False, model_data=True)

    data = alldata["model_data"]              # raw
    datanorm = alldata["model_data_norm"]     # normalized
    norm_stats = alldata["model_norm_stats"]  # mean/std table

    # Split by water year (most recent n_years_test are test)
    data_train, data_test = helpers.split_train_test_by_water_year(
        data, n_years_test=n_years_test
    )

    # Use the same index split for the normalized data
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
    """
    Construct the SB3 agent for a given environment.
    """
    algo = algo.lower()
    if algo != "ppo":
        raise ValueError(f"Unsupported algo: {algo}")

    # Start from SB3 defaults; only override when user passes a value
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
    """
    Build the composite reward function from a text spec.

    Example:
        "dam_safety:storage_band,esa_min_flow:baseline,flooding:baseline,niip:baseline"
    """
    spec = drl_rewards.parse_objective_spec(reward_spec_str)
    composite = drl_rewards.build_composite_reward(spec, weights=None)
    return composite


def make_env(
    data_raw: pd.DataFrame,
    data_norm: pd.DataFrame,
    norm_stats: pd.DataFrame,
    reward_spec_str: str,
    *,
    is_eval: bool = False,
) -> gym.Env:
    """
    Build a NavajoReservoirEnv for training or evaluation.
    """
    reward_fn = build_reward(reward_spec_str)

    env = NavajoReservoirEnv(
        data_raw=data_raw,
        data_norm=data_norm,
        norm_stats=norm_stats,
        reward_fn=reward_fn,
        episode_length=None,  # full series
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
    is_eval: bool = False,
    use_subproc: bool = False,
):
    """
    Build a VecEnv (Dummy or Subproc) of NavajoReservoirEnv instances.

    For training, set is_eval=False; for eval you usually just want 1 env
    and can skip this helper.
    """
    vec_cls = SubprocVecEnv if use_subproc else DummyVecEnv

    def _make_single_env():
        # Wrap your existing make_env
        return make_env(
            data_raw=data_raw,
            data_norm=data_norm,
            norm_stats=norm_stats,
            reward_spec_str=reward_spec_str,
            is_eval=is_eval,
        )

    # SB3 expects a list of callables that each create an env
    env_fns = [ _make_single_env for _ in range(n_envs) ]
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
    Load a trained model from `run_dir` and roll it out over the test period.

    Returns a DataFrame with one row per test day, including:
      - date (index), step, reward
      - action_raw
      - reward component columns named rc_<key>
      - some raw hydrology columns if available
    """
    run_dir = Path(run_dir)

    # 1) Load datasets and build eval env over TEST period
    datasets = load_datasets(n_years_test=n_years_test)
    eval_env = make_env(
        data_raw=datasets["test_raw"],
        data_norm=datasets["test_norm"],
        norm_stats=datasets["norm_stats"],
        reward_spec_str=reward_spec,
        is_eval=True,
    )

    # 2) Load trained model (no env; we drive eval_env manually)
    model_path = run_dir / model_name
    if model_path.suffix == "":
        model_path = model_path.with_suffix(".zip")
    agent = PPO.load(model_path, device=device)

    # 3) Roll out deterministically over the whole test episode
    obs, info = eval_env.reset()
    step = 0
    records: list[dict] = []

    while True:
        # --- date & agent state BEFORE the step ---
        if hasattr(eval_env, "_current_date") and callable(eval_env._current_date):  # type: ignore[attr-defined]
            date = eval_env._current_date()  # type: ignore[attr-defined]
        else:
            t = getattr(eval_env, "t", step)
            start_idx = getattr(eval_env, "start_idx", 0)
            global_idx = start_idx + t
            date = eval_env.data_raw.index[global_idx]

        # Agent storage + elevation
        storage_agent_af = float(eval_env.storage_af)
        elev_agent_ft = float(eval_env.capacity_to_elev(storage_agent_af))

        # --- action and env step ---
        action, _ = agent.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info_step = eval_env.step(action)
        done = bool(terminated or truncated)

        rec: dict[str, float | int | pd.Timestamp] = {
            "step": step,
            "date": pd.to_datetime(date),
            "reward": float(reward),
            "storage_agent_af": storage_agent_af,
            "elev_ft": elev_agent_ft,          # agent elevation
        }

        # Reward components
        comps = info_step.get("reward_components", {})
        for key, val in comps.items():
            rec[f"rc_{key}"] = float(val)

        # Agent releases (from info)
        sanjuan_rel_cfs = info_step.get("sanjuan_release_cfs", None)
        total_rel_cfs = info_step.get("total_release_cfs", None)
        if sanjuan_rel_cfs is not None:
            rec["sanjuan_release_cfs"] = float(sanjuan_rel_cfs)
        if total_rel_cfs is not None:
            rec["release_agent_cfs"] = float(total_rel_cfs)

        # Historic data for that date
        date_row = eval_env.data_raw.loc[rec["date"]]

        # Historic storage + release, inflow, evap, etc.
        if "storage_af" in date_row.index:
            rec["storage_hist_af"] = float(date_row["storage_af"])
        for col in [
            "release_cfs",
            "inflow_cfs",
            "evap_cfs",
            "sj_farmington_q_cfs",
            "animas_farmington_q_cfs",
            "sj_bluff_q_cfs",
        ]:
            if col in date_row.index:
                rec[col] = float(date_row[col])

        # --- Hydropower: agent & historic, using elevations already in hand ---
        # Agent hydropower: use mainstem (San Juan) release + agent elevation
        if sanjuan_rel_cfs is not None:
            hp_agent = navajo_power_generation_model(
                cfs_values=float(sanjuan_rel_cfs),
                elevation_ft=elev_agent_ft,
            )
            rec["hydro_agent_mwh"] = float(hp_agent)

        # Historic hydropower: use historic release + elevation from historic storage
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
    Generate a basic set of diagnostic plots for the test rollout.
    Saves PNGs into `outdir`.
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

    # Reward components (rc_<key>)
    rc_cols = [c for c in df.columns if c.startswith("rc_")]
    if rc_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        df[rc_cols].plot(ax=ax)
        ax.set_title("Reward components (weighted) over test period")
        ax.set_ylabel("component value")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / "reward_components.png", dpi=150)
        plt.close(fig)

    # Storage
    if "storage_af" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        df["storage_af"].plot(ax=ax)
        ax.set_title("Reservoir storage (AF)")
        ax.set_ylabel("storage [AF]")
        fig.tight_layout()
        fig.savefig(outdir / "storage_af.png", dpi=150)
        plt.close(fig)

    # Release vs inflow
    rel_cols = [c for c in ["release_cfs", "inflow_cfs"] if c in df.columns]
    if rel_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        df[rel_cols].plot(ax=ax)
        ax.set_title("Release and inflow (cfs)")
        ax.set_ylabel("cfs")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / "release_inflow.png", dpi=150)
        plt.close(fig)

    # San Juan at Farmington
    if "sj_farmington_q_cfs" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        df["sj_farmington_q_cfs"].plot(ax=ax)
        ax.set_title("San Juan at Farmington (cfs)")
        ax.set_ylabel("cfs")
        fig.tight_layout()
        fig.savefig(outdir / "sj_farmington_q_cfs.png", dpi=150)
        plt.close(fig)


def compute_test_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple metrics table from the test rollout.
    """
    metrics: Dict[str, float] = {}

    # Total and mean reward
    metrics["total_reward"] = float(df["reward"].sum())
    metrics["mean_reward"] = float(df["reward"].mean())

    # Per-component reward metrics
    rc_cols = [c for c in df.columns if c.startswith("rc_")]
    for col in rc_cols:
        key = col[len("rc_") :]
        metrics[f"sum_rc_{key}"] = float(df[col].sum())
        metrics[f"mean_rc_{key}"] = float(df[col].mean())

    # Simple hydrology diagnostics if present
    if "storage_af" in df.columns:
        metrics["mean_storage_af"] = float(df["storage_af"].mean())
        metrics["min_storage_af"] = float(df["storage_af"].min())
        metrics["max_storage_af"] = float(df["storage_af"].max())

    if "release_cfs" in df.columns:
        metrics["mean_release_cfs"] = float(df["release_cfs"].mean())
        metrics["max_release_cfs"] = float(df["release_cfs"].max())

    if "sj_farmington_q_cfs" in df.columns:
        metrics["mean_sj_farmington_cfs"] = float(df["sj_farmington_q_cfs"].mean())
        metrics["min_sj_farmington_cfs"] = float(df["sj_farmington_q_cfs"].min())

    return pd.DataFrame([metrics])


# ---------------------------------------------------------------------
# DRLModel class: single training entry point
# ---------------------------------------------------------------------
class DRLModel:
    def __init__(
        self,
        n_years_test: int,
        reward_spec: str,
        *,
        algo: str = "ppo",
        logdir: str | Path = "runs/debug",
        seed: int | None = None,
        device: str = "auto",
        n_envs: int = 1,              # NEW
        use_subproc_vec: bool = False # NEW
    ) -> None:
        self.n_years_test = n_years_test
        self.reward_spec = reward_spec
        self.algo = algo.lower()
        self.logdir = Path(logdir)
        self.seed = seed
        self.device = device
        self.n_envs = int(n_envs)
        self.use_subproc_vec = bool(use_subproc_vec)

        # Load data
        self.datasets = load_datasets(n_years_test)

        # TRAIN ENV: single or VecEnv
        if self.n_envs == 1:
            self.train_env = make_env(
                data_raw=self.datasets["train_raw"],
                data_norm=self.datasets["train_norm"],
                norm_stats=self.datasets["norm_stats"],
                reward_spec_str=self.reward_spec,
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
                is_eval=False,
                use_subproc=self.use_subproc_vec,
            )

        # EVAL ENV: keep it single-env for now
        self.eval_env = make_env(
            data_raw=self.datasets["test_raw"],
            data_norm=self.datasets["test_norm"],
            norm_stats=self.datasets["norm_stats"],
            reward_spec_str=self.reward_spec,
            is_eval=True,
        )
        # Wrap in Monitor for correct eval metrics
        self.eval_env = Monitor(self.eval_env)


        # Agent will be built in train() or load_model()
        self.agent: PPO | None = None

    def train(
        self,
        total_timesteps: int = 350_000,
        eval_freq: int = 10_000,
        *,
        device: str | None = None,
        n_steps: int | None = None,
        batch_size: int | None = None,
        n_epochs: int | None = None,
        track_reward_components: bool = True,
        gamma: float | None = None,

    ) -> None:
        """
        Build the agent (with the given hyperparams) and train it.

        If track_reward_components=True, records mean reward per objective
        for each episode in `self.episode_reward_components_` (a DataFrame).
        """
        device_eff = device or self.device

        self.agent = build_agent(
            self.train_env,
            algo=self.algo,
            seed=self.seed,
            device=device_eff,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,  # <-- this was missing
        )

        best_dir = self.logdir / "best"
        eval_log_dir = self.logdir / "eval"
        best_dir.mkdir(parents=True, exist_ok=True)
        eval_log_dir.mkdir(parents=True, exist_ok=True)

        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(eval_log_dir),
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
        )

        cb_list = [eval_callback]
        self._reward_components_cb = None

        if track_reward_components:
            self._reward_components_cb = EpisodeRewardComponentsCallback()
            cb_list.append(self._reward_components_cb)

        callback = (
            cb_list[0]
            if len(cb_list) == 1
            else CallbackList(cb_list)
        )

        self.agent.learn(total_timesteps=total_timesteps, callback=callback)
        self.save_model("last_model")

        # convert callback history to a DataFrame for later plotting
        if self._reward_components_cb is not None:
            import pandas as pd
            self.episode_reward_components_ = pd.DataFrame(
                self._reward_components_cb.episode_history
            )
        else:
            self.episode_reward_components_ = None

    def save_model(self, name: str = "last_model") -> Path:
        """
        Save the current SB3 agent to `logdir/name.zip`.
        """
        if self.agent is None:
            raise RuntimeError("No agent to save (train or load a model first).")
        path = self.logdir / name
        if path.suffix == "":
            path = path.with_suffix(".zip")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(path))
        return path

    def load_model(self, name: str = "last_model") -> PPO:
        """
        Load an SB3 agent from `logdir/name.zip` into this DRLModel.
        """
        path = self.logdir / name
        if path.suffix == "":
            path = path.with_suffix(".zip")
        self.agent = PPO.load(path, env=self.train_env, device=self.device)
        return self.agent

    def evaluate_test(
        self,
        model_name: str = "last_model",
        *,
        save_rollout: bool = True,
        save_plots: bool = True,
        save_metrics: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Roll out the trained policy over the test period and generate:
          - a test rollout DataFrame
          - a metrics table DataFrame
        """
        df = run_test_rollout(
            run_dir=self.logdir,
            reward_spec=self.reward_spec,
            n_years_test=self.n_years_test,
            model_name=model_name,
            device=self.device,  # or "cpu" if you want eval always on CPU
        )

        if save_rollout:
            df.to_parquet(self.logdir / "eval_test_rollout.parquet")

        if save_plots:
            make_standard_plots(df, self.logdir / "eval_plots")

        metrics_df = compute_test_metrics(df)
        if save_metrics:
            metrics_df.to_csv(self.logdir / "eval_metrics.csv", index=False)

        return df, metrics_df


class EpisodeRewardComponentsCallback(BaseCallback):
    """
    Accumulates `info["reward_components"]` over each episode and
    stores per-episode mean values in `episode_history`.

    Works with VecEnv (n_envs >= 1).
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_sums: dict[int, dict[str, float]] = {}
        self.episode_counts: dict[int, int] = {}
        self.episode_history: list[dict] = []
        self._next_episode_idx: int = 0

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.episode_sums = {i: {} for i in range(n_envs)}
        self.episode_counts = {i: 0 for i in range(n_envs)}
        self.episode_history = []
        self._next_episode_idx = 0

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        n_envs = len(infos)

        for i in range(n_envs):
            info = infos[i]

            # accumulate this step's components
            comps = info.get("reward_components", {})
            if comps:
                sums = self.episode_sums.setdefault(i, {})
                for k, v in comps.items():
                    sums[k] = sums.get(k, 0.0) + float(v)

            # count steps
            self.episode_counts[i] = self.episode_counts.get(i, 0) + 1

            # if episode ended in this env, finalize stats
            if dones[i]:
                count = max(self.episode_counts[i], 1)
                sums = self.episode_sums.get(i, {})

                rec: dict[str, float | int] = {
                    "episode_idx": self._next_episode_idx,
                    "env_idx": i,
                    "n_steps": count,
                }
                for k, total in sums.items():
                    rec[f"mean_{k}"] = total / count

                self.episode_history.append(rec)
                self._next_episode_idx += 1

                # reset accumulators for this env
                self.episode_sums[i] = {}
                self.episode_counts[i] = 0

        return True
