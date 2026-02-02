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
    run_dir = Path(run_dir)

    datasets = load_datasets(n_years_test=n_years_test)
    eval_env = make_env(
        data_raw=datasets["test_raw"],
        data_norm=datasets["test_norm"],
        norm_stats=datasets["norm_stats"],
        reward_spec_str=reward_spec,
        episode_length=None,  # FULL test series like your Colab testing loop
        is_eval=True,
    )

    model_path = run_dir / model_name
    if model_path.suffix == "":
        model_path = model_path.with_suffix(".zip")
    agent = PPO.load(model_path, device=device)

    obs, info = eval_env.reset()
    step = 0
    records: list[dict] = []

    while True:
        if hasattr(eval_env, "_current_date") and callable(eval_env._current_date):  # type: ignore[attr-defined]
            date = eval_env._current_date()  # type: ignore[attr-defined]
        else:
            t = getattr(eval_env, "t", step)
            start_idx = getattr(eval_env, "start_idx", 0)
            global_idx = start_idx + t
            date = eval_env.data_raw.index[global_idx]

        storage_agent_af = float(eval_env.storage_af)
        elev_agent_ft = float(eval_env.capacity_to_elev(storage_agent_af))

        action, _ = agent.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info_step = eval_env.step(action)
        done = bool(terminated or truncated)

        rec: dict[str, float | int | pd.Timestamp] = {
            "step": step,
            "date": pd.to_datetime(date),
            "reward": float(reward),
            "storage_agent_af": storage_agent_af,
            "elev_ft": elev_agent_ft,
        }

        comps = info_step.get("reward_components", {})
        for key, val in comps.items():
            rec[f"rc_{key}"] = float(val)

        # Save BOTH actions if your env returns them (recommended)
        if "q1_release_cfs" in info_step:
            rec["q1_release_cfs"] = float(info_step["q1_release_cfs"])
        if "q2_release_cfs" in info_step:
            rec["q2_release_cfs"] = float(info_step["q2_release_cfs"])
        if "total_release_cfs" in info_step:
            rec["release_agent_cfs"] = float(info_step["total_release_cfs"])

        date_row = eval_env.data_raw.loc[rec["date"]]

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

        # Hydropower (agent uses Q2 typically, but you can choose)
        if "q2_release_cfs" in rec:
            hp_agent = navajo_power_generation_model(
                cfs_values=float(rec["q2_release_cfs"]),
                elevation_ft=elev_agent_ft,
            )
            rec["hydro_agent_mwh"] = float(hp_agent)

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
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    df["reward"].plot(ax=ax)
    ax.set_title("Total reward over test period")
    ax.set_ylabel("reward")
    fig.tight_layout()
    fig.savefig(outdir / "reward_total.png", dpi=150)
    plt.close(fig)

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

    # Storage: HIST vs AGENT (matches your Colab-style plot intent)
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

    # Release: HIST vs AGENT
    if "release_cfs" in df.columns and "release_agent_cfs" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        df["release_cfs"].plot(ax=ax, label="Actual release (cfs)")
        df["release_agent_cfs"].plot(ax=ax, label="Predicted total release (cfs)")
        ax.set_title("Predicted vs Actual Release (cfs)")
        ax.set_ylabel("cfs")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / "release_pred_vs_actual.png", dpi=150)
        plt.close(fig)

    # Q1 vs Q2
    if "q1_release_cfs" in df.columns and "q2_release_cfs" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        df["q1_release_cfs"].plot(ax=ax, label="Q1 (cfs)")
        df["q2_release_cfs"].plot(ax=ax, label="Q2 (cfs)")
        ax.set_title("Q1 vs Q2 (cfs)")
        ax.set_ylabel("cfs")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / "q1_vs_q2.png", dpi=150)
        plt.close(fig)

    # Hydropower
    if "hydro_agent_mwh" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        df["hydro_agent_mwh"].plot(ax=ax)
        ax.set_title("Hydropower generation (agent) [MWh]")
        ax.set_ylabel("MWh")
        fig.tight_layout()
        fig.savefig(outdir / "hydropower_agent.png", dpi=150)
        plt.close(fig)


def compute_test_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics: Dict[str, float] = {}

    metrics["total_reward"] = float(df["reward"].sum())
    metrics["mean_reward"] = float(df["reward"].mean())

    rc_cols = [c for c in df.columns if c.startswith("rc_")]
    for col in rc_cols:
        key = col[len("rc_") :]
        metrics[f"sum_rc_{key}"] = float(df[col].sum())
        metrics[f"mean_rc_{key}"] = float(df[col].mean())

    if "storage_agent_af" in df.columns:
        metrics["mean_storage_agent_af"] = float(df["storage_agent_af"].mean())
        metrics["min_storage_agent_af"] = float(df["storage_agent_af"].min())
        metrics["max_storage_agent_af"] = float(df["storage_agent_af"].max())

    if "release_agent_cfs" in df.columns:
        metrics["mean_release_agent_cfs"] = float(df["release_agent_cfs"].mean())
        metrics["max_release_agent_cfs"] = float(df["release_agent_cfs"].max())

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
        gamma: float | None = None,
        n_envs: int = 1,
        use_subproc_vec: bool = False,
        episode_length_train: int = 3600,   # <-- MATCH COLAB
    ) -> None:

        self.n_years_test = n_years_test
        self.reward_spec = reward_spec
        self.algo = algo.lower()
        self.logdir = Path(logdir)
        self.seed = seed
        self.device = device
        self.n_envs = int(n_envs)
        self.use_subproc_vec = bool(use_subproc_vec)
        self.gamma = gamma
        self.episode_length_train = int(episode_length_train)

        self._train_updates_cb = None
        self.train_update_metrics_ = None

        try:
            self.load_train_update_metrics()
        except Exception:
            pass

        self.datasets = load_datasets(n_years_test)

        # TRAIN ENV (3600-step episodes, random starts inside env.reset)
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

        # EVAL ENV: full test series
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
        n_episodes: int = 350,            # <-- MATCH COLAB
        eval_freq: int = 10_000,
        *,
        device: str | None = None,
        n_steps: int | None = None,
        batch_size: int | None = None,
        n_epochs: int | None = None,
        gamma: float | None = None,
        track_reward_components: bool = True,
    ) -> None:
        device_eff = device or self.device

        # Colab total_timesteps = n_episodes * episode_length
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
            eval_freq=eval_freq,
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
            df.to_csv(out_path.with_suffix(".csv"), index=False)

        if save_plots:
            make_standard_plots(df, self.logdir / "eval_plots")

        metrics_df = compute_test_metrics(df)
        if save_metrics:
            metrics_df.to_csv(self.logdir / "eval_metrics.csv", index=False)

        return df, metrics_df


class TrainUpdateRewardComponentsCallback(BaseCallback):
    """
    Per-rollout (per PPO update) mean reward component tracker.
    Expects env to expose info["reward_components_step"] or info["reward_components"].
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

        self.rollout_sums = {}
        self.rollout_reward_sum = 0.0
        self.rollout_count = 0
