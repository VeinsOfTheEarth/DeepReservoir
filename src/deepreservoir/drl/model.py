# model.py
"""
High-level training utilities for DeepReservoir:

- loads data via NavajoData
- splits into train / test
- builds reward (from registry)
- builds Gymnasium env
- trains & evaluates an SB3 agent

Intended to be called from cli.py
"""

from pathlib import Path
from typing import Dict

import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from deepreservoir.data import loader
from deepreservoir.drl import helpers
from deepreservoir.drl import rewards as drl_rewards
from deepreservoir.drl.environs import NavajoReservoirEnv


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

    # These keys are produced by the updated loader with normalization
    data = alldata["model_data"]               # raw
    datanorm = alldata["model_data_norm"]      # normalized
    norm_stats = alldata["model_norm_stats"]   # mean/std table

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
def build_agent(env: gym.Env, algo: str = "ppo", seed: int | None = None):
    """
    Construct the SB3 agent for a given environment.
    """
    algo = algo.lower()
    if algo == "ppo":
        return PPO("MlpPolicy", env, verbose=1, seed=seed)
    else:
        raise ValueError(f"Unsupported algo: {algo}")


def build_reward(reward_spec_str: str):
    """
    Build the composite reward function from a text spec.

    Example:
        "dam_safety:storage_band,esa_min_flow:baseline,hydropower:baseline,niip:baseline"
    """
    spec = drl_rewards.parse_objective_spec(reward_spec_str)
    # no per-objective weights yet → all 1.0
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
        # use defaults for now; we can parameterize later
        episode_length=None,   # full series
        is_eval=is_eval,
    )
    return env


# ---------------------------------------------------------------------
# High-level training entry point (used by cli.py)
# ---------------------------------------------------------------------
def train(
    n_years_test: int,
    reward_spec: str,
    *,
    algo: str = "ppo",
    total_timesteps: int = 350_000,
    logdir: str | Path = "runs/debug",
    seed: int | None = None,
) -> None:
    """
    High-level training entry point used by the CLI.

    Args
    ----
    n_years_test:
        Number of most recent water-years reserved for testing.
    reward_spec:
        Text spec for objectives/rewards (see build_reward docstring).
    algo:
        RL algorithm name (currently only "ppo").
    total_timesteps:
        Number of SB3 training timesteps.
    logdir:
        Directory where models and logs are written.
    seed:
        Optional random seed passed to SB3.
    """
    # 1) Data
    datasets = load_datasets(n_years_test)

    # 2) Environments
    train_env = make_env(
        data_raw=datasets["train_raw"],
        data_norm=datasets["train_norm"],
        norm_stats=datasets["norm_stats"],
        reward_spec_str=reward_spec,
        is_eval=False,
    )

    eval_env = make_env(
        data_raw=datasets["test_raw"],
        data_norm=datasets["test_norm"],
        norm_stats=datasets["norm_stats"],
        reward_spec_str=reward_spec,
        is_eval=True,
    )

    # 3) Agent
    agent = build_agent(train_env, algo=algo, seed=seed)

    # 4) Logging / evaluation callback
    logdir = Path(logdir)
    best_dir = logdir / "best"
    eval_log_dir = logdir / "eval"
    best_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(eval_log_dir),
        eval_freq=10_000,  # TODO: expose as CLI arg later if needed
        deterministic=True,
        render=False,
    )

    # 5) Train
    agent.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # 6) Save final model
    agent.save(str(logdir / "last_model"))
