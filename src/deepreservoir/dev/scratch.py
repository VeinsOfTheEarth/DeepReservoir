from pathlib import Path
import sys
import pandas as pd
from deepreservoir.drl.model import DRLModel
from deepreservoir.drl import metrics as drl_metrics
from pathlib import Path

from deepreservoir.drl.model import DRLModel
from deepreservoir.drl import metrics as drl_metrics


run_dir = Path("runs/alphas_test")

# Example alphas (1–10 range). Alpha defaults to 1.0 if '@' is omitted.
reward_spec = (
    "dam_safety:storage_band@30.0,"
    "esa_min_flow:baseline@3.0,"
    "flooding:baseline@5.0,"
    "hydropower:baseline@2.0,"
    "niip:baseline@7.5,"
    "physics:baseline@1.5"
)

# 140 * 3600 = 504,000 timesteps
episode_length_train = 3600
n_episodes = 140

m = DRLModel(
    n_years_test=8,
    reward_spec=reward_spec,
    logdir=run_dir,
    device="cpu",
    n_envs=8,               # reasonable for DummyVecEnv on CPU
    use_subproc_vec=False,  # safest on Windows; set True only if you want/need it
    episode_length_train=episode_length_train,
)

m.train(
    n_episodes=n_episodes,
    eval_freq=50_000,        # don’t spam eval; it’s expensive
    n_steps=2048,            # PPO rollout length per env
    batch_size=4096,         # must be <= n_steps*n_envs (here 2048*8=16384)
    n_epochs=10,
    gamma=0.999,
    track_reward_components=True,
)

# Evaluate + write outputs
df_test, df_metrics = m.evaluate_test(
    save_rollout=True,
    save_plots=True,
    save_metrics=True,
)

print("\nSaved eval metrics:")
print(df_metrics.to_string(index=False))

# Optional: show more detailed dam safety
df_ds = drl_metrics.compute_metrics(df_test, which="dam_safety_detail")
print("\nDam safety detail:")
print(df_ds.to_string(index=False))

