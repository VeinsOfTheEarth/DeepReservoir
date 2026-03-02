from pathlib import Path
from deepreservoir.drl.model import DRLModel
from pathlib import Path

run_dir = Path("runs/alphas_are_1_4million_timesteps")

# Reward spec (alpha defaults to 1.0 if '@' is omitted).
# Keep your dam_safety alpha high to see whether the new (total, allocation)
# action space makes the constraint learnable without "always dump water".
reward_spec = (
    "dam_safety:storage_band_shaped@2.0,"
    "esa_min_flow:baseline,"
    "flooding:baseline@1.0,"
    "hydropower:baseline@1.0,"
    "niip:baseline@1.0,"
)

# ~500k timesteps (matches your previous scratch convention)
# Note: DRLModel uses total_timesteps = n_episodes * episode_length_train.
episode_length_train = 3600
n_episodes = 280 # 140 * 3600 = 504,000

m = DRLModel(
    n_years_test=8,
    reward_spec=reward_spec,
    logdir=run_dir,
    device="cpu",
    n_envs=8,               # reasonable on CPU (DummyVecEnv)
    use_subproc_vec=False,  # safest on Windows
    episode_length_train=episode_length_train,
)

# m.train(
#     n_episodes=n_episodes,
#     eval_freq=50_000,        # evaluation is expensive; don't spam it
#     n_steps=2048,            # PPO rollout length per env
#     batch_size=4096,         # must be <= n_steps*n_envs (2048*8=16384)
#     n_epochs=10,
#     gamma=0.999,
#     track_reward_components=True,
# )

# Load explicitly (not strictly required, but mirrors prior scratch behavior)
m.load_model("last_model")

# Evaluate on the full test period + write outputs
df_test, df_metrics = m.evaluate_test(
    save_rollout=True,
    save_plots=True,
    save_metrics=True,
)
