import pandas as pd
from deepreservoir.drl.model import DRLModel
from pathlib import Path
from importlib import reload
from deepreservoir.drl import plotting as drl_plot

experiment_name = 'test3'
# reward_spec = "dam_safety:storage_band,esa_min_flow:baseline,flooding:baseline,niip:baseline,hydropower:baseline,physics:scale_penalty"
reward_spec = "dam_safety:storage_band,physics:scale_penalty"
run_dir = Path("runs/debug_parallel")

m = DRLModel(
    n_years_test=8,
    reward_spec=reward_spec,
    logdir=run_dir,
    device="cpu",
    n_envs=12,              # try 4 first
    use_subproc_vec=False, # start with DummyVecEnv for debugging
)

m.train(
    total_timesteps=500_000,
    n_steps=4096,
    batch_size=1024,
    n_epochs=10,
    gamma=0.999,
    track_reward_components=True,
)

# load the policy weights
m.load_model("last_model")

# # 4
# elapsed : 2938
# timesteps : 507904

# # 16
# time_elapsed : 964
# total_timesteps : 524288

# # 32
# time_elapsed : 875
# total_timesteps : 524288

# # 20 (but with n_steps = 1024)
# time_elapsed : 843
# total_timesteps : 512000

# # 20 (only two rewards)
# te : 226
# tt : 512000

# load per-episode reward components
df_train_updates = m.load_train_update_metrics()

# evaluate test rollout (for df_test)
df_test, metrics = m.evaluate_test(
    save_rollout=True,
    save_plots=True,
    save_metrics=False,
)

# 1) save all plots
drl_plot.save_plots(
    df_test=df_test,
    df_train_updates=df_train_updates,
    outdir=m.logdir,
    which="all",
)

