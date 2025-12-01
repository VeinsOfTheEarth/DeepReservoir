import pandas as pd
from deepreservoir.drl.model import DRLModel
from pathlib import Path
from importlib import reload

reward_spec = "dam_safety:storage_band,esa_min_flow:baseline,flooding:baseline,niip:baseline,hydropower:baseline,physics:scale_penalty"
reward_spec = "dam_safety:storage_band,physics:scale_penalty"
run_dir = Path("runs/debug_parallel")

m = DRLModel(
    n_years_test=8,
    reward_spec=reward_spec,
    logdir=run_dir,
    device="cpu",
    n_envs=1,              # try 4 first
    use_subproc_vec=False, # start with DummyVecEnv for debugging
)

m.train(
    total_timesteps=500_000,
    device="cpu",
    n_steps=1024,
    batch_size=4096,
    track_reward_components=True,
    gamma = 0.9995
)

# 4
elapsed : 2938
timesteps : 507904

# 16
time_elapsed : 964
total_timesteps : 524288

# 32
time_elapsed : 875
total_timesteps : 524288

# 20 (but with n_steps = 1024)
time_elapsed : 843
total_timesteps : 512000

# 20 (only two rewards)
te : 226
tt : 512000
# m.load_model("last_model")

df_test, metrics = m.evaluate_test(
    save_rollout=True,   # if you want to overwrite the parquet
    save_plots=False,
    save_metrics=False,
)

from deepreservoir.drl import plotting as drl_plot
reload(drl_plot)

# Episode rewards
cb = m._reward_components_cb
df_ep = pd.DataFrame(cb.episode_history).sort_values("episode_idx")

# 1) Save all plots
drl_plot.save_plots(
    df_test=df_test,
    df_ep=df_ep,
    outdir=m.logdir,
    which="all",
)

# 2) If you want to save a subset of plots:
# drl_plot.save_plots(
#     df_test=df_test,
#     df_ep=df_ep,
#     outdir=m.logdir,
#     which=["core", "doy"],  # groups
# )
