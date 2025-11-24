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
    n_envs=20,              # try 4 first
    use_subproc_vec=True, # start with DummyVecEnv for debugging
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

# Storage plot
fig, ax, ax2 = drl_plot.plot_storage_timeseries(df_test)
drl_plot.save(fig, m.logdir / "storage_timeseries.png")

# Episode mean rewards
cb = m._reward_components_cb
df_ep = pd.DataFrame(cb.episode_history)
df_ep = df_ep.sort_values("episode_idx").set_index("episode_idx")
fig, ax = drl_plot.plot_episode_mean_rewards(df_ep)
drl_plot.save(fig, m.logdir / "episode_mean_rewards.png")

# Release timeseries
fig, ax = drl_plot.plot_release_timeseries(df_test)
drl_plot.save(fig, m.logdir / "release_timeseries.png")

# Storage: DOY stats plot
fig, ax = drl_plot.plot_storage_doy(df_test)
drl_plot.save(fig, m.logdir / "storage_doy.png")

# SStorage: spaghetti trajectories
fig, ax = drl_plot.plot_storage_doy_traces(df_test)
drl_plot.save(fig, m.logdir / "storage_doy_traces.png")

# Hydropower: DOY stats plot
fig, ax = drl_plot.plot_hydropower_doy(df_test)
drl_plot.save(fig, m.logdir / "hydropower_doy.png")

# Hydropower: spaghetti trajectories
fig, ax = drl_plot.plot_hydropower_doy_traces(df_test)
drl_plot.save(fig, m.logdir / "hydropower_doy_traces.png")
