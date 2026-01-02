from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from deepreservoir.drl import model as drl_model
from importlib import reload

reward_spec = "dam_safety:storage_band,esa_min_flow:baseline,flooding:baseline,niip:baseline,hydropower:baseline,physics:scale_penalty"
run_dir = Path("runs/debug_ipy")

m = drl_model.DRLModel(
    n_years_test=8,
    reward_spec=reward_spec,
    logdir=run_dir,
    device="cpu",   # default; can override in train()
)

m.train(
    total_timesteps=50000,
    device="cpu",
    n_steps=4096,
    batch_size=4096,
    n_epochs=400,
    track_reward_components=True,
    gamma = 0.999
)

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

# Training update mean rewards
df_train_updates = m.train_update_metrics_ if m.train_update_metrics_ is not None else m.load_train_update_metrics()
if df_train_updates is not None and not df_train_updates.empty:
    fig, ax = drl_plot.plot_train_update_mean_rewards(df_train_updates)
    drl_plot.save(fig, m.logdir / "train_update_mean_rewards.png")

# Release timeseries
fig, ax = drl_plot.plot_release_timeseries(df_test)
drl_plot.save(fig, m.logdir / "release_timeseries.png")



