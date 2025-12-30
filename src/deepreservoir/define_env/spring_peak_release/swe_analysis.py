import os
import pandas as pd
from importlib import reload
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl

from deepreservoir.define_env.downstream import helpers
from deepreservoir.define_env.spring_peak_release import swe_helpers
from deepreservoir.data.loader import NavajoData # for daily data

# Load daily data
nd = NavajoData()
nd.load_all(model_data=False)
dailies = nd.tables['reservoir']
q_daily = dailies['release_cfs']
swe_an = nd.tables['swe_animas']
an_daily = nd.tables['animas_farmington']

# Clip to years of interest
start = '1980-01-01'
end = '2100-01-01'
q_daily = q_daily.loc[pd.to_datetime(start) : pd.to_datetime(end)].copy()
swe_an = swe_an.loc[pd.to_datetime(start) : pd.to_datetime(end)].copy()
an_daily = an_daily.loc[pd.to_datetime(start) : pd.to_datetime(end)].copy()

# Build SWE WY metrics (includes SWE_Feb_mean_mm, SWE_Mar1_mm, SWE_peak_mm, ...)
wy = swe_helpers.assemble_wy_metrics(
    swe_an, an_daily,
    swe_col="animas_swe_m",
    q_col="animas_farmington_q_cfs",
)

spring_start = (3,1) # (month, day)
spring_end = (7,31) # (month, day)
spe = swe_helpers.detect_spr_absolute(dailies, threshold_cfs=3500, min_days=3,
                                      spring_start=spring_start, spring_end=spring_end,
                                      prefer_col="release_cfs")


# Plot SPR detection timeline 
swe_helpers.plot_spe_timeline(
    dailies, spe,
    shade_mode="success_spring",
    spring_start=spring_start, spring_end=spring_end,
    year_min=2000,   # plotting only
    title="Navajo: Spring Peak Events"
)



# Discriminant scatter
swe_helpers.scatter_storage_vs_swe(
    wy, dailies, spe_df=spe,
    storage_method="feb_mean",
    swe_col="SWE_Feb_max_mm",
    year_split=2000, annotate_wy=True,
    x_label="Reservoir storage (af) — February mean",
    y_label="Animas Basin SWE peak through Mar 1 (mm)",
    title="SPR  discriminator"
)


# Find the discriminating line
# 1) Build x (Feb-mean storage) and y (choose your SWE metric)
x_series = swe_helpers.prespring_storage_by_wy(dailies, method="feb_mean")  # from your USBR daily
y_series = wy["SWE_Feb_max_mm"]   # or "SWE_peak_by_Mar1_mm"

# 2) Align and keep WY >= 2000
df_xy = pd.concat([x_series.rename("x"), y_series.rename("y")], axis=1).dropna()
df_xy = df_xy.loc[df_xy.index >= 2000]

# 3) Labels: attempted SPR (blue) from your 'spe' DataFrame
attempted = spe["classified_SPE"].reindex(df_xy.index).fillna(False).astype(bool)

# 4) Guess-and-check parameter fit
out = swe_helpers.plot_sigmoid_rule(
    x=df_xy["x"], y=df_xy["y"],
    attempted=attempted,
    s=0.1e6,                 # ← tweak this one number to match your purple curve
    x0 = 1250407, # (np.meadian(df_xy["x}"]))
    y_high = 200,
    y_low = 10,
    x_label="Reservoir storage (af) — February mean",
    y_label="Animas SWE peak through Mar 1 (mm)",
    title="SPR discriminator",
    annotate=True, labels=df_xy.index
)
plt.show()

# continuous OI
p = swe_helpers.SigmoidRuleParams(x0=out['x0'], s=out['s'], y_low=out['y_low'], y_high=out['y_high'])

# Choose the “feel”: set m0 and beta
beta = swe_helpers.beta_from_target(omega_on_line=0.75, oi_at_pos_m0=0.90)    # ≈ 0.916
out = swe_helpers.plot_oi_scatter(
    x=df_xy['x'], y=df_xy['y'], 
    params=p,
    attempted=attempted,
    m0=40.0, beta=beta, omega_on_line=0.75,
    annotate=True, 
    labels=df_xy['x'].index,
    x_label="Reservoir storage (af) — February mean",
    y_label="Animas SWE peak through Mar 1 (mm)",
    title="SPR Opportunity Index (colored by OI)",
    contour_levels=(0.5, 0.75, 0.9)
)
plt.show()


base = mpl.colormaps["Blues"]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "Blues_trunc", base(np.linspace(0.00, 0.85, 256))  # top 15% removed
)

swe_helpers.plot_oi_field(
     x=df_xy['x'], y=df_xy['y'], params=p,
    attempted=attempted, labels=df_xy['x'].index, annotate=True,
    # # NEW label controls
    # label_fontsize: int = 11,
    # label_weight: str = "bold",
    # label_color: str = "black",
    # label_outline_color: str = "white",
    # label_outline_width: float = 3.0,
    m0=40.0, beta=beta, omega_on_line=0.75,    # feel of the map
    gridsize=(500, 500), alpha=0.85,           # density + transparency
    xlim=(df_xy['x'].min()*0.95, df_xy['x'].max()*1.05),
    ylim=(0, max(320, float(df_xy['y'].max()*1.05))),   # force down to y=0
    cmap=cmap,                              
    cbar=True, cbar_kw={"orientation": "vertical", "shrink": 0.9},
    title=None,
    x_label="Feb. mean storage (af)",
    y_label="Max SWE by March 1 (mm)",
)
plt.show()

## After tuning parameters to desired, this section saves them
from deepreservoir.data.metadata import project_metadata
pm = project_metadata()
out = pm.path("params.spr_oi_params_json")
out.parent.mkdir(parents=True, exist_ok=True)
import json
json.dump({
    "boundary": {"x0": float(p.x0), "s": float(p.s), "y_low": float(p.y_low), "y_high": float(p.y_high)},
    "m0": 40.0, "beta": float(beta), "omega_on_line": 0.75,
    "storage_method": "feb_mean",
    "swe_metric": "SWE_peak_by_Mar1_mm"
}, open(out, "w"), indent=2)
print("Wrote OI params ->", out)
