import os
import pandas as pd
from importlib import reload
os.chdir(r"X:\Research\DeepReservoir\Code\DeepReservoir\data\animus")
helpers = __import__("helpers")  # local module (cleaned helpers.py)
swe_helpers = __import__("swe_helpers")  # local module (cleaned helpers.py)

# Load SWE
path_an_swe = r"X:\Research\DeepReservoir\finalize_model\swe\gee_csvs\combined\sites_parquet\Animas.parquet"
swe_an = pd.read_parquet(path_an_swe)
swe_an.head()

# Load streamflow
cfs = helpers.load_usgs_continuous_dir_grid() # 15 minute resampling
cfs_an = cfs['Animas @ Farmington']
cfs_an.head()
cfs_navajo = cfs['SJ @ Archuleta']
cfs_navajo.head()

usbr = swe_helpers.load_usbr_navajo_daily()
# -------------------- Run for Animas
# SWE: your swe_an already loaded; ensure datetime + sort
swe_an = swe_an.copy()
swe_an['date'] = pd.to_datetime(swe_an['date'])
swe_daily = swe_helpers.build_daily_swe(swe_an)  # meters

# Flow (15-min) -> daily mean with coverage rule

# Build WY metrics
q15 = cfs['Animas @ Farmington']

q_daily = swe_helpers.build_daily_q(q15)
q_daily = usbr["release_cfs"]  # daily

wy = swe_helpers.assemble_wy_metrics(swe_daily, q_daily)

# Compute relationships for mean and peak spring flow
rel_mean, yhat_mean, ycv_mean = swe_helpers.correlations_and_fit(wy, y_col='Q_AprJul_mean_cfs', x_col='SWE_peak_mm')
rel_peak, yhat_peak, ycv_peak = swe_helpers.correlations_and_fit(wy, y_col='Q_AprJul_peak_cfs', x_col='SWE_peak_mm')
rel_total, yhat_total, ycv_total = swe_helpers.correlations_and_fit(
    wy, y_col='Q_AprJul_total_acft_scaled', x_col='SWE_peak_mm'
)
rel_mean, yhat_mean, ycv_mean = swe_helpers.correlations_and_fit(
    wy, y_col='Q_AprJul_mean_cfs', x_col='SWE_peak_mm'
)
rel_peak3d, yhat_peak3d, ycv_peak3d = swe_helpers.correlations_and_fit(
    wy, y_col='Q_AprJul_peak3d_cfs', x_col='SWE_peak_mm'
)

# -------------------- Plotting

# ------ Detect years that Spring Peak was attempted at Navajo
params = swe_helpers.SPEParams(
    thresh_pctl=0.90,   # 90th percentile threshold
    min_dur_days=4,     # need at least 4 days above threshold
    max_cv_top=0.20,    # plateau-ish
    smooth_days=3,      # 3-day smooth
    require_core_peak=True
)

spe = swe_helpers.detect_spe_all_years(usbr, params=params, prefer_col="release_cfs")
print(spe.index.values[spe['classified_SPE']==True])
# 2) Plot with shading
swe_helpers.plot_spe_timeline(usbr, spe)      # shade detected event windows

swe_helpers.plot_animas_vs_prespring_storage(cfs_an, usbr, spe_df=spe, storage_method="mar_mean")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Data
d = wy[['SWE_peak_mm', 'Q_AprJul_mean_cfs']].dropna().copy()

# SPR flags from `spe` (default False if missing)
spr_flags = d.index.to_series().map(
    lambda wy_: bool(spe.loc[wy_, 'classified_SPE']) if ('spe' in globals()
        and spe is not None and not spe.empty and wy_ in spe.index
        and 'classified_SPE' in spe.columns) else False
)
d['spr'] = spr_flags.values

# Colors
color_spr = 'tab:blue'
color_no  = '#f4a261'
point_colors = np.where(d['spr'].to_numpy(), color_spr, color_no)

# Regression (all points)
x = d['SWE_peak_mm'].to_numpy()
y = d['Q_AprJul_mean_cfs'].to_numpy()
a, b = np.polyfit(x, y, 1)
xx = np.linspace(x.min(), x.max(), 300)
yy = a*xx + b
r = np.corrcoef(x, y)[0, 1]

# Plot
plt.figure(figsize=(7.8, 5.6))
ax = plt.gca()
ax.scatter(x, y, s=38, c=point_colors, alpha=0.9,
           edgecolor='white', linewidth=0.7)
line_handle, = ax.plot(xx, yy, lw=2.2, color='#333333', alpha=0.85,
                       label=f'OLS fit: y = {a:.2f}x + {b:.0f}')

# Grid / cosmetics
ax.minorticks_on()
ax.grid(which="major", linestyle="-", alpha=0.25)
ax.grid(which="minor", linestyle=":", alpha=0.12)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.tick_params(axis="both", which="both", length=0)

ax.set_xlabel("Peak SWE (mm)")
ax.set_ylabel("Apr–Jul mean Q (cfs)")
ax.set_title(f"Peak SWE vs Spring Mean Flow\nr = {r:.2f}, n = {len(d)}")

# Legend
pt_spr = Line2D([], [], marker='o', linestyle='None', markersize=7,
                markerfacecolor=color_spr, markeredgecolor='white',
                markeredgewidth=0.7, label='SPR attempted')
pt_no  = Line2D([], [], marker='o', linestyle='None', markersize=7,
                markerfacecolor=color_no, markeredgecolor='white',
                markeredgewidth=0.7, label='No SPR')
ax.legend([pt_spr, pt_no, line_handle],
          ['SPR attempted', 'No SPR', 'OLS fit'],
          frameon=False, loc='best')

plt.tight_layout()
plt.show()
