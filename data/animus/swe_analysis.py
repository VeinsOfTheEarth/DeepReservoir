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
an_daily = swe_helpers.build_daily_q(cfs['Animas @ Farmington'])
q_daily = usbr["release_cfs"]  # daily

start = '2000-01-01'
end = '2100-01-01'

q_daily = q_daily.loc[pd.to_datetime(start) : pd.to_datetime(end)].copy()
swe_daily = swe_daily.loc[pd.to_datetime(start) : pd.to_datetime(end)].copy()
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
swe_helpers.plot_spe_timeline(q_daily, spe, title='Navajo: Spring Peak Events')      # shade detected event windows

swe_helpers.plot_animas_vs_prespring_storage(
    an_daily, usbr, spe_df=spe,
    storage_method="mar_mean",
    colors = {"pre": "#9ca3af", "no": "#f50b46", "spr": "#2563eb"},
    title="PSR conditions",
    annotate_wy=True,  # set True if you want WY labels
    year_split=2000,
    point_size = 80
)

swe_helpers.plot_swe_vs_prespring_storage(
    wy, usbr, spe_df=spe,
    swe_col="SWE_peak_mm",          # or another SWE metric column if you add one
    storage_method="mar_mean",
    title="Peak SWE vs pre-Spring storage",
    annotate_wy=True,
    point_size=96
)
