# -*- coding: utf-8 -*-
"""
San Juan River gage analysis
---------------------------
1) Load USGS daily and continuous (15–60 min) data.
2) Quick overview plots (daily).
3) Step-release, event-window correlation:
      SJ @ Archuleta (upstream) → SJF-An (SJ @ Farmington – Animas @ Farmington)
   - detect step/ramp windows at the dam
   - classify windows (keep/reject) with simple QC
   - compute travel-time (weighted mean lag) for a season and summarize by year
4) Peak-based correlation for longer lags:
      SJ @ Farmington (upstream) → SJ @ Four Corners (downstream)
   - detect peak windows each year; compute yearly best lag
   - report weighted mean lag across years
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Project setup
# -----------------------------------------------------------------------------
# adjust to your working repo path if needed
os.chdir(r"X:\Research\DeepReservoir\Code\DeepReservoir\data\animus")
helpers = __import__("helpers")  # local module (cleaned helpers.py)

# -----------------------------------------------------------------------------
# 0) LOAD DATA
# -----------------------------------------------------------------------------
# Daily USGS (one CSV per site)
df_daily = helpers.load_usgs_daily_dir(
    dir_path=r"X:\Research\DeepReservoir\finalize_model\gages_usgs",
    pattern="daily_*.csv",
    renamer=helpers.USGS_RENAMER,
)

# Continuous USGS, aligned to a common grid (choose freq as needed)
df15 = helpers.load_usgs_continuous_dir_grid(
    dir_path=r"X:\Research\DeepReservoir\finalize_model\gages_usgs\realtime",
    pattern="*.txt",
    renamer=helpers.USGS_CONT_RENAMER,
    freq="15min",          # set "h" for hourly
    method="resample",     # or "snap"
    agg="mean",
)

# Quick sanity: total years of daily data available per series
years_available = (df_daily.notna().sum() / 365.0).rename("years_available")
print(years_available.sort_values(ascending=False))

# -----------------------------------------------------------------------------
# 1) QUICK PLOTS (DAILY OVERVIEWS)
# -----------------------------------------------------------------------------
sj_cols   = ["SJ @ Bluff", "SJ @ Four Corners", "SJ @ Shiprock", "SJ @ Farmington", "SJ @ Archuleta"]
trib_cols = ["Animas @ Farmington", "Chinle @ MW", "La Plata @ Farmington", "Mancos @ Towaoc"]

helpers.plot_columns(
    df_daily, sj_cols, start="2000-01-01", end="2024-12-31",
    mode="overlay", tick="monthly", suptitle="San Juan River gages"
)
helpers.plot_columns(
    df_daily, trib_cols, start="2000-01-01", end="2024-12-31",
    mode="overlay", tick="monthly", suptitle="Tributary gages"
)

# Boxplots of annual maxima (drop years with >10% missing)
helpers.boxplot_annual_discharge(
    df_daily,
    ["SJ @ Bluff", "Chinle @ MW", "SJ @ Four Corners", "Mancos @ Towaoc",
     "SJ @ Shiprock", "Chaco @ Waterflow", "La Plata @ Farmington",
     "SJ @ Farmington", "Animas @ Farmington", "SJ @ Archuleta"],
    agg="max",
    start="1970-01-01", end="2024-12-31",
    title="Max Annual Discharge (1970–2024)",
    max_missing_frac=0.10,
    color_scheme="group_sj",
    add_color_legend=True,
)
plt.show()

# -----------------------------------------------------------------------------
# 2) STEP-RELEASE WINDOWS: Archuleta → (Farmington − Animas)
# -----------------------------------------------------------------------------
# Build the "San Juan minus Animas" signal at Farmington to isolate mainstem response
df15["SJF-An"] = df15["SJ @ Farmington"] - df15["Animas @ Farmington"]

start, end = "2024-04-01", "2024-08-31"
sub = df15.loc[start:end, ["SJ @ Archuleta", "SJF-An"]]
mask_global = helpers.global_nan_mask(sub)

# Detect step/ramp changes at the dam and pad a response window downstream
step_mask = helpers.build_step_event_mask(
    sub["SJ @ Archuleta"],
    method="diff", smooth="45min", direction="up",
    p=99.5, min_run="30min", pad_before="3h", pad_after="36h",
)

# Score each window and classify keep/reject (QC by r, n, slope, amplitude ratio, lag bounds)
res_steps, kept_steps, rej_steps = helpers.evaluate_step_windows(
    sub, "SJ @ Archuleta", "SJF-An", step_mask,
    mask_global=mask_global,
    lags=range(-96, 97), lag_unit="15min",
    method="pearson",
    min_n=24, min_r=0.6, slope_sign="positive",
    amp_ratio_bounds=(0.2, 6.0),
    plausible_lag=(-96, -12),
    use_diff=True,
)

# Visual sanity check
helpers.plot_columns_with_classified_windows(
    sub, ["SJ @ Archuleta", "SJF-An"],
    kept_mask=kept_steps, rejected_mask=rej_steps,
    start=start, end=end, tick="daily",
    suptitle="2024 step windows (kept=green, rejected=red)",
)
plt.show()

# Weighted lag across kept windows (hours)
if not res_steps.query("keep").empty:
    mean_lag_h = float(np.average(res_steps.query("keep")["lag_hours"],
                                  weights=res_steps.query("keep")["n_pair"]))
    print("Archuleta → (Farmington−Animas) | 2024 weighted lag (h):", mean_lag_h)

# Optional: quick seasonal (May–Aug) year-by-year summary using a coarser evaluation grid
def _seasonal_year_summary(df, months=(5, 6, 7, 8), work_freq="30min"):
    rows = []
    for yr, subyr in df.groupby(df.index.year):
        subm = subyr[subyr.index.month.isin(months)]
        if subm.empty:
            continue
        mg = helpers.global_nan_mask(subm[["SJ @ Archuleta", "SJF-An"]])
        x = subm["SJ @ Archuleta"].resample(work_freq).mean()
        y = subm["SJF-An"].resample(work_freq).mean()
        mg_r = mg.resample(work_freq).max().astype(bool)
        half = pd.Timedelta("24h"); step = pd.to_timedelta(work_freq)
        nl = int(half / step)
        stats = helpers.lagcorr_series_stats_fast(x, y, mask_global=mg_r,
                                                  lags=range(-nl, nl + 1), lag_unit=work_freq)
        best = helpers.best_lag_from_r(stats["r"])
        lag_h = float((best * step) / pd.Timedelta("1h")) if best is not None else np.nan
        rows.append(dict(year=yr, lag_hours=lag_h, r=stats["r"].get(best), n=stats["n"].get(best)))
    return pd.DataFrame(rows).sort_values("year")

seasonal_summary = _seasonal_year_summary(df15.loc["1990":"2024"])
print(seasonal_summary.dropna())

# -----------------------------------------------------------------------------
# 3) PEAK-BASED WINDOWS: Farmington → Four Corners
# -----------------------------------------------------------------------------
x_col, y_col = "SJ @ Farmington", "SJ @ Four Corners"
freq = "h"                 # hourly grid for ~1–3 day lags
lags = range(-72, 73)      # search ±3 days

# Inspect a single year and compute r(k) across all kept windows
fig, ax, res_pk, stats_pk, best_pk = helpers.plot_peakcorr_window(
    df15.resample(freq).mean(),
    x_col, y_col,
    year=2024,
    tick="daily",
    lag_unit=freq,
    lags=lags,
    smooth="6h",
    min_separation="72h",   # avoid overlapping padded windows
    prominence_q=0.80,
    pad_before="0h",
    pad_after="60h",
    min_r=0.6,
    min_n=10,
    plausible_lag=None,     # unconstrained; helpers normalize sign so +lag = y follows x
)
plt.show()

# Year-by-year summary and weighted mean lag
years = list(range(1990, 2025))
summary = helpers.peak_lag_yearly_summary(
    df15.resample(freq).mean(),
    x_col, y_col,
    years=years,
    lag_unit=freq,
    lags=lags,
    smooth="6h",
    min_separation="72h",
    prominence_q=0.80,
    pad_before="0h",
    pad_after="60h",
    min_r=0.6,
    min_n=10,
    plausible_lag=None,
)
print(summary)

wm_lag_h = helpers.weighted_mean_lag_hours(summary)
print("Farmington → Four Corners | weighted lag (hours):", wm_lag_h)
print("Farmington → Four Corners | weighted lag (days):", (wm_lag_h / 24.0) if np.isfinite(wm_lag_h) else np.nan)

# Repeat above for Bluff