import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

path_csv = r"X:\Research\DeepReservoir\navajo_irr_project\NAVAJOINDIANIRRIGATIONPROJECT07-17-2025T13.21.47.csv"
df = pd.read_csv(path_csv)

dates, flows = [], []
for i in df.index[:-4]:
    dates.append(pd.to_datetime(i[0]))
    flows.append(float(i[1]))

df = pd.DataFrame(index=dates, data={'Flow (cfs)': flows})

df.plot(); plt.show()

# Compute the average curve
# Filter to 1990 and later
df_1990 = df[df.index >= "1990-01-01"]
# Add day-of-year column
df_1990["doy"] = df_1990.index.dayofyear
# Compute mean flow for each day-of-year
doy_avg = df_1990.groupby("doy")["Flow (cfs)"].mean()

plt.figure(figsize=(10, 4))
plt.plot(doy_avg.index, doy_avg.values)
plt.xlabel("Day of Year")
plt.ylabel("Average Flow (cfs)")
plt.title("Average Daily Flow (1990–Present)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Fit a polynomial to the DOY average
# Restrict to DOY 50–300
mask = (doy_avg.index >= 50) & (doy_avg.index <= 300)
x_fit = doy_avg.index[mask]
y_fit = doy_avg.values[mask]

# Fit a smoothing spline (s is the smoothing factor; lower means less smoothing)
spline = UnivariateSpline(x_fit, y_fit, s=100000.0)  # use s=0 for interpolating spline, increase s to smooth

# Define the callable seasonal function
def seasonal_avg_flow(doy):
    """
    Returns seasonal average flow for given day-of-year(s), using a spline fit.
    - Outside DOY 50–300: returns 0
    - Negative values are clipped to 0
    """
    doy = np.asarray(doy)
    raw = np.where((doy >= 50) & (doy <= 300), spline(doy), 0.0)
    return np.maximum(raw, 0.0)

days = np.arange(1, 367)
flow = seasonal_avg_flow(days)

plt.figure(figsize=(10, 4))
plt.plot(days, doy_avg, label='Average')
plt.plot(days, flow, label="Seasonal Flow Fit")
plt.xlabel("Day of Year")
plt.ylabel("Flow (cfs)")
plt.legend()
plt.title("Navajo Irrigation Function based on 1990-2025 DOY averages")
plt.grid(True)
plt.tight_layout()
plt.show()











# Annual volumes
annual_volume = df.resample("Y")["Flow (cfs)"].sum()
# 1 cfs-day = 1.9835 acre-feet
annual_volume_af = annual_volume * 1.9835

annual_volume_af.index = annual_volume_af.index.year

plt.figure(figsize=(10, 4))
annual_volume_af.plot(kind="bar")
plt.ylabel("Annual Volume (acre-feet)")
plt.title("Total Annual Streamflow")
plt.tight_layout()
plt.show()
