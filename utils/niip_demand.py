# Use:
# from niip_demand import niip_daily_demand
# doy = 100 # example
# demand_in_cfs = niip_daily_demand(doy)

from scipy.interpolate import UnivariateSpline
import numpy as np
import pickle

with open("niip_demand_spline.pkl", "rb") as f:
    spline_data = pickle.load(f)

spline = UnivariateSpline._from_tck((spline_data["t"], spline_data["c"], spline_data["k"]))

def niip_daily_demand(doy):
    # Computes daily demand in cfs
    doy = np.asarray(doy)
    raw = np.where((doy >= 50) & (doy <= 300), spline(doy), 0.0)
    return np.maximum(raw, 0.0)
