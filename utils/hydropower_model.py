## USAGE
# You may need to update the path to the parameters.pkl file depending on how your paths/environment is structured.
# from navajo_model import navajo_power_generation_model

# # Provide your own input data and optimization result:
# energy = navajo_power_generation_model(dates, flows, elevations, result_eta)
# dates are pd.datetime, flows are cfs, elevations are reservoir elevations in feet.


import numpy as np
import pandas as pd
import pickle
from scipy.interpolate import CubicSpline, interp1d

# Load model parameters only once
path_model_params = "parameters.pkl"
with open(path_model_params, "rb") as f:
    _params = pickle.load(f)


# Internal tailwater model (hidden from import)
def _create_tailwater_model():
    data_points = [
        (0, 5711.7),
        (2000, 5712.3),
        (4000, 5712.9),
        (8000, 5714.0),
        (16000, 5716.0),
        (24000, 5718.1),
        (32000, 5720.3),
        (40000, 5721.5),
    ]
    x, y = zip(*data_points)
    return interp1d(x, y, kind="linear", fill_value="extrapolate")


_tailwater_model = _create_tailwater_model()


def navajo_power_generation_model(dates, cfs_values, elevation_ft):
    """
    Predict daily energy production from date(s), release(s), and reservoir elevation(s)
    using fitted monthly efficiency splines by era.

    Parameters
    ----------
    dates : str, pd.Timestamp, or array-like of datetime-like
    cfs_values : float or array-like
    elevation_ft : float or array-like

    Returns
    -------
    energy_MWh : float or np.ndarray
    """
    # Vectorize inputs
    dates = pd.to_datetime(dates)
    cfs_values = np.asarray(cfs_values)
    elevation_ft = np.asarray(elevation_ft)

    # Determine era
    years = pd.DatetimeIndex(dates).year
    day_of_year = pd.DatetimeIndex(dates).dayofyear

    # Extract eta values and build splines
    eta_pre2010 = np.append(_params.x[:12], _params.x[0])
    eta_2010s = np.append(_params.x[12:24], _params.x[12])
    eta_post2020 = np.append(_params.x[24:], _params.x[24])

    knot_days = np.linspace(1, 366, 13, endpoint=True)

    spline_pre2010 = CubicSpline(knot_days, eta_pre2010, bc_type="periodic")
    spline_2010s = CubicSpline(knot_days, eta_2010s, bc_type="periodic")
    spline_post2020 = CubicSpline(knot_days, eta_post2020, bc_type="periodic")

    era_spline = np.where(
        years < 2010,
        spline_pre2010(day_of_year),
        np.where(years < 2020, spline_2010s(day_of_year), spline_post2020(day_of_year)),
    )

    eta = np.clip(era_spline, 0, 0.95)

    # Calculate head
    tailwater_ft = _tailwater_model(cfs_values)
    head_m = (elevation_ft - tailwater_ft) * 0.3048
    head_m = np.clip(head_m, 0, None)

    # Flow in m3/s
    Q_m3s = np.clip(cfs_values, 0, 1300) * 0.0283168

    rho = 1000
    g = 9.81
    power_MW = eta * rho * g * Q_m3s * head_m / 1e6
    power_MW = np.minimum(power_MW, 32)
    energy_MWh = power_MW * 24

    return energy_MWh if isinstance(energy_MWh, np.ndarray) else float(energy_MWh)
