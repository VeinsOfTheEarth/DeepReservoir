# navajo_model.py
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Union, Sequence
from scipy.interpolate import interp1d

# -------------------------------------------------------------------
# Load single-eta parameter from pickle
# -------------------------------------------------------------------
path_model_params = Path(r"X:\Research\DeepReservoir\Hydropower\parameters.pkl")

def _load_eta_from_pickle(pkl_path: Path) -> float:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "eta_eff" in obj:
        return float(obj["eta_eff"])
    raise ValueError("Unsupported parameters.pkl format (expected dict with 'eta_eff').")

_eta_loaded: Optional[float] = None
if path_model_params.exists():
    try:
        _eta_loaded = _load_eta_from_pickle(path_model_params)
    except Exception:
        _eta_loaded = None

# -------------------------------------------------------------------
# Internal tailwater model
# -------------------------------------------------------------------
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

# Constants
_RHO = 1000.0
_G = 9.81
_CFS_TO_CMS = 0.0283168
_FT_TO_M = 0.3048
_TURBINE_LIMIT_CFS = 1300.0
_PLANT_CAPACITY_MW = 32.0

# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def navajo_power_generation_model(
    cfs_values: Union[float, Sequence[float]],
    elevation_ft: Union[float, Sequence[float]],
    eta_eff: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Predict daily energy production (MWh) given releases (cfs) and reservoir
    elevations (feet), using a single global efficiency eta.

    Parameters
    ----------
    cfs_values : float or array-like
        Releases in cubic feet per second.
    elevation_ft : float or array-like
        Reservoir elevation in feet.
    eta_eff : float, optional
        Efficiency to use. If None, loads from parameters.pkl.

    Returns
    -------
    energy_MWh : float or np.ndarray
        Daily energy production in MWh.
    """
    if eta_eff is None:
        if _eta_loaded is None:
            raise RuntimeError("No eta provided and could not load from parameters.pkl.")
        eta_eff = _eta_loaded

    q_cfs = np.asarray(cfs_values, dtype=float)
    elev_ft = np.asarray(elevation_ft, dtype=float)

    # Tailwater and head
    tw_ft = _tailwater_model(q_cfs)
    head_m = (elev_ft - tw_ft) * _FT_TO_M
    head_m = np.clip(head_m, 0, None)

    # Flow to m³/s with turbine limit
    q_cfs_capped = np.clip(q_cfs, 0, _TURBINE_LIMIT_CFS)
    q_cms = q_cfs_capped * _CFS_TO_CMS

    power_MW = eta_eff * _RHO * _G * q_cms * head_m / 1e6
    power_MW = np.minimum(power_MW, _PLANT_CAPACITY_MW)
    energy_MWh = power_MW * 24.0

    # Return scalar if scalar input
    return energy_MWh if energy_MWh.ndim > 0 and energy_MWh.size > 1 else float(energy_MWh)
