# -*- coding: utf-8 -*-
"""
Build elevation–area–capacity interpolators and export to a pickle.

Important modeling convention (Navajo):
  - The supplied 'Capacity' / 'Storage' is **volume above deadpool**.
  - Deadpool elevation is 5775.0 ft and corresponds to 0.0 acre-feet.
  - Below 5775.0 ft, capacity is defined as 0.0 (no extrapolated negatives).

This file builds *clamped* linear interpolators so the environment physics is
numerically stable and consistent at the deadpool boundary:
  - elevation_to_capacity(elev <= 5775) -> 0.0
  - capacity_to_elevation(cap <= 0)     -> 5775.0

The resulting pickle is used by `deepreservoir.drl.environs.NavajoReservoirEnv`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from deepreservoir.data.metadata import project_metadata


DEADPOOL_ELEV_FT = 5775.0


def _first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"None of the candidate columns were found. Candidates={list(candidates)}. "
        f"Found={list(df.columns)}"
    )


def _dedupe_sorted_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by x and collapse duplicate x via mean(y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Collapse duplicates (robust to float rounding artifacts)
    xu, inv = np.unique(x, return_inverse=True)
    yu = np.zeros_like(xu, dtype=float)
    counts = np.zeros_like(xu, dtype=float)
    np.add.at(yu, inv, y)
    np.add.at(counts, inv, 1.0)
    yu /= np.maximum(counts, 1.0)

    return xu, yu


def _clamped_linear_interp(x: np.ndarray, y: np.ndarray, *, left: float, right: float):
    """interp1d with constant clamp outside domain (no extrapolation)."""
    return interp1d(
        x,
        y,
        kind="linear",
        bounds_error=False,
        fill_value=(float(left), float(right)),
        assume_sorted=True,
    )


def build_and_save(*, plot: bool = False) -> Path:
    m = project_metadata()

    # Correct: csv is a TABLE, pickle is a PARAMETER file.
    path_eas_csv = Path(m.path("elev_area_storage_data"))
    path_eas_pickle = Path(m.path("elev_area_storage_pickle"))

    df = pd.read_csv(path_eas_csv)

    # Robust column detection (we have seen minor header variants).
    elev_col = _first_existing_col(df, ["Elevation (ft)", "Elevation (feet)", "elev_ft"])
    cap_col = _first_existing_col(df, ["Capacity (ac-ft)", "Capacity (af)", "Capacity (acft)", "capacity_af"])
    area_col = _first_existing_col(df, ["Area (ac)", "Area (acres)", "area_ac"])

    e = df[elev_col].to_numpy(dtype=float)
    c = df[cap_col].to_numpy(dtype=float)
    a = df[area_col].to_numpy(dtype=float)

    # Drop NaNs
    ok = np.isfinite(e) & np.isfinite(c) & np.isfinite(a)
    e, c, a = e[ok], c[ok], a[ok]

    # Enforce the deadpool convention at 5775 ft: capacity == 0.
    # If the table already includes this (it should), this is a no-op.
    # If the table starts above deadpool, we insert an anchor point.
    if not np.any(np.isclose(e, DEADPOOL_ELEV_FT)):
        e = np.append(e, DEADPOOL_ELEV_FT)
        c = np.append(c, 0.0)
        # area at deadpool is ambiguous without data; use min area as conservative.
        a = np.append(a, float(np.nanmin(a)))

    # Ensure no negative capacities in the training table; clamp just in case.
    c = np.maximum(c, 0.0)

    # Build clamped interpolators in both directions.
    e_u, c_u = _dedupe_sorted_xy(e, c)
    e_u2, a_u = _dedupe_sorted_xy(e, a)

    # Elev -> Capacity: clamp below 5775 to 0; clamp above to max.
    e_to_c = _clamped_linear_interp(e_u, c_u, left=0.0, right=float(np.max(c_u)))

    # Capacity -> Elev: invert using the (monotone) capacity axis.
    c_u2, e_for_c = _dedupe_sorted_xy(c, e)
    c_to_e = _clamped_linear_interp(c_u2, e_for_c, left=DEADPOOL_ELEV_FT, right=float(np.max(e_for_c)))

    # Elev -> Area: clamp below to area at deadpool (min elev) and above to max.
    area_at_deadpool = float(a_u[np.argmin(e_u2)])
    e_to_a = _clamped_linear_interp(e_u2, a_u, left=area_at_deadpool, right=float(np.max(a_u)))

    # Area -> Elev: invert using area axis (monotone in practice for this table).
    a_u2, e_for_a = _dedupe_sorted_xy(a, e)
    a_to_e = _clamped_linear_interp(a_u2, e_for_a, left=float(np.min(e_for_a)), right=float(np.max(e_for_a)))

    # Optional quick plot for sanity (off by default so this works headless).
    if plot:
        import matplotlib.pyplot as plt

        plt.close("all")
        xs = np.linspace(min(e_u), max(e_u), 500)
        plt.figure()
        plt.plot(e_u, c_u, ".", ms=2, label="table")
        plt.plot(xs, e_to_c(xs), "-", label="e->c")
        plt.axvline(DEADPOOL_ELEV_FT, ls="--")
        plt.legend()
        plt.title("Elevation → Capacity (clamped)")
        plt.xlabel("Elevation (ft)")
        plt.ylabel("Capacity (ac-ft)")
        plt.show()

    path_eas_pickle.parent.mkdir(parents=True, exist_ok=True)
    with open(path_eas_pickle, "wb") as f:
        pickle.dump(
            {
                "elevation_to_area": e_to_a,
                "area_to_elevation": a_to_e,
                "elevation_to_capacity": e_to_c,
                "capacity_to_elevation": c_to_e,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"Wrote: {path_eas_pickle}")
    return path_eas_pickle


if __name__ == "__main__":
    build_and_save(plot=False)
