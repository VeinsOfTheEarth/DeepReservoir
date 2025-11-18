# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:33:18 2024

Builds elevation-area-storage interpolatable functions.
Exports a pickled set of parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
import pickle
from deepreservoir.data.loader import DataPaths


def linear_interpolator(x, y, nbreaks, plot=True):
    x = np.asarray(x)
    y = np.asarray(y)

    breakpoints = np.quantile(x, np.linspace(0, 1, nbreaks + 1))

    x_piecewise = []
    y_piecewise = []
    for i in range(len(breakpoints) - 1):
        mask = (x >= breakpoints[i]) & (x < breakpoints[i + 1])
        if not np.any(mask):
            continue  # skip empty bin
        x_piecewise.append(np.mean(x[mask]))
        y_piecewise.append(np.mean(y[mask]))

    x_piecewise = np.asarray(x_piecewise)
    y_piecewise = np.asarray(y_piecewise)

    piecewise_model = interp1d(
        x_piecewise,
        y_piecewise,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    
    # Generate fitted y values for the actual x values
    y_fitted = piecewise_model(x)
    
    # Compute R-squared value
    r_squared = r2_score(y, y_fitted)
    
    # Compute average percent error
    percent_errors = np.abs((y - y_fitted) / y) * 100
    average_percent_error = np.mean(percent_errors)
    
    # Print R-squared and average percent error
    print("R-squared:", r_squared)
    print("Average Percent Error:", average_percent_error)
    
    if plot is True:
        plt.close('all')
        # Plot original data and interpolated curve
        x_fit = np.linspace(x.min(), x.max(), 1000)  # Points for a smooth line
        y_fit = piecewise_model(x_fit)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'o', label="Original Data", markersize=5)
        plt.plot(x_fit, y_fit, '-', label="Piecewise Linear Interpolation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Piecewise Linear Interpolation Fit")
        # Position the legend outside the plot to the right
        plt.legend(loc="upper left")
        
        # Add R-squared and average percent error in the lower-left corner to avoid the legend
        plt.text(
            0.75, 0.05, 
            f"R-squared: {r_squared:.3f}\nAvg. Percent Error: {average_percent_error:.2f}%", 
            transform=plt.gca().transAxes, 
            fontsize=10, 
            verticalalignment='bottom', 
            bbox=dict(facecolor='white', alpha=0.5)
        )
        
        plt.tight_layout()  # Adjust layout to make room for the legend
        plt.show()
        
    return piecewise_model

# Load data
df = pd.read_csv(DataPaths.elev_area_storage_csv)

# Extract values from the DataFrame
e = df["Elevation (ft)"].values
a = df["Area (ac)"].values
c = df["Capacity (ac-ft)"].values

# Number of pieces
n = 100

e_to_a = linear_interpolator(e, a, 100)
a_to_e = linear_interpolator(a, e, 100)
e_to_c = linear_interpolator(e, c, 100)
c_to_e = linear_interpolator(c, e, 100)

## Store as pickle
with open(DataPaths.elev_area_storage_pickle, "wb") as file:
    pickle.dump(
        {
            "elevation_to_area": e_to_a,
            "area_to_elevation": a_to_e,
            "elevation_to_capacity": e_to_c,
            "capacity_to_elevation": c_to_e,
        },
        file,
    )
