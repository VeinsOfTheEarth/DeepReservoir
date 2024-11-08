# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:33:18 2024

@author: 318596
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
import pickle


def linear_interpolator(x, y, nbreaks, plot=True):

    """
    Returns a scipy interpolated function object.
    """
   
    # Define breakpoints using quantiles
    breakpoints = np.quantile(x, np.linspace(0, 1, nbreaks+1))  # 10 pieces, 11 breakpoints
    
    # Calculate the mean y-values at these breakpoints
    x_piecewise = []
    y_piecewise = []
    for i in range(len(breakpoints) - 1):
        mask = (x >= breakpoints[i]) & (x < breakpoints[i+1])
        x_piecewise.append(np.mean(x[mask]))  # Mean x in this interval
        y_piecewise.append(np.mean(y[mask]))  # Mean y in this interval
    
    # Set up piecewise linear interpolation with extrapolation
    piecewise_model = interp1d(
        x_piecewise, y_piecewise, kind="linear", bounds_error=False, fill_value="extrapolate"
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
path = r"X:\Research\DeepReservoir\Code\reservoir\data\elevation_area_storage_relationships\elevation_storage_area_2019.csv"
df = pd.read_csv(path)

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
with open(r"X:\Research\DeepReservoir\Code\reservoir\data\elevation_area_storage_relationships\2019_elevation_area_capacity.pkl", "wb") as file:
    models = pickle.dump({'elevation_to_area': e_to_a,
                          'area_to_elevation': a_to_e,
                          'elevation_to_capacity': e_to_c,
                          'capacity_to_elevation': c_to_e}, file)

