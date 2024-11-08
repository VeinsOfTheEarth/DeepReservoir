# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:37:26 2024

@author: 318596
"""

import pickle

def elevation_area_storage(path_pickle=r"X:\Research\DeepReservoir\Code\reservoir\data\elevation_area_storage_relationships\2019_elevation_area_capacity.pkl"):
    """
    Loads a pickle file with four pre-computed models that convert between
    elevation, surface area, and capacity (storage). 
    
    Elevation is in feet.
    Area is in acres.
    Capacity in in acre-feet.
    """
    with open(path_pickle, "rb") as file:
        models = pickle.load(file)

    return models