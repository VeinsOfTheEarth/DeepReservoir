# reservoir.py
"""
Entry point for DeepReservoir experiments:
- loads data via NavajoData
- builds reward (from registry)
- builds Gymnasium env
- trains & evaluates an SB3 agent
"""
from importlib import reload
from pathlib import Path
import argparse
import pandas as pd
from stable_baselines3 import PPO

from deepreservoir.data import loader
from deepreservoir.drl import helpers

# Load required data
reload(loader)
nav_data = loader.NavajoData()
alldata = nav_data.load_all(include_cont_streamflow=False, model_data=True)

data      = alldata["model_data"]        # raw
datanorm = alldata["model_data_norm"]   # normalized
norm_stats      = alldata["model_norm_stats"]  # mean/std table

# Split into train/test sets
# Everything not used for testing will be used for training
# Testing uses the most recent n_years_testing available
n_years_test = 10
data_train, data_test = helpers.split_train_test_by_water_year(data, n_years_test=n_years_test)
# Use the same index split for the normalized data
datanorm_train, datanorm_test = datanorm.loc[data_train.index], datanorm.loc[data_test.index]
