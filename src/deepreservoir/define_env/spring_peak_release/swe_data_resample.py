import os
import pandas as pd
from deepreservoir.define_env.downstream import helpers
from deepreservoir.define_env.spring_peak_release import swe_helpers

# Resample hourly SWE data from ERA5-Land (GEE-sampled) to daily
# Hourly data are not stored in the git repoistory
path_out = r'X:\Research\DeepReservoir\Code\DeepReservoir\data\snow_water_equivalent'
name = 'Animas' # choose from Animas, LosPinos, Piedra, Spring, UpperSJ

# Ready hourly (not in repo)
df = pd.read_parquet(r"X:\Research\DeepReservoir\data\swe\{}.parquet".format(name))

# Convert to daily and store in repo
df['date'] = pd.to_datetime(df['date'])
swe_daily = swe_helpers.build_daily_swe(df)  # meters
swe_daily.to_csv('data/snow_water_equivalent/{}_swe_daily.csv'.format(name))
