# Use Four Corners for SPR measurements

import pandas as pd

# This dictionary is derived from Table 8.1 in the 1999 SJRIP Flow Recommendations docuement. The values
# quantify the recommended spring peak release thresholds, durations, and frequencies.
spr_targets = {'threshold_cfs' : [10000, 8000, 5000, 2500],
                'frequency' : [0.2, 0.33, 0.5, 0.8], # describes the annual frequency these thresholds should be met for at least duration days
                'duration_days' : [5, 19, 20, 10]
                }

# Create daily SWE for Animas and UpperSJ
path_hourly_swes = r'X:\Research\DeepReservoir\data\swe'
df = pd.read_parquet(r"X:\Research\DeepReservoir\data\swe\Animas.parquet")

from deepreservoir.data.loader import NavajoData
nd = NavajoData()
nd.load_all(model_data=True)
nd.tables["model_data"].filter(regex="animas_swe_m|uppersj_swe_m").head()
