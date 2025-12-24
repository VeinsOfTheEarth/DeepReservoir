# Use Four Corners for SPR measurements

# This dictionary is derived from Table 8.1 in the 1999 SJRIP Flow Recommendations docuement. The values
# quantify the recommended spring peak release thresholds, durations, and frequencies.
spr_targets = {'threshold_cfs' : [10000, 8000, 5000, 2500],
                'frequency' : [0.2, 0.33, 0.5, 0.8], # describes the annual frequency these thresholds should be met for at least duration days
                'duration_days' : [5, 19, 20, 10]
                }