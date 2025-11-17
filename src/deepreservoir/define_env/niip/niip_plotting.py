# -*- coding: utf-8 -*-
"""
@author: Shubhendu
"""
# ==== NIIP Daily Demand Sanity Check (365-day version) ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from niip_demand import niip_daily_demand  # <-- your function

# 1) Build DOY array (1..365)
doys = np.arange(1, 366, dtype=int)

# 2) Evaluate demand (cfs)
try:
    demand_cfs = np.asarray(niip_daily_demand(doys), dtype=float)
    if demand_cfs.shape != doys.shape:
        demand_cfs = np.array([float(niip_daily_demand(d)) for d in doys], dtype=float)
except Exception:
    demand_cfs = np.array([float(niip_daily_demand(d)) for d in doys], dtype=float)

# 3) Convert to AF/day
CFS_TO_AF_PER_DAY = 1.98211
demand_af = demand_cfs * CFS_TO_AF_PER_DAY

# 4) Save CSV
df = pd.DataFrame({
    "DOY": doys,
    "Demand_cfs": demand_cfs,
    "Demand_af": demand_af
})
df.to_csv("niip_daily_demand_365.csv", index=False)
print("Wrote niip_daily_demand_365.csv")
print(df.head(10))
print(df.iloc[45:55])   # around DOY ~50
print(df.iloc[295:305]) # around DOY ~300

# 5) Plot (cfs)
plt.figure(figsize=(10, 5))
plt.plot(doys, demand_cfs, label="NIIP Daily Demand (cfs)")
plt.axvspan(50, 300, color='gray', alpha=0.15, label="Expected demand window (50–300)")
plt.xlabel("Day of Year")
plt.ylabel("Demand (cfs)")
plt.title("NIIP Daily Demand vs DOY (365 days, cfs)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("niip_daily_demand_cfs_365.png", dpi=150, bbox_inches="tight")
plt.show()

# 6) Plot (AF/day)
plt.figure(figsize=(10, 5))
plt.plot(doys, demand_af, label="NIIP Daily Demand (af/day)", color="tab:orange")
plt.axvspan(50, 300, color='gray', alpha=0.15, label="Expected demand window (50–300)")
plt.xlabel("Day of Year")
plt.ylabel("Demand (af/day)")
plt.title("NIIP Daily Demand vs DOY (365 days, af/day)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("niip_daily_demand_af_365.png", dpi=150, bbox_inches="tight")
plt.show()

# Quick stats
season_mask = (doys >= 50) & (doys <= 300)
print("\nSanity stats:")
print(f"Nonzero days: {(demand_cfs > 0).sum()} (expected ~250)")
print(f"Mean demand in-season (cfs): {demand_cfs[season_mask].mean():.2f}")
print(f"Peak demand in-season (cfs): {demand_cfs[season_mask].max():.2f} @ DOY {doys[season_mask][np.argmax(demand_cfs[season_mask])]}")
print(f"Total season demand (af):    {demand_af[season_mask].sum():.0f}")
