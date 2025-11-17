"""
This file walks through the analysis used to show reasonable models for the two required downstream
gages: SJ @ Farmington and SJ @ Bluff.
"""
# Load main data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from importlib import reload
reload(loader)
from deepreservoir.data import loader

# Data
dataclass = loader.NavajoData()
data = dataclass.load_all(include_cont_streamflow=False, model_data=True)
# assuming each df has a single column
sj_arch = data["sj_archuleta"].iloc[:, 0]
animas  = data["animas_farmington"].iloc[:, 0]
sj_farm = data["sj_farmington"].iloc[:, 0]
sj_bluff = data["sj_bluff"].iloc[:,0]


# -------------- SJ @ Farmington ------------
# After looking at the data, it appears that summing the dam's release (SJ @ Archuleta) and
# contributions from the Animas (Animas @ Farmington) well-represents SJ @ Farmington. These
# plots quantify that simple model.


sj_arch = data["sj_archuleta"].iloc[:, 0]
animas  = data["animas_farmington"].iloc[:, 0]
sj_farm = data["sj_farmington"].iloc[:, 0]

# sum Animas + Archuleta, align on time
arch_plus_animas = sj_arch.add(animas, fill_value=0)

df_scatter = pd.concat(
    {"arch_plus_animas": arch_plus_animas,
     "sj_farmington": sj_farm},
    axis=1
).dropna()

x = df_scatter["arch_plus_animas"].values
y = df_scatter["sj_farmington"].values

# linear fit
m, b = np.polyfit(x, y, 1)
y_hat = m * x + b

# R^2
r = np.corrcoef(x, y)[0, 1]
r2 = r**2

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(x, y, s=5, alpha=0.5, label="Data")

# trendline (sorted so the line isn't jagged)
order = np.argsort(x)
ax.plot(x[order], y_hat[order], linewidth=2, label=f"Fit (R² = {r2:.2f})")

ax.set_xlabel("SJ Archuleta + Animas Farmington (cfs)")
ax.set_ylabel("SJ Farmington (cfs)")
ax.legend()
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.show()

