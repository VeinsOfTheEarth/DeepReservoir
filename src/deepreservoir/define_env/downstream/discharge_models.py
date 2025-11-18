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

def farmington_timeseries(data, start_date="1955-01-01"):  # sj_arch starts at 1955
    sj_arch = data["sj_archuleta"].iloc[:, 0]
    animas  = data["animas_farmington"].iloc[:, 0]
    sj_farm = data["sj_farmington"].iloc[:, 0]

    # combine, keep only days where all three exist
    df = pd.concat(
        [sj_arch, animas, sj_farm],
        axis=1,
        keys=["sj_arch", "animas", "sj_farm"],
    ).dropna()

    # apply start_date
    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        df = df[df.index >= start_ts]

    arch_plus_animas = df["sj_arch"] + df["animas"]

    fig, ax = plt.subplots(figsize=(18, 6))
    df["sj_farm"].plot(ax=ax, label="SJ Farmington", linewidth=1.5)
    df["sj_arch"].plot(ax=ax, label="SJ Archuleta", linewidth=1)
    df["animas"].plot(ax=ax, label="Animas Farmington", linewidth=1)
    arch_plus_animas.plot(ax=ax, label="SJ Archuleta + Animas Farmington", linewidth=1.5)

    ax.set_ylabel("Discharge (cfs)")
    ax.set_xlabel("Time")
    ax.legend()
    return ax

farmington_timeseries(data)   # or farmington_timeseries(data, start_date="1960-01-01")
plt.show()


def farmington_scatter(data):
    # grab series
    sj_arch = data["sj_archuleta"].iloc[:, 0]
    animas  = data["animas_farmington"].iloc[:, 0]
    sj_farm = data["sj_farmington"].iloc[:, 0]

    # combine and keep only rows where all 3 are present
    df = pd.concat(
        [sj_arch, animas, sj_farm],
        axis=1,
        keys=["arch", "animas", "farm"]
    ).dropna()

    # upstream sum and downstream
    x = (df["arch"] + df["animas"]).to_numpy()
    y = df["farm"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 6))

    # scatter
    ax.scatter(x, y, s=5, alpha=0.4, label="Daily data")

    # linear fit
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m * x_line + b
    ax.plot(x_line, y_line, linewidth=2, color='k', label="Linear fit")

    # R²
    r = np.corrcoef(x, y)[0, 1]
    r2 = r**2

    ax.set_xlabel("SJ Archuleta + Animas Farmington (cfs)")
    ax.set_ylabel("SJ Farmington (cfs)")
    ax.legend()

    ax.text(
        0.05, 0.95,
        f"y = {m:.3f}x + {b:.1f}\nR² = {r2:.2f}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    plt.tight_layout()
farmington_scatter(data)
plt.show()

# -------------- SJ @ Bluff ------------
# After looking at the data, it appears that simply lagging SJ @ Farmington by 1 day
# provides a very good model for SJ @ Bluff. Here we look at the effects of lagging.

def farmington_bluff_scatter(data):
    sj_farm = data["sj_farmington"].iloc[:, 0]
    sj_bluff = data["sj_bluff"].iloc[:, 0]

    lags = [0, 1, 2, 3]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    for ax, lag in zip(axes.flat, lags):
        # move SJ Farmington ahead by `lag` days
        farm_shift = sj_farm.shift(lag)

        # align and drop NaNs
        df = pd.concat(
            [farm_shift, sj_bluff],
            axis=1,
            keys=["farm", "bluff"],
        ).dropna()

        x = df["farm"].to_numpy()
        y = df["bluff"].to_numpy()

        ax.scatter(x, y, s=5, alpha=0.4)  # scatterpoints (default color)

        if x.size > 1:
            # linear fit
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = m * x_line + b
            ax.plot(x_line, y_line, color="black", linewidth=2)  # black trendline

            # R²
            y_hat = m * x + b
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot

            ax.text(
                0.05, 0.95,
                f"y = {m:.3f}x + {b:.1f}\nR² = {r2:.2f}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        ax.set_title(f"Lag = {lag} day{'s' if lag != 1 else ''}")

    # axis labels
    axes[1, 0].set_xlabel("SJ Farmington (cfs)")
    axes[1, 1].set_xlabel("SJ Farmington (cfs)")
    axes[0, 0].set_ylabel("SJ Bluff (cfs)")
    axes[1, 0].set_ylabel("SJ Bluff (cfs)")

    plt.tight_layout()
farmington_bluff_scatter(data)
plt.show()
