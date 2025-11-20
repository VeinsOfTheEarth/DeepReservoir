from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Global font (DejaVu is bundled with Matplotlib, so it’s safe/portable)
FONT_FAMILY = "DejaVu Sans"
mpl.rcParams["font.family"] = FONT_FAMILY

OBJECTIVE_COLOR_MAP = {
    "dam_safety":   "#1f77b4",  # blue
    "esa_min_flow": "#ff7f0e",  # orange
    "flooding":     "#2ca02c",  # green
    "niip":         "#d62728",  # red
    "physics":      "#9467bd",  # purple (dedicated color)
}

FALLBACK_COLORS = [
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # olive
    "#17becf",  # teal
]

def save(
    fig: plt.Figure,
    path: Path | str,
    dpi: int = 300,
    close: bool = True,
) -> None:
    """
    Generic figure saver.

    Examples
    --------
    fig, ax = plot_storage_timeseries(df)
    save(fig, "runs/debug/storage_timeseries.png")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)


def plot_storage_timeseries(
    df: pd.DataFrame,
    *,
    storage_agent_col: str = "storage_agent_af",
    storage_hist_col: str = "storage_hist_af",
    elev_col: str = "elev_ft",
    figsize: Tuple[float, float] = (10, 4),
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Plot agent vs historic storage with elevation shaded in the background.

    - Historic storage: lighter blue
    - Agent storage: strong/bold blue
    - Dead pool & max storage: light grey dashed lines (one legend entry)
    - Elevation: lightly shaded yellow band with a golden line on a right y-axis
    """
    # Hard-coded storage band for Navajo (AF)
    S_MIN = 500_000.0    # dead pool / safe min [AF]
    S_MAX = 1_731_750.0  # max storage [AF]

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for time series plots.")

    x = df.index

    fig, ax = plt.subplots(figsize=figsize)

    # Make it look nicer than default matplotlib
    ax.grid(True, which="major", linestyle=":", alpha=0.6)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)

    # Slightly larger fonts
    ax.tick_params(labelsize=11)
    ax.set_ylabel("Storage [AF]", fontsize=13)

    # Left axis: storage (historic vs agent)
    if storage_hist_col in df.columns:
        ax.plot(
            x,
            df[storage_hist_col],
            label="Historic storage",
            linewidth=1.5,
            color="#9ecae1",  # light blue
        )
    if storage_agent_col in df.columns:
        ax.plot(
            x,
            df[storage_agent_col],
            label="Agent storage",
            linewidth=1.8,
            color="#08519c",  # strong blue
        )

    # Dead pool / max storage lines (one legend entry)
    ax.axhline(
        S_MIN,
        linestyle="--",
        linewidth=1.0,
        color="#b0b0b0",
        label="Deadpool / Spill",
    )
    ax.axhline(
        S_MAX,
        linestyle="--",
        linewidth=1.0,
        color="#b0b0b0",
        label="_nolegend_",  # don't duplicate in legend
    )

    # Right axis: elevation as shaded yellow background + golden line
    ax2 = ax.twinx()
    ax2.tick_params(labelsize=11)

    if elev_col in df.columns:
        elev = df[elev_col].dropna()
        if not elev.empty:
            e_min = elev.min()
            e_max = elev.max()
            margin = max(1.0, 0.02 * (e_max - e_min))
            ax2.set_ylim(e_min - margin, e_max + margin)

            # Shade under elevation
            ax2.fill_between(
                x,
                ax2.get_ylim()[0],
                elev.reindex(x),
                color="#fff7bc",  # light yellow
                alpha=0.4,
                zorder=0,
            )

            # # Elevation line (more golden)
            # ax2.plot(
            #     x,
            #     elev,
            #     linewidth=1.3,
            #     color="#DAA520",  # goldenrod
            #     alpha=0.9,
            #     label="Elevation",
            #     zorder=1,
            # )

    ax2.set_ylabel(
        "Reservoir elevation [ft]",
        fontsize=13,
        color="#DAA520",
        rotation=270,          # rotate text 180°
    )
    # tweak position so it doesn't overlap the axis
    ax2.yaxis.set_label_coords(1.09, 0.5)  # (x, y) in axes fraction coords
    ax2.tick_params(axis="y", colors="#DAA520")

    # Title (hard-coded)
    ax.set_title("Navajo Reservoir storage and elevation (test period)", fontsize=14)

    # Combined legend with opaque box
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines2:
        lines = lines1 + lines2
        labels = labels1 + labels2
    else:
        lines, labels = lines1, labels1

    if lines:
        leg = ax.legend(lines, labels, loc="upper left", frameon=True)
        leg.get_frame().set_alpha(0.9)      # slightly transparent, but opaque enough
        leg.get_frame().set_facecolor("white")

    fig.tight_layout()
    return fig, ax, ax2


def plot_episode_mean_rewards(
    df_ep: pd.DataFrame,
    figsize: tuple[float, float] = (8, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot per-episode mean reward for each objective.

    Expects a DataFrame like DRLModel.episode_reward_components_, with:
      - 'episode_idx' column (or index)
      - columns starting with 'mean_' for each objective.
    """
    # X-axis: episode index
    if "episode_idx" in df_ep.columns:
        x = df_ep["episode_idx"].values
    else:
        x = df_ep.index.values

    mean_cols = [c for c in df_ep.columns if c.startswith("mean_")]
    if not mean_cols:
        raise ValueError("No columns starting with 'mean_' found in df_ep.")

    fig, ax = plt.subplots(figsize=figsize)

    # Nicer style
    ax.grid(True, which="major", linestyle=":", alpha=0.6)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)

    ax.tick_params(labelsize=11)
    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel("Mean reward per step", fontsize=13)

    def get_color_for_objective(obj_key: str, idx: int) -> str:
        """
        obj_key is like 'dam_safety.storage_band'.
        We use the part before the first dot as the 'type'.
        """
        base = obj_key.split(".", 1)[0]
        if base in OBJECTIVE_COLOR_MAP:
            return OBJECTIVE_COLOR_MAP[base]
        return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]

    # Plot individual objectives
    for i, col in enumerate(mean_cols):
        full_key = col[len("mean_") :]   # e.g. 'dam_safety.storage_band'
        color = get_color_for_objective(full_key, i)
        label = full_key

        ax.plot(
            x,
            df_ep[col].values,
            linewidth=1.6,
            color=color,
            label=label,
        )

    # Total reward = sum of component means (per episode)
    total_mean = df_ep[mean_cols].sum(axis=1)
    ax.plot(
        x,
        total_mean.values,
        linewidth=2.0,
        color="black",
        label="total",
    )

    ax.set_title("Per-episode mean reward by objective", fontsize=14)

    leg = ax.legend(loc="best", frameon=True)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor("white")

    fig.tight_layout()
    return fig, ax


def plot_release_timeseries(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot agent vs historic release, along with inflow and evaporation (all in cfs).

    Expects df columns:
      - 'release_agent_cfs' (agent)
      - 'release_cfs'       (historic)
      - 'inflow_cfs'
      - 'evap_cfs'
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for time series plots.")

    x = df.index

    fig, ax = plt.subplots(figsize=figsize)

    # Clean style
    ax.grid(True, which="major", linestyle=":", alpha=0.6)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)

    ax.tick_params(labelsize=11)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Flow [cfs]", fontsize=13)

    # Historic vs agent release
    if "release_cfs" in df.columns:
        ax.plot(
            x,
            df["release_cfs"],
            label="Historic release",
            linewidth=1.5,
            color="#9ecae1",  # light blue
        )

    if "release_agent_cfs" in df.columns:
        ax.plot(
            x,
            df["release_agent_cfs"],
            label="Agent release",
            linewidth=1.8,
            color="#08519c",  # strong blue
        )

    # Inflow and evaporation (same axis, all cfs)
    if "inflow_cfs" in df.columns:
        ax.plot(
            x,
            df["inflow_cfs"],
            label="Inflow",
            linewidth=1.4,
            color="#2ca02c",  # green
            alpha=0.9,
        )

    if "evap_cfs" in df.columns:
        ax.plot(
            x,
            df["evap_cfs"],
            label="Evaporation (eq. cfs)",
            linewidth=1.2,
            color="#ff7f0e",  # orange
            alpha=0.9,
        )

    ax.set_title("Navajo release, inflow, and evaporation (test period)", fontsize=14)

    leg = ax.legend(loc="best", frameon=True)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor("white")

    fig.tight_layout()
    return fig, ax
