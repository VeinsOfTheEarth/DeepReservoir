from typing import Tuple, Sequence, Mapping

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

def save_plots(
    *,
    df_test: pd.DataFrame,
    outdir: Path | str,
    df_train_updates: pd.DataFrame | None = None,
    which: str | Sequence[str] | None = "all",
    plot_kwargs: Mapping[str, Mapping[str, object]] | None = None,
    dpi: int = 300,
) -> dict[str, Path]:
    """
    Generate and save one or more standard plots.

    Parameters
    ----------
    df_test
        Test-period rollout dataframe (from DRLModel.evaluate_test()).
    outdir
        Directory where PNGs will be written.
    df_train_updates
        Per-update (per-rollout) training reward dataframe (from
        m.train_update_metrics_ or m.load_train_update_metrics()). Only required
        for plots that summarize training updates (train_update_mean_rewards).
    which
        - "all" (default): all plots in PLOT_REGISTRY
        - name of a single plot (e.g. "storage_timeseries")
        - name of a group (e.g. "storage", "hydropower", "core")
        - list of plot and/or group names
    plot_kwargs
        Optional dict mapping plot-name -> kwargs dict passed to that plot
        function, e.g. {"storage_timeseries": {"figsize": (12, 4)}}.
    dpi
        DPI for saved PNGs.

    Returns
    -------
    dict
        Mapping {plot_name: saved_path}.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if plot_kwargs is None:
        plot_kwargs = {}

    selected = _resolve_plot_keys(which)
    saved: dict[str, Path] = {}

    for name in selected:
        spec = PLOT_REGISTRY[name]
        func = spec["func"]  # type: ignore[assignment]
        requires = spec["requires"]  # type: ignore[assignment]
        filename = spec["filename"]  # type: ignore[assignment]

        # Build args
        args: list[object] = []
        if "df_test" in requires:
            args.append(df_test)
        if "df_train_updates" in requires:
            if df_train_updates is None:
                import warnings
                warnings.warn(
                    f"Plot {name!r} requires df_train_updates but none was provided; skipping."
                )
                continue
            args.append(df_train_updates)

        kw = dict(plot_kwargs.get(name, {}))
        res = func(*args, **kw)  # type: ignore[misc]
        # All plot_* functions return (fig, ax[, ...])
        fig = res[0]

        path = outdir / filename
        save(fig, path, dpi=dpi)
        saved[name] = path

    return saved


# Global font (DejaVu is bundled with Matplotlib, so it’s safe/portable)
FONT_FAMILY = "DejaVu Sans"
mpl.rcParams["font.family"] = FONT_FAMILY

OBJECTIVE_COLOR_MAP = {
    "dam_safety":   "#1f77b4",  # blue
    "esa_min_flow": "#ff7f0e",  # orange
    "flooding":     "#2ca02c",  # green
    "niip":         "#d62728",  # red
    "physics":      "#9467bd",  # purple 
}

FALLBACK_COLORS = [
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # olive
    "#17becf",  # teal
]

# Colors for agent vs historic (tweak as you like)
AGENT_COLOR = "#08519c"       # dark blue
AGENT_FILL_COLOR = "#9ecae1"  # light blue

HIST_COLOR = "#e6550d"        # dark orange
HIST_FILL_COLOR = "#fdae6b"   # light orange

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


def plot_train_update_mean_rewards(
    df_train_updates: pd.DataFrame,
    *,
    x_axis: str = "timesteps",
    figsize: tuple[float, float] = (8, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot per-update mean reward for each objective.

    This uses the per-rollout summaries produced during PPO training, where
    each row corresponds to one rollout / policy update iteration.

    Expected columns
    ----------------
    - 'update_idx' (int)
    - 'timesteps' (int, optional): cumulative env steps at end of the rollout
    - 'mean_total_reward' (float, optional)
    - columns starting with 'mean_' for each objective component, e.g.
      'mean_dam_safety.storage_band'

    """
    # Choose x-axis
    if x_axis in df_train_updates.columns:
        x = df_train_updates[x_axis].values
        xlabel = "Timesteps" if x_axis == "timesteps" else x_axis
    elif "update_idx" in df_train_updates.columns:
        x = df_train_updates["update_idx"].values
        xlabel = "Update"
    else:
        x = df_train_updates.index.values
        xlabel = "Update"

    mean_cols = [
        c for c in df_train_updates.columns
        if c.startswith("mean_") and c != "mean_total_reward"
    ]
    if not mean_cols and "mean_total_reward" not in df_train_updates.columns:
        raise ValueError("No 'mean_*' columns found in df_train_updates.")

    fig, ax = plt.subplots(figsize=figsize)

    # Nicer style
    ax.grid(True, which="major", linestyle=":", alpha=0.6)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)

    ax.tick_params(labelsize=11)
    ax.set_xlabel(xlabel, fontsize=13)
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
        ax.plot(
            x,
            df_train_updates[col].values,
            linewidth=1.6,
            color=color,
            label=full_key,
        )

    # Total reward
    if "mean_total_reward" in df_train_updates.columns:
        total_mean = df_train_updates["mean_total_reward"].values
    elif mean_cols:
        total_mean = df_train_updates[mean_cols].sum(axis=1).values
    else:
        total_mean = df_train_updates.index.to_numpy(dtype=float) * 0.0

    ax.plot(
        x,
        total_mean,
        linewidth=2.0,
        color="black",
        label="total",
    )

    ax.set_title("Per-update mean reward by objective", fontsize=14)

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



def plot_storage_doy(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot day-of-year storage (agent vs historic) using only complete years.

    Uses:
      - 'storage_hist_af'
      - 'storage_agent_af'
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for DOY plot.")

    if "storage_hist_af" not in df.columns or "storage_agent_af" not in df.columns:
        raise ValueError("df must contain 'storage_hist_af' and 'storage_agent_af'.")

    # Only complete years for each series
    hist_full = _select_full_years(df["storage_hist_af"].dropna())
    agent_full = _select_full_years(df["storage_agent_af"].dropna())

    hist_stats = _doy_stats(hist_full)
    agent_stats = _doy_stats(agent_full)

    doy = hist_stats.index.values  # 1..365

    fig, ax = plt.subplots(figsize=figsize)

    # Clean style
    ax.grid(True, which="major", linestyle=":", alpha=0.6)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)

    ax.tick_params(labelsize=11)
    ax.set_xlabel("Day of year", fontsize=13)
    ax.set_ylabel("Storage [AF]", fontsize=13)

    # --- Historic storage: orange ---
    ax.fill_between(
        doy,
        hist_stats["min"],
        hist_stats["max"],
        color=HIST_FILL_COLOR,
        alpha=0.10,
        label="_nolegend_",
    )
    ax.fill_between(
        doy,
        hist_stats["q25"],
        hist_stats["q75"],
        color=HIST_FILL_COLOR,
        alpha=0.25,
        label="Historic IQR",
    )
    ax.plot(
        doy,
        hist_stats["median"],
        color=HIST_COLOR,
        linewidth=1.8,
        label="Historic median",
    )

    # --- Agent storage: blue ---
    ax.fill_between(
        doy,
        agent_stats["min"],
        agent_stats["max"],
        color=AGENT_FILL_COLOR,
        alpha=0.10,
        label="_nolegend_",
    )
    ax.fill_between(
        doy,
        agent_stats["q25"],
        agent_stats["q75"],
        color=AGENT_FILL_COLOR,
        alpha=0.20,
        label="Agent IQR",
    )
    ax.plot(
        doy,
        agent_stats["median"],
        color=AGENT_COLOR,
        linewidth=1.8,
        label="Agent median",
    )

    ax.set_title("Day-of-year storage (complete years, test period)", fontsize=14)

    leg = ax.legend(loc="best", frameon=True)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor("white")

    fig.tight_layout()
    return fig, ax


def _iter_full_year_traces(series: pd.Series):
    """
    Yield (year, x_doy, values) for each complete calendar year in series.
    Feb 29 is dropped.
    """
    full = _select_full_years(series.dropna())
    if full.empty:
        return

    for y in sorted(np.unique(full.index.year)):
        s_y = full[full.index.year == y]
        if s_y.empty:
            continue
        # Drop Feb 29
        mask = ~((s_y.index.month == 2) & (s_y.index.day == 29))
        s_y = s_y[mask]
        x = s_y.index.dayofyear
        yield y, x, s_y.values


def plot_storage_doy_traces(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot all storage trajectories (per year) for agent vs historic
    on the same DOY axes.

    Uses:
      - 'storage_hist_af'
      - 'storage_agent_af'
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for DOY plot.")

    if "storage_hist_af" not in df.columns or "storage_agent_af" not in df.columns:
        raise ValueError("df must contain 'storage_hist_af' and 'storage_agent_af'.")

    fig, ax = plt.subplots(figsize=figsize)

    # Clean style
    ax.grid(True, which="major", linestyle=":", alpha=0.6)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)

    ax.tick_params(labelsize=11)
    ax.set_xlabel("Day of year", fontsize=13)
    ax.set_ylabel("Storage [AF]", fontsize=13)

    # Historic trajectories
    first_hist = True
    for y, x_doy, vals in _iter_full_year_traces(df["storage_hist_af"]):
        label = "Historic (per year)" if first_hist else "_nolegend_"
        ax.plot(
            x_doy,
            vals,
            color=HIST_COLOR,
            alpha=0.35,
            linewidth=1.0,
            label=label,
        )
        first_hist = False

    # Agent trajectories
    first_agent = True
    for y, x_doy, vals in _iter_full_year_traces(df["storage_agent_af"]):
        label = "Agent (per year)" if first_agent else "_nolegend_"
        ax.plot(
            x_doy,
            vals,
            color=AGENT_COLOR,
            alpha=0.35,
            linewidth=1.0,
            label=label,
        )
        first_agent = False

    ax.set_title("Day-of-year storage trajectories (complete years, test period)", fontsize=14)

    leg = ax.legend(loc="best", frameon=True)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor("white")

    fig.tight_layout()
    return fig, ax


def _select_full_years(series: pd.Series) -> pd.Series:
    """
    Keep only years where the series has a complete calendar year
    (Jan 1–Dec 31 with no missing days).
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex.")

    idx = series.index
    years = np.unique(idx.year)
    mask_keep = np.zeros(len(series), dtype=bool)

    for y in years:
        mask_y = (idx.year == y)
        s_y = series[mask_y]
        if s_y.empty:
            continue
        first = s_y.index.min().normalize()
        last = s_y.index.max().normalize()
        expected_days = (last - first).days + 1
        # require full calendar year with no gaps
        if (
            first == pd.Timestamp(y, 1, 1)
            and last == pd.Timestamp(y, 12, 31)
            and len(s_y) == expected_days
        ):
            mask_keep |= mask_y

    return series[mask_keep]


def _doy_stats(series: pd.Series) -> pd.DataFrame:
    """
    Compute day-of-year stats (min, 25%, median, 75%, max) for a series.

    Drops Feb 29 so all years have 365 days.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex.")

    # Drop Feb 29 (leap day)
    mask = ~((series.index.month == 2) & (series.index.day == 29))
    s = series[mask]

    grouped = s.groupby(s.index.dayofyear)
    stats = pd.DataFrame({
        "min": grouped.min(),
        "q25": grouped.quantile(0.25),
        "median": grouped.median(),
        "q75": grouped.quantile(0.75),
        "max": grouped.max(),
    })
    return stats  # index = day-of-year (1..365)


def plot_hydropower_doy(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot day-of-year hydropower (agent vs historic) using precomputed columns:
      - 'hydro_hist_mwh'
      - 'hydro_agent_mwh'
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for DOY plot.")

    if "hydro_hist_mwh" not in df.columns or "hydro_agent_mwh" not in df.columns:
        raise ValueError("df must contain 'hydro_hist_mwh' and 'hydro_agent_mwh'.")

    hist_stats = _doy_stats(df["hydro_hist_mwh"].dropna())
    agent_stats = _doy_stats(df["hydro_agent_mwh"].dropna())

    doy = hist_stats.index.values  # 1..365

    fig, ax = plt.subplots(figsize=figsize)

    # Clean style
    ax.grid(True, which="major", linestyle=":", alpha=0.6)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)

    ax.tick_params(labelsize=11)
    ax.set_xlabel("Day of year", fontsize=13)
    ax.set_ylabel("Hydropower [MWh/day]", fontsize=13)

    # Historic hydropower bands
    ax.fill_between(
        doy,
        hist_stats["min"],
        hist_stats["max"],
        color="#dadaeb",
        alpha=0.10,
        label="_nolegend_",
    )
    ax.fill_between(
        doy,
        hist_stats["q25"],
        hist_stats["q75"],
        color="#bcbddc",
        alpha=0.25,
        label="Historic IQR",
    )
    ax.plot(
        doy,
        hist_stats["median"],
        color="#756bb1",
        linewidth=1.8,
        label="Historic median",
    )

    # Agent hydropower bands
    ax.fill_between(
        doy,
        agent_stats["min"],
        agent_stats["max"],
        color="#9e9ac8",
        alpha=0.10,
        label="_nolegend_",
    )
    ax.fill_between(
        doy,
        agent_stats["q25"],
        agent_stats["q75"],
        color="#807dba",
        alpha=0.20,
        label="Agent IQR",
    )
    ax.plot(
        doy,
        agent_stats["median"],
        color="#54278f",
        linewidth=1.8,
        label="Agent median",
    )

    ax.set_title("Day-of-year hydropower (test period)", fontsize=14)

    leg = ax.legend(loc="best", frameon=True)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor("white")

    fig.tight_layout()
    return fig, ax


def plot_hydropower_doy_traces(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot all hydropower trajectories (per year) for agent vs historic
    on the same DOY axes.

    Uses:
      - 'hydro_hist_mwh'
      - 'hydro_agent_mwh'
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for DOY plot.")

    if "hydro_hist_mwh" not in df.columns or "hydro_agent_mwh" not in df.columns:
        raise ValueError("df must contain 'hydro_hist_mwh' and 'hydro_agent_mwh'.")

    fig, ax = plt.subplots(figsize=figsize)

    # Clean style
    ax.grid(True, which="major", linestyle=":", alpha=0.6)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)

    ax.tick_params(labelsize=11)
    ax.set_xlabel("Day of year", fontsize=13)
    ax.set_ylabel("Hydropower [MWh/day]", fontsize=13)

    # Historic trajectories (orange)
    first_hist = True
    for y, x_doy, vals in _iter_full_year_traces(df["hydro_hist_mwh"]):
        label = "Historic (per year)" if first_hist else "_nolegend_"
        ax.plot(
            x_doy,
            vals,
            color=HIST_COLOR,
            alpha=0.35,
            linewidth=1.0,
            label=label,
        )
        first_hist = False

    # Agent trajectories (blue)
    first_agent = True
    for y, x_doy, vals in _iter_full_year_traces(df["hydro_agent_mwh"]):
        label = "Agent (per year)" if first_agent else "_nolegend_"
        ax.plot(
            x_doy,
            vals,
            color=AGENT_COLOR,
            alpha=0.35,
            linewidth=1.0,
            label=label,
        )
        first_agent = False

    ax.set_title("Day-of-year hydropower trajectories (complete years, test period)", fontsize=14)

    leg = ax.legend(loc="best", frameon=True)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor("white")

    fig.tight_layout()
    return fig, ax


def plot_spr_farmington_10k_timeseries(
    df: pd.DataFrame,
    *,
    threshold_cfs: float = 10_000.0,
    figsize: tuple[float, float] = (10, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """
    SPR plot:
      x-axis: date
      y-axis: Farmington discharge = (agent San Juan release) + (observed Animas gauge)
      - horizontal line at threshold (default 10,000 cfs)
      - dots on dates where discharge >= threshold

    Expected df columns:
      - 'sanjuan_release_cfs'        (agent release component)
      - 'animas_farmington_q_cfs'    (gauge data)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for time series plots.")

    required = ["sanjuan_release_cfs", "animas_farmington_q_cfs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    x = df.index
    sanjuan = df["sanjuan_release_cfs"].astype(float)
    animas = df["animas_farmington_q_cfs"].astype(float)

    farmington = sanjuan + animas
    exceed = farmington >= float(threshold_cfs)

    fig, ax = plt.subplots(figsize=figsize)

    # Clean style (match your plotting style)
    ax.grid(True, which="major", linestyle=":", alpha=0.6)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)

    ax.tick_params(labelsize=11)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Discharge at Farmington [cfs]", fontsize=13)

    # Main line
    ax.plot(
        x,
        farmington.values,
        linewidth=1.8,
        label="Farmington = San Juan(agent) + Animas(gauge)",
    )

    # Threshold line
    ax.axhline(
        float(threshold_cfs),
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold = {float(threshold_cfs):,.0f} cfs",
    )

    # Dots where exceed
    if exceed.any():
        ax.scatter(
            x[exceed],
            farmington.loc[exceed].values,
            s=22,
            zorder=5,
            label=f"Exceedances (n={int(exceed.sum())})",
        )

    ax.set_title("SPR: Farmington discharge and 10,000 cfs threshold", fontsize=14)

    leg = ax.legend(loc="best", frameon=True)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor("white")

    fig.tight_layout()
    return fig, ax






# ---------------------------------------------------------------------------
# Plot registry + groups + convenience driver
# ---------------------------------------------------------------------------

# Each entry: name -> { "func": callable, "requires": ("df_test" / "df_train_updates"), "filename": str }
PLOT_REGISTRY: dict[str, Mapping[str, object]] = {
    "storage_timeseries": {
        "func": plot_storage_timeseries,
        "requires": ("df_test",),
        "filename": "storage_timeseries.png",
    },
    "train_update_mean_rewards": {
        "func": plot_train_update_mean_rewards,
        "requires": ("df_train_updates",),
        "filename": "train_update_mean_rewards.png",
    },
    "release_timeseries": {
        "func": plot_release_timeseries,
        "requires": ("df_test",),
        "filename": "release_timeseries.png",
    },
    "storage_doy": {
        "func": plot_storage_doy,
        "requires": ("df_test",),
        "filename": "storage_doy.png",
    },
    "storage_doy_traces": {
        "func": plot_storage_doy_traces,
        "requires": ("df_test",),
        "filename": "storage_doy_traces.png",
    },
    "hydropower_doy": {
        "func": plot_hydropower_doy,
        "requires": ("df_test",),
        "filename": "hydropower_doy.png",
    },
    # Only include this if you added plot_hydropower_doy_traces above
    "hydropower_doy_traces": {
        "func": plot_hydropower_doy_traces,  # type: ignore[name-defined]
        "requires": ("df_test",),
        "filename": "hydropower_doy_traces.png",
    },

    "spr_farmington_10k_timeseries": {
        "func": plot_spr_farmington_10k_timeseries,
        "requires": ("df_test",),
        "filename": "spr_farmington_10k_timeseries.png",
    },   
}


# Optional groups, for convenience
PLOT_GROUPS: dict[str, tuple[str, ...]] = {
    "core": (
        "storage_timeseries",
        "train_update_mean_rewards",
        "release_timeseries",
    ),
    "storage": (
        "storage_timeseries",
        "storage_doy",
        "storage_doy_traces",
    ),
    "hydropower": (
        "hydropower_doy",
        "hydropower_doy_traces",
    ),
    "rewards": (
        "train_update_mean_rewards",
    ),
    "timeseries": (
        "storage_timeseries",
        "release_timeseries",
    ),
    "doy": (
        "storage_doy",
        "storage_doy_traces",
        "hydropower_doy",
        "hydropower_doy_traces",
    ),
    "spr": (
        "spr_farmington_10k_timeseries",
    ),   
    "timeseries": (
        "storage_timeseries",
        "release_timeseries",
        "spr_farmington_10k_timeseries",
    ),
}

def _resolve_plot_keys(which: str | Sequence[str] | None) -> list[str]:
    """
    Expand 'which' into a concrete list of plot keys.

    - 'all' or None -> all registry keys
    - group names (PLOT_GROUPS) expand to their members
    - otherwise treated as individual plot names
    - comma-separated strings are allowed
    """
    if which is None or which == "all":
        return list(PLOT_REGISTRY.keys())

    keys: list[str] = []

    def add_name(name: str) -> None:
        if name in PLOT_GROUPS:
            for k in PLOT_GROUPS[name]:
                if k not in keys:
                    keys.append(k)
        elif name in PLOT_REGISTRY:
            if name not in keys:
                keys.append(name)
        else:
            raise ValueError(f"Unknown plot or group name: {name!r}")

    if isinstance(which, str):
        parts = [p.strip() for p in which.split(",") if p.strip()]
        for p in parts:
            add_name(p)
    else:
        for item in which:
            add_name(str(item))

    return keys
