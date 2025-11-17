# -*- coding: utf-8 -*-
"""
Helper utilities for San Juan River gage analysis

Key groups:
- Loading:
    load_usgs_daily_dir
    load_usgs_continuous_dir_grid
- Plotting:
    plot_columns
    boxplot_annual_discharge
    plot_columns_with_event_mask
    plot_columns_with_classified_windows
- Step/peak event logic and evaluation:
    build_step_event_mask
    build_peak_event_mask
    evaluate_step_windows
    evaluate_peak_windows
- Correlation helpers:
    global_nan_mask
    lagcorr_series_stats_fast
    best_lag_from_r
    peak_lag_yearly_summary
    weighted_mean_lag_hours
"""

from __future__ import annotations
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# Optional SciPy for p-values / peak finding
try:
    from scipy import stats as _st
    from scipy.signal import find_peaks, peak_prominences
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

# ----------------------------- Renamers --------------------------------------

USGS_RENAMER = {
    "Animas_farmington": "Animas @ Farmington",
    "chinle_mexicanwater": "Chinle @ MW",
    "laplata_farmington": "La Plata @ Farmington",
    "mancos_towaoc": "Mancos @ Towaoc",
    "chaco_waterflow": "Chaco @ Waterflow",
    "sf_archuleta": "SJ @ Archuleta",
    "sf_bluff": "SJ @ Bluff",
    "sf_farmington": "SJ @ Farmington",
    "sf_fourcorners": "SJ @ Four Corners",
    "sf_shiprock": "SJ @ Shiprock",
}

USGS_CONT_RENAMER = {
    "animas_at_farmington": "Animas @ Farmington",
    "sj_at_archuleta": "SJ @ Archuleta",
    "sj_at_bluff": "SJ @ Bluff",
    "sj_at_farmington": "SJ @ Farmington",
    "sj_at_fourcorners": "SJ @ Four Corners",
    "sj_at_shiprock": "SJ @ Shiprock",
}

# ----------------------------- Plot helpers ----------------------------------

def _format_time_axis(ax, rotate=45, tick="auto"):
    if tick == "monthly":
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_formatter(mdates.NullFormatter())
    elif tick == "daily":
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_minor_formatter(mdates.NullFormatter())
    elif tick == "hourly":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        maj = mdates.AutoDateLocator(interval_multiples=True)
        ax.xaxis.set_major_locator(maj)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(maj))
    ax.tick_params(axis="x", which="major", rotation=rotate, length=6)
    ax.grid(True, which="major", alpha=0.35)

def _style_axis(ax):
    ax.grid(True, which="major", alpha=0.30, linewidth=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_alpha(0.5)
    ax.spines["bottom"].set_alpha(0.5)

# ----------------------------- Loaders ---------------------------------------

_TZ_OFFSETS = {"MDT": 6, "MST": 7, "PDT": 7, "PST": 8, "CDT": 5, "CST": 6, "EDT": 4, "EST": 5}

def load_usgs_daily_dir(
    dir_path=r"X:\Research\DeepReservoir\finalize_model\gages_usgs",
    pattern="daily_*.csv",
    *,
    time_col="time",
    value_col="value",
    renamer=None,
    join="outer",
    drop_dupes="last",
    reindex_daily=False,
    sort_ascending=True,
    coerce_numeric=True,
) -> pd.DataFrame:
    """Load a directory of daily CSVs (outer-join on date)."""
    files = sorted(Path(dir_path).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {dir_path}")
    series = []
    for fp in files:
        label = fp.stem
        if label.startswith("daily_"):
            label = label[len("daily_") :]
        if renamer:
            label = renamer.get(label, renamer.get(_try_int(label), label))
        try:
            df = pd.read_csv(fp, usecols=[time_col, value_col])
        except ValueError:
            warnings.warn(f"Skipping {fp.name}: expected columns '{time_col}' & '{value_col}'.")
            continue
        t = pd.to_datetime(df[time_col], errors="coerce")
        v = pd.to_numeric(df[value_col], errors="coerce") if coerce_numeric else df[value_col]
        s = pd.Series(v.values, index=t, name=str(label)).dropna()
        s = s[~s.index.duplicated(keep=drop_dupes)].sort_index(ascending=sort_ascending)
        if reindex_daily and len(s):
            s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        series.append(s)
    df_all = pd.concat(series, axis=1, join=join).sort_index(ascending=sort_ascending)
    return df_all

def _try_int(x):
    try: return int(x)
    except Exception: return None

def _read_usgs_continuous_file_irregular(
    path: Path,
    *,
    time_col_candidates=("datetime", "dateTime", "time"),
    tz_col_candidates=("tz_cd", "tz"),
    prefer_param="00060",
    drop_dupes="last",
    coerce_numeric=True,
) -> pd.Series:
    """USGS tab file (row-wise time zone). Return Series indexed by UTC."""
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    if df.empty:
        raise ValueError(f"{path} contained no rows after comments.")
    # drop the "5s 15s ..." header row if present
    if "agency_cd" in df.columns and df["agency_cd"].str.contains(r"\ds", na=False).any():
        df = df[~df["agency_cd"].str.fullmatch(r"\ds", na=False)]
    tcol = next((c for c in time_col_candidates if c in df.columns), None)
    if tcol is None:
        raise KeyError(f"{path}: datetime column not found.")
    zcol = next((c for c in tz_col_candidates if c in df.columns), None)
    patt = re.compile(rf".*_(?:{re.escape(prefer_param)})$")
    vcol = next((c for c in df.columns if patt.fullmatch(c)), None) \
        or next((c for c in df.columns if c.endswith("_00060")), None) \
        or next((c for c in df.columns if c.lower() == "value"), None)
    if vcol is None:
        raise KeyError(f"{path}: flow/value column not found (*_00060 or 'value').")

    t = pd.to_datetime(df[tcol], errors="coerce")
    if zcol in df.columns:
        tz = df[zcol].str.upper().map(_TZ_OFFSETS).fillna(0).astype(int)
        t = t + pd.to_timedelta(tz, unit="h")  # local -> UTC

    v = pd.to_numeric(df[vcol], errors="coerce") if coerce_numeric else df[vcol]
    s = pd.Series(v.values, index=t).dropna().sort_index()
    s = s[~s.index.duplicated(keep=drop_dupes)]
    return s

def _snap_to_grid_nearest(s: pd.Series, freq: str, max_delta=None, how="last") -> pd.Series:
    """Round timestamps to nearest grid line; drop points farther than tolerance."""
    if s.empty:
        return s
    rounded = s.index.round(freq)
    if max_delta is None:
        max_delta = pd.Timedelta(freq) / 2
    delta = (rounded - s.index).to_series().abs()
    keep = delta <= max_delta
    out = s[keep].copy()
    out.index = rounded[keep]
    if how == "mean":
        out = out.groupby(level=0).mean()
    else:  # 'last' or 'first'
        out = out[~out.index.duplicated(keep=how)]
    return out.sort_index()

def load_usgs_continuous_dir_grid(
    dir_path=r"X:\Research\DeepReservoir\finalize_model\gages_usgs\realtime",
    pattern="*.txt",
    *,
    renamer=USGS_CONT_RENAMER,
    freq="15min",
    method="resample",     # 'resample' | 'snap'
    agg="mean",            # for resample
    nearest_tolerance=None,
    fill=None,
    snap_how="last",       # for snap
    max_snap_fraction=0.5,
    join="outer",
) -> pd.DataFrame:
    """Read all files and align them onto a single common grid (same index for all)."""
    files = sorted(Path(dir_path).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {dir_path}")
    series = []
    for fp in files:
        name = renamer.get(fp.stem, fp.stem) if renamer else fp.stem
        try:
            s = _read_usgs_continuous_file_irregular(fp)  # UTC index
        except Exception as e:
            warnings.warn(f"Skipping {fp.name}: {e}")
            continue
        if s.empty:
            continue
        if method == "snap":
            tol = pd.Timedelta(freq) * max_snap_fraction
            s = _snap_to_grid_nearest(s, freq, max_delta=tol, how=snap_how)
        elif method == "resample":
            rs = s.resample(freq, origin="epoch", offset="0min")
            if agg == "nearest":
                tol = nearest_tolerance or (pd.Timedelta(freq) / 2)
                s = rs.nearest(tolerance=tol)
            else:
                s = getattr(rs, agg)()
            if fill:
                if isinstance(fill, tuple):
                    how, limit = fill
                    s = getattr(s, how)(limit=limit)
                else:
                    s = getattr(s, fill)()
        else:
            raise ValueError("method must be 'resample' or 'snap'")
        s.name = name
        series.append(s)
    start = min(s.index.min() for s in series)
    end = max(s.index.max() for s in series)
    grid = pd.date_range(start.floor(freq), end.ceil(freq), freq=freq, tz=start.tz)
    aligned = [s.reindex(grid) for s in series]
    return pd.concat(aligned, axis=1, join=join)

# ----------------------------- Core utilities --------------------------------

def global_nan_mask(df: pd.DataFrame) -> pd.Series:
    """True where any column is NaN (union mask)."""
    return df.isna().any(axis=1)

def best_lag_from_r(r_series: pd.Series):
    """Return lag (index) at |r| maximum; None if all NaN."""
    s = r_series.dropna()
    return s.abs().idxmax() if len(s) else None

def _periods_per_lag_unit(index, lag_unit="15min"):
    step = index.to_series().diff().dropna().median()
    if not pd.notna(step):
        step = pd.to_timedelta(lag_unit)
    return int(round(pd.to_timedelta(lag_unit) / step))

def lagcorr_series_stats_fast(
    x: pd.Series, y: pd.Series, mask_global: pd.Series,
    lags=range(-96, 97), lag_unit="15min", method="pearson",
):
    """Fast r vs lag using integer period shifts on aligned arrays; +k = y follows x."""
    idx = x.index
    y = y.reindex(idx)
    mg = mask_global.reindex(idx, fill_value=False)
    xv = pd.to_numeric(x, errors="coerce").to_numpy()
    yv = pd.to_numeric(y, errors="coerce").to_numpy()
    valid = (~mg.to_numpy()) & np.isfinite(xv) & np.isfinite(yv)
    p = _periods_per_lag_unit(idx, lag_unit)
    r_out, r2_out, p_out, n_out = {}, {}, {}, {}
    for k in lags:
        shift = k * p
        if shift >= 0:
            xa = xv[: len(xv) - shift] if shift else xv
            ya = yv[shift:]
            va = valid[: len(valid) - shift] & valid[shift:]
        else:
            sh = -shift
            xa = xv[sh:]
            ya = yv[: len(yv) - sh]
            va = valid[sh:] & valid[: len(valid) - sh]
        if va.sum() < 3:
            r_out[k] = np.nan; r2_out[k] = np.nan; p_out[k] = np.nan; n_out[k] = 0
            continue
        xa = xa[va]; ya = ya[va]
        if method == "spearman":
            xa = pd.Series(xa).rank().to_numpy()
            ya = pd.Series(ya).rank().to_numpy()
        r = float(np.corrcoef(xa, ya)[0, 1])
        n = int(len(xa))
        if _HAVE_SCIPY and method == "pearson":
            _, pval = _st.pearsonr(xa, ya)
        elif _HAVE_SCIPY and method == "spearman":
            _, pval = _st.spearmanr(xa, ya)
        else:
            pval = np.nan
        r_out[k] = r; r2_out[k] = r * r if np.isfinite(r) else np.nan; p_out[k] = pval; n_out[k] = n
    return { "r": pd.Series(r_out), "r2": pd.Series(r2_out), "p": pd.Series(p_out), "n": pd.Series(n_out) }

# ----------------------------- Plotting --------------------------------------

def plot_columns(
    df: pd.DataFrame, columns, *,
    start=None, end=None, labels=None,
    mode="overlay", suptitle=None,
    figsize_per_ax=(12, 2.6), overlay_figsize=(12, 5),
    sharey=False, rotate_xticks=45,
    linewidth=1.6, alpha=0.95,
    tick="auto",
):
    """Stacked or overlay plot of selected columns over a window."""
    cols = [c for c in columns if c in df.columns]
    if not cols:
        raise ValueError("None of the requested columns exist in df.")
    data = df[cols]
    if start is not None or end is not None:
        data = data.loc[pd.to_datetime(start) if start else None : pd.to_datetime(end) if end else None]
    if mode == "overlay":
        fig, ax = plt.subplots(figsize=overlay_figsize, constrained_layout=True)
        _style_axis(ax)
        for col in cols:
            lab = labels.get(col, col) if isinstance(labels, dict) else col
            ax.plot(data.index, data[col].values, linewidth=linewidth, alpha=alpha, label=lab)
        ax.set_xlabel("Date"); ax.set_ylabel("Value")
        _format_time_axis(ax, rotate=rotate_xticks, tick=tick)
        ax.legend(ncol=2, fontsize=9, frameon=False, loc="best")
        if suptitle: ax.set_title(suptitle)
        return fig, ax
    # stacked
    n = len(cols); fig_h = max(1.8, figsize_per_ax[1]) * n
    fig, axes = plt.subplots(n, 1, sharex=True, sharey=sharey,
                             figsize=(figsize_per_ax[0], fig_h), constrained_layout=True)
    if n == 1: axes = [axes]
    for i, (ax, col) in enumerate(zip(axes, cols)):
        _style_axis(ax)
        y = data[col]; lab = labels.get(col, col) if isinstance(labels, dict) else col
        ax.plot(y.index, y.values, linewidth=linewidth, alpha=alpha)
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", labelpad=10)
        if i == n - 1: ax.set_xlabel("Date")
        _format_time_axis(ax, rotate=rotate_xticks, tick=tick)
    if suptitle: fig.suptitle(suptitle, y=1.02, fontsize=12)
    return fig, axes

def boxplot_annual_discharge(
    df: pd.DataFrame, columns, *,
    start=None, end=None, agg="mean",
    labels=None, renamer=None,
    max_missing_frac=0.10,
    figsize=(12, 5), ylabel=None, title=None,
    xtick_rotation=0,
    show_points=True, point_alpha=0.6, point_size=20, jitter=0.07,
    color_scheme="default", add_color_legend=False, sj_group_label="San Juan River",
):
    """Boxplots of annual mean/max per series; screen years with too much missing data."""
    agg = agg.lower()
    if agg not in ("mean", "max"):
        raise ValueError("agg must be 'mean' or 'max'")
    data = df.loc[pd.to_datetime(start) if start else None : pd.to_datetime(end) if end else None]
    cols = [c for c in columns if c in data.columns]
    if not cols:
        raise ValueError("None of the specified columns exist in df.")
    def _label(col):
        if isinstance(labels, dict) and col in labels: return labels[col]
        try:
            cid = int(col); return renamer.get(cid, str(col)) if renamer else str(col)
        except Exception:
            return str(col)
    def _is_sj(name: str) -> bool:
        n = name.strip().lower(); return n.startswith("sj") or n.startswith("san juan")
    start_ts = data.index.min() if start is None else pd.to_datetime(start)
    end_ts   = data.index.max() if end   is None else pd.to_datetime(end)
    all_records, distributions, xlabels = [], [], []
    for col in cols:
        s = data[col]; lbl = _label(col); xlabels.append(lbl)
        vals = []
        for y in range(start_ts.year, end_ts.year + 1):
            y0 = max(pd.Timestamp(y, 1, 1), start_ts)
            y1 = min(pd.Timestamp(y, 12, 31), end_ts)
            if y0 > y1: continue
            chunk = s.loc[y0:y1]
            exp_days = (y1 - y0).days + 1
            obs_days = int(chunk.notna().sum())
            missing_frac = 1 - obs_days / exp_days if exp_days > 0 else 1.0
            included = (missing_frac <= max_missing_frac) and (obs_days > 0)
            if included:
                annual_val = float(chunk.mean()) if agg == "mean" else float(chunk.max())
                vals.append(annual_val)
            else:
                annual_val = np.nan
            all_records.append(dict(series=lbl, year=y, annual_value=annual_val,
                                    observed_days=obs_days, expected_days=exp_days,
                                    missing_frac=missing_frac, included=included, stat=agg))
        distributions.append(vals)
    annual_table = pd.DataFrame(all_records)
    base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not base_colors:
        base_colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B", "#B279A2"]
    if color_scheme == "group_sj":
        sj_color = base_colors[0]; other = iter(base_colors[1:] or base_colors)
        colors_assigned = [sj_color if _is_sj(lbl) else next(other) for lbl in xlabels]
    else:
        colors_assigned = [base_colors[i % len(base_colors)] for i in range(len(xlabels))]
    ylabel = ylabel or f"Annual {agg} discharge (cfs)"
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _style_axis(ax)
    bp = ax.boxplot(distributions, labels=xlabels, patch_artist=True, widths=0.6, whis=1.5,
                    medianprops=dict(linewidth=1.8, color="black"),
                    boxprops=dict(linewidth=1.2), whiskerprops=dict(linewidth=1.0),
                    capprops=dict(linewidth=1.0))
    for box, c in zip(bp["boxes"], colors_assigned):
        box.set_facecolor(c); box.set_alpha(0.55)
    if show_points:
        for i, (vals, c) in enumerate(zip(distributions, colors_assigned), start=1):
            if vals:
                xs = np.random.normal(loc=i, scale=jitter, size=len(vals))
                ax.scatter(xs, vals, s=point_size, alpha=point_alpha, color=c, zorder=3)
    ax.set_ylabel(ylabel); ax.set_title(title or "")
    ax.tick_params(axis="x", rotation=xtick_rotation)
    if add_color_legend and color_scheme == "group_sj":
        handles, labels_ = [], []
        sj_idxs = [i for i, lbl in enumerate(xlabels) if _is_sj(lbl)]
        if sj_idxs: handles.append(Patch(facecolor=colors_assigned[sj_idxs[0]], alpha=0.55)); labels_.append(sj_group_label)
        for lbl, face in zip(xlabels, colors_assigned):
            if not _is_sj(lbl):
                handles.append(Patch(facecolor=face, alpha=0.55)); labels_.append(lbl)
        if handles: ax.legend(handles, labels_, loc="best", frameon=False)
    return fig, ax, annual_table

# ----------------------------- Event detection -------------------------------

def _td_to_n(s_or_idx, td) -> int:
    """Convert '45min'/'2h'/Timedelta/int -> samples based on median step."""
    if isinstance(td, (int, np.integer)): return max(int(td), 1)
    if isinstance(td, str): td = td.strip().lower()
    td = pd.to_timedelta(td)
    idx = s_or_idx.index if isinstance(s_or_idx, pd.Series) else s_or_idx
    diffs = pd.Series(idx).diff().dropna()
    step = diffs.median() if len(diffs) else pd.to_timedelta("15min")
    return max(int(round(td / step)), 1)

def build_step_event_mask(
    s: pd.Series, *, method="diff", smooth="1h", window="2h",
    direction="up", thresh=None, p=99.0, min_run=1,
    pad_before="3h", pad_after="36h",
) -> pd.Series:
    """Boolean Series marking windows around sharp steps/ramps in `s`."""
    s = s.sort_index().astype(float)
    s_s = s.rolling(_td_to_n(s, smooth), center=True, min_periods=1).median() if smooth else s
    if method == "diff":
        x = s_s.diff()
    elif method == "edge":
        n_w = _td_to_n(s, window)
        pre = s_s.rolling(n_w, min_periods=n_w).mean()
        post = s_s.shift(-n_w).rolling(n_w, min_periods=n_w).mean()
        x = post - pre
    else:
        raise ValueError("method must be 'diff' or 'edge'")
    if thresh is None:
        if direction == "up":
            vals = x[x > 0].dropna().values; thresh = np.percentile(vals, p) if len(vals) else np.inf
        elif direction == "down":
            vals = (-x[x < 0]).dropna().values; thresh = np.percentile(vals, p) if len(vals) else np.inf
        else:
            vals = np.abs(x.dropna().values); thresh = np.percentile(vals, p) if len(vals) else np.inf
    det = (x > thresh) if direction == "up" else ((x < -thresh) if direction == "down" else (np.abs(x) > thresh))
    det = det.reindex(s.index, fill_value=False)
    n_run = _td_to_n(s, min_run)
    if n_run > 1:
        det = (det.rolling(n_run).sum() >= n_run).reindex(s.index, fill_value=False)
    mask = pd.Series(False, index=s.index)
    pre = _td_to_n(s, pad_before); post = _td_to_n(s, pad_after)
    starts = det & ~det.shift(1, fill_value=False)
    pos = np.flatnonzero(starts.to_numpy(dtype=bool))
    for i0 in pos:
        i_start = max(0, i0 - pre); i_end = min(len(mask) - 1, i0 + post)
        mask.iloc[i_start : i_end + 1] = True
    # also expand contiguous runs
    if det.any():
        groups = (det.ne(det.shift())).cumsum()
        for _, block in det[det].groupby(groups):
            p0 = s.index.get_indexer([block.index[0]])[0]
            p1 = s.index.get_indexer([block.index[-1]])[0]
            i_start = max(0, p0 - pre); i_end = min(len(mask) - 1, p1 + post)
            mask.iloc[i_start : i_end + 1] = True
    mask.name = "step_event_mask"
    return mask

def build_peak_event_mask(
    s: pd.Series, *, smooth="12h", min_separation="18h",
    prominence_q=0.80, pad_before="0h", pad_after="48h",
) -> pd.Series:
    """Boolean Series marking windows around local peaks in `s`."""
    s = s.sort_index().astype(float)
    s_s = s.rolling(_td_to_n(s, smooth), center=True, min_periods=1).median() if smooth else s
    dist = _td_to_n(s, min_separation)
    y = s_s.to_numpy()
    if _HAVE_SCIPY:
        peaks, _ = find_peaks(y, distance=max(dist, 1))
        if len(peaks):
            prom = peak_prominences(y, peaks)[0]
            thr = np.nanquantile(prom, float(prominence_q)) if np.isfinite(prom).any() else 0.0
            peaks = peaks[prom >= thr]
    else:
        candid = ((s_s.shift(1) < s_s) & (s_s.shift(-1) <= s_s)).fillna(False).to_numpy()
        p_idx = np.flatnonzero(candid)
        peaks = [p_idx[0]] if len(p_idx) else []
        for p in p_idx[1:]:
            if p - peaks[-1] >= max(dist, 1): peaks.append(p)
        peaks = np.array(peaks, dtype=int)
    mask = pd.Series(False, index=s.index)
    pre = _td_to_n(s, pad_before); post = _td_to_n(s, pad_after)
    for p in peaks:
        i0 = max(0, p - pre); i1 = min(len(mask) - 1, p + post)
        mask.iloc[i0 : i1 + 1] = True
    mask.name = "peak_event_mask"
    return mask

def _mask_to_spans(mask: pd.Series):
    m = mask.astype(bool)
    if m.empty or not m.any(): return []
    v = m.to_numpy(dtype=bool); idx = m.index
    dv = np.diff(v.astype(np.int8))
    starts = np.where(dv == 1)[0] + 1; ends = np.where(dv == -1)[0] + 1
    if v[0]: starts = np.r_[0, starts]
    if v[-1]: ends = np.r_[ends, len(v)]
    return [(idx[s], idx[e - 1]) for s, e in zip(starts, ends)]

# ----------------------------- Evaluation + plots ----------------------------

def evaluate_step_windows(
    df: pd.DataFrame, x_col: str, y_col: str, step_mask: pd.Series, *,
    mask_global=None, lags=range(-96, 97), lag_unit="15min", method="pearson",
    min_n=24, min_r=0.5, slope_sign="positive",
    amp_q=(0.05, 0.95), amp_ratio_bounds=(0.1, 8.0),
    plausible_lag=None, use_diff=False,
):
    """Score each step window; return (results_df, kept_mask, rejected_mask)."""
    mask_global = global_nan_mask(df[[x_col, y_col]]) if mask_global is None else mask_global.reindex(df.index, fill_value=False)
    x, y = df[x_col], df[y_col]
    spans = _mask_to_spans(step_mask.reindex(df.index, fill_value=False))
    rows, kept_mask, rejected_mask = [], pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    for i, (t0, t1) in enumerate(spans, start=1):
        wmask = pd.Series(False, index=df.index); wmask.loc[t0:t1] = True
        mg = mask_global | (~wmask)
        stats = lagcorr_series_stats_fast(x, y, mask_global=mg, lags=lags, lag_unit=lag_unit, method=method)
        best = best_lag_from_r(stats["r"])
        r_best = float(stats["r"].get(best, np.nan)); n_best = int(stats["n"].get(best, 0))
        # Pair series inside window at best lag
        xw = x.mask(mg).loc[t0:t1]
        yw = y.shift(int(best or 0), freq=lag_unit).mask(mg).loc[t0:t1]
        pair = pd.concat([xw, yw], axis=1).dropna(); n_pair = len(pair)

        # Simple slope & amplitude checks
        if n_pair >= 2 and np.isfinite(pair.iloc[:, 0].var()):
            cov = np.cov(pair.iloc[:, 0], pair.iloc[:, 1], ddof=1)[0, 1]
            varx = float(pair.iloc[:, 0].var(ddof=1)); slope = float(cov / varx) if varx > 0 else np.nan
        else:
            slope = np.nan
        if use_diff and n_pair >= 3:
            xd = pair.iloc[:, 0].diff().dropna(); yd = pair.iloc[:, 1].diff().dropna()
            m = min(len(xd), len(yd)); r_diff = float(np.corrcoef(xd.iloc[:m], yd.iloc[:m])[0, 1]) if m >= 3 else np.nan
        else:
            r_diff = np.nan
        def _qrange(s, q0, q1): return float(np.nanquantile(s, q1) - np.nanquantile(s, q0))
        amp_x = _qrange(pair.iloc[:, 0], *amp_q) if n_pair else np.nan
        amp_y = _qrange(pair.iloc[:, 1], *amp_q) if n_pair else np.nan
        amp_ratio = float(amp_y / amp_x) if (amp_x and np.isfinite(amp_x) and amp_x != 0) else np.nan

        step_td = _lagunit_to_step(lag_unit)
        lag_hours = float((int(best or 0) * step_td) / pd.Timedelta("1h"))

        ok = n_pair >= max(int(min_n), 2)
        if np.isfinite(r_best): ok &= r_best >= float(min_r)
        if slope_sign == "positive": ok &= (np.isfinite(slope) and slope > 0)
        elif slope_sign == "negative": ok &= (np.isfinite(slope) and slope < 0)
        if plausible_lag is not None and best is not None:
            ok &= int(plausible_lag[0]) <= int(best) <= int(plausible_lag[1])
        if np.isfinite(amp_ratio):
            lo, hi = amp_ratio_bounds; ok &= float(lo) <= amp_ratio <= float(hi)

        rows.append(dict(
            window_id=i, start=t0, end=t1,
            duration_h=float((t1 - t0) / pd.Timedelta("1h")),
            best_lag=int(best) if best is not None else None,
            lag_hours=lag_hours, r=r_best, n=n_best, n_pair=n_pair,
            slope=slope, r_diff=r_diff, amp_ratio=amp_ratio, keep=bool(ok),
        ))
        (kept_mask if ok else rejected_mask).loc[t0:t1] = True

    if not rows:
        cols = ["window_id","start","end","duration_h","best_lag","lag_hours","r","n","n_pair","slope","r_diff","amp_ratio","keep"]
        return pd.DataFrame(columns=cols), kept_mask, rejected_mask
    res = pd.DataFrame.from_records(rows).sort_values("start").reset_index(drop=True)
    return res, kept_mask, rejected_mask

def plot_columns_with_event_mask(
    df: pd.DataFrame, columns, event_mask: pd.Series, *,
    start=None, end=None, labels=None, mode="overlay",
    suptitle=None, figsize_per_ax=(12, 2.6), overlay_figsize=(12, 5),
    sharey=False, rotate_xticks=45, linewidth=1.6, alpha=0.95,
    tick="auto", mask_color="#f4d03f", mask_alpha=0.20,
    mask_edgecolor="none", mask_on_top=False, show_mask_legend=True,
    mask_label="Step event window", clip_series_to_mask=False,
):
    """Overlay/stack plot with semi-transparent shaded event windows."""
    tz = getattr(df.index, "tz", None)
    s_ts = pd.Timestamp(start, tz=tz) if start is not None else None
    e_ts = pd.Timestamp(end, tz=tz) if end is not None else None
    if s_ts is not None or e_ts is not None:
        df = df.loc[s_ts:e_ts]
    cols = [c for c in columns if c in df.columns]
    if not cols:
        raise ValueError("None of the requested columns exist in df.")
    m = event_mask.reindex(df.index, fill_value=False).astype(bool)
    spans = _mask_to_spans(m)

    if mode == "overlay":
        fig, ax = plt.subplots(figsize=overlay_figsize, constrained_layout=True)
        _style_axis(ax)
        z_shade = 4 if mask_on_top else 0
        for t0, t1 in spans:
            ax.axvspan(t0, t1, facecolor=mask_color, alpha=mask_alpha, edgecolor=mask_edgecolor, zorder=z_shade)
        for c in cols:
            s = df[c]; 
            if clip_series_to_mask: s = s.where(m)
            lab = labels.get(c, c) if isinstance(labels, dict) else c
            ax.plot(s.index, s.values, lw=linewidth, alpha=alpha, label=lab)
        ax.set_ylabel("Value"); ax.set_xlabel("Date")
        _format_time_axis(ax, rotate=rotate_xticks, tick=tick)
        if suptitle: ax.set_title(suptitle)
        if show_mask_legend:
            handles, labs = ax.get_legend_handles_labels()
            handles.append(Patch(facecolor=mask_color, alpha=mask_alpha, edgecolor="none")); labs.append(mask_label)
            ax.legend(handles, labs, frameon=False, ncol=2)
        else:
            ax.legend(frameon=False, ncol=2)
        if s_ts is not None or e_ts is not None: ax.set_xlim(left=s_ts, right=e_ts)
        return fig, ax

    # stacked
    n = len(cols); fig_h = max(2.0, figsize_per_ax[1] * n)
    fig, axes = plt.subplots(n, 1, figsize=(figsize_per_ax[0], fig_h),
                             sharex=True, sharey=sharey, constrained_layout=True)
    if n == 1: axes = [axes]
    for ax, c in zip(axes, cols):
        _style_axis(ax)
        for t0, t1 in spans:
            ax.axvspan(t0, t1, facecolor=mask_color, alpha=mask_alpha, edgecolor=mask_edgecolor, zorder=4 if mask_on_top else 0)
        s = df[c]; 
        if clip_series_to_mask: s = s.where(m)
        ax.plot(s.index, s.values, lw=linewidth, alpha=alpha, label=c)
        ax.set_ylabel(c, rotation=0, ha="right", va="center", labelpad=10)
    axes[-1].set_xlabel("Date"); _format_time_axis(axes[-1], rotate=rotate_xticks, tick=tick)
    if suptitle: fig.suptitle(suptitle, y=1.02)
    if show_mask_legend:
        axes[0].legend([Patch(facecolor=mask_color, alpha=mask_alpha, edgecolor="none")],
                       [mask_label], frameon=False, loc="best")
    if s_ts is not None or e_ts is not None: axes[-1].set_xlim(left=s_ts, right=e_ts)
    return fig, axes

def plot_columns_with_classified_windows(
    df: pd.DataFrame, columns, kept_mask: pd.Series, rejected_mask: pd.Series, **kwargs,
):
    """Overlay plot with kept (green) and rejected (red) windows."""
    fig, ax = plot_columns_with_event_mask(
        df, columns, kept_mask, mode="overlay",
        mask_color="#2ecc71", mask_alpha=0.18, mask_on_top=True,
        show_mask_legend=False, **kwargs
    )
    for t0, t1 in _mask_to_spans(rejected_mask.reindex(df.index, fill_value=False)):
        ax.axvspan(t0, t1, facecolor="#e74c3c", alpha=0.20, edgecolor="none", zorder=5)
    ax.legend(frameon=False, ncol=2)
    return fig, ax

# ----------------------------- Peak windows & yearly summary -----------------

def _lagunit_to_step(lag_unit):
    if isinstance(lag_unit, pd.Timedelta): return lag_unit
    if isinstance(lag_unit, str):
        u = lag_unit.strip().lower()
        if re.fullmatch(r"\d+\s*[a-z]+", u): return pd.to_timedelta(u)  # e.g., '15min'
        if re.fullmatch(r"[a-z]+", u): return pd.to_timedelta(1, unit=u)  # 'h','min'
    return pd.to_timedelta(lag_unit)

def _normalize_lags_for_orientation(lags, orient="x_to_y"):
    arr = np.array(list(lags), dtype=int)
    if orient == "x_to_y":
        if (arr < 0).any() and not (arr > 0).any():
            arr = np.abs(arr)  # keep semantics: +k = y follows x
    return list(arr)

def _normalize_bounds_for_orientation(bounds, orient="x_to_y"):
    if bounds is None: return None
    a, b = int(bounds[0]), int(bounds[1])
    if orient == "x_to_y" and a <= 0 and b <= 0:
        a, b = sorted((abs(a), abs(b)))
    return (a, b)

def evaluate_peak_windows(
    df: pd.DataFrame, x_col: str, y_col: str, *,
    mask_global=None, peak_mask=None,
    smooth="12h", min_separation="18h", prominence_q=0.80,
    pad_before="0h", pad_after="60h",
    lags=range(-96, 97), lag_unit="15min", method="pearson",
    min_n=16, min_r=0.6, slope_sign="positive",
    amp_ratio_bounds=(0.2, 6.0), plausible_lag=None, use_diff=False,
):
    """Find peak windows on x_col, then score with evaluate_step_windows."""
    mask_global = global_nan_mask(df[[x_col, y_col]]) if mask_global is None else mask_global
    peak_mask = build_peak_event_mask(
        df[x_col], smooth=smooth, min_separation=min_separation,
        prominence_q=prominence_q, pad_before=pad_before, pad_after=pad_after,
    ) if peak_mask is None else peak_mask
    res, kept, rejected = evaluate_step_windows(
        df, x_col, y_col, step_mask=peak_mask, mask_global=mask_global,
        lags=lags, lag_unit=lag_unit, method=method,
        min_n=min_n, min_r=min_r, slope_sign=slope_sign,
        amp_ratio_bounds=amp_ratio_bounds, plausible_lag=plausible_lag, use_diff=use_diff,
    )
    return res, kept, rejected, peak_mask

def plot_peakcorr_window(
    df: pd.DataFrame, x_col: str, y_col: str, *,
    year: int, tick="daily", mask_global=None,
    lag_unit="h", lags=range(-72, 73),
    smooth="6h", min_separation="18h", prominence_q=0.80,
    pad_before="0h", pad_after="60h",
    min_r=0.6, min_n=12, plausible_lag=None, title=None,
):
    """
    Upstream = x_col, downstream = y_col.
    Positive lag means y follows x by k*lag_unit.
    Any all-negative lag specifications are auto-flipped to positive.
    """
    lag_unit = (lag_unit or "h").lower()
    lags = _normalize_lags_for_orientation(lags, orient="x_to_y")
    plausible_lag = _normalize_bounds_for_orientation(plausible_lag, orient="x_to_y")
    s_ts = pd.Timestamp(year=year, month=1, day=1)
    e_ts = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)
    sub = df.loc[s_ts:e_ts]
    mask_global = global_nan_mask(sub[[x_col, y_col]]) if mask_global is None else mask_global
    res, kept, rej, _ = evaluate_peak_windows(
        sub, x_col, y_col, mask_global=mask_global,
        smooth=smooth, min_separation=min_separation, prominence_q=prominence_q,
        pad_before=pad_before, pad_after=pad_after,
        lags=lags, lag_unit=lag_unit, min_r=min_r, min_n=min_n,
        plausible_lag=plausible_lag, method="pearson",
    )
    fig, ax = plot_columns_with_classified_windows(
        sub, [x_col, y_col], kept_mask=kept, rejected_mask=rej,
        start=s_ts, end=e_ts, tick=tick,
        suptitle=title or f"{x_col} → {y_col} | {year} (peak windows; +lag=y follows x)",
    )
    if res.empty or not kept.any():
        return fig, ax, res, {"r": pd.Series(dtype=float), "p": pd.Series(dtype=float), "n": pd.Series(dtype=int)}, None
    stats_kept = lagcorr_series_stats_fast(
        sub[x_col], sub[y_col], mask_global=mask_global | (~kept),
        lags=lags, lag_unit=lag_unit, method="pearson",
    )
    best = best_lag_from_r(stats_kept["r"])
    return fig, ax, res, stats_kept, best

def peak_lag_yearly_summary(
    df: pd.DataFrame, x_col: str, y_col: str, *, years=None,
    lag_unit="h", lags=range(-72, 73),
    smooth="6h", min_separation="18h", prominence_q=0.80,
    pad_before="0h", pad_after="60h",
    min_r=0.6, min_n=12, plausible_lag=None,
) -> pd.DataFrame:
    """Compute best lag per year using peak windows; +lag=y follows x."""
    lag_unit = (lag_unit or "h").lower()
    lags = _normalize_lags_for_orientation(lags, orient="x_to_y")
    plausible_lag = _normalize_bounds_for_orientation(plausible_lag, orient="x_to_y")
    step_td = _lagunit_to_step(lag_unit)
    years = sorted(df.index.year.unique()) if years is None else list(years)
    rows = []
    for y in years:
        s_ts = pd.Timestamp(year=y, month=1, day=1)
        e_ts = pd.Timestamp(year=y, month=12, day=31, hour=23, minute=59, second=59)
        sub = df.loc[s_ts:e_ts]
        if sub.empty: 
            continue
        mg = global_nan_mask(sub[[x_col, y_col]])
        res, kept, _, _ = evaluate_peak_windows(
            sub, x_col, y_col, mask_global=mg,
            smooth=smooth, min_separation=min_separation, prominence_q=prominence_q,
            pad_before=pad_before, pad_after=pad_after,
            lags=lags, lag_unit=lag_unit, min_r=min_r, min_n=min_n,
            plausible_lag=plausible_lag, method="pearson",
        )
        if res.empty or not kept.any():
            rows.append(dict(year=y, best_lag=None, lag_hours=np.nan, r_at_best=np.nan,
                             n_at_best=0, n_pair_weight=0, kept_windows=0, rejected_windows=0))
            continue
        stats_kept = lagcorr_series_stats_fast(
            sub[x_col], sub[y_col], mask_global=mg | (~kept),
            lags=lags, lag_unit=lag_unit, method="pearson",
        )
        best = best_lag_from_r(stats_kept["r"])
        lag_hours = float((best * step_td) / pd.Timedelta("1h")) if best is not None else np.nan
        rows.append(dict(
            year=y,
            best_lag=(int(best) if best is not None else None),
            lag_hours=lag_hours,
            r_at_best=(float(stats_kept["r"].get(best)) if best is not None else np.nan),
            n_at_best=(int(stats_kept["n"].get(best)) if best is not None else 0),
            n_pair_weight=int(res.query("keep")["n_pair"].sum()),
            kept_windows=int(res["keep"].sum()),
            rejected_windows=int((~res["keep"]).sum()),
        ))
    return pd.DataFrame(rows).sort_values("year")

def weighted_mean_lag_hours(summary_df: pd.DataFrame) -> float:
    """Weighted mean of lag_hours using n_pair_weight as weights."""
    sub = summary_df.dropna(subset=["lag_hours", "n_pair_weight"])
    if sub.empty or (sub["n_pair_weight"] <= 0).all():
        return float("nan")
    return float(np.average(sub["lag_hours"], weights=sub["n_pair_weight"]))


