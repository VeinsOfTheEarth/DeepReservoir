import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -------------------- CONFIG (edit as needed)
WINTER_PEAK_START = (11, 1)   # Nov 1
WINTER_PEAK_END   = (5, 31)   # May 31
SPRING_START      = (4, 1)    # Apr 1
SPRING_END        = (7, 31)   # Jul 31
DAILY_MIN_FRAC    = 0.80      # require ≥80% of 15-min samples present to accept a daily mean
FREQ_15MIN_PER_DAY = 24*60//15  # 96
CFS_DAY_TO_ACFT = 1.983471  # 1 cfs sustained for 1 day -> acre-feet
MIN_SPRING_DAY_FRAC = 0.60     # ≥60% of Apr–Jul days must be present

ANIMAS_SERIES_NAME = 'Animas @ Farmington'  # matches your helpers output

# -------------------- Helpers
def water_year_index(dt_index: pd.DatetimeIndex) -> pd.Index:
    """Return water year for each timestamp (WY starts Oct 1)."""
    return dt_index.year + (dt_index.month >= 10)

def date_in_month_day_range(date: pd.Timestamp, start_md: tuple, end_md: tuple) -> bool:
    """Check if a calendar date falls within a month-day (inclusive) window within the same WY."""
    y = date.year
    start = pd.Timestamp(year=y, month=start_md[0], day=start_md[1])
    end   = pd.Timestamp(year=y, month=end_md[0],   day=end_md[1])
    return (date >= start) & (date <= end)

def build_daily_q(series_15min: pd.Series, daily_min_frac=DAILY_MIN_FRAC) -> pd.Series:
    """Resample 15-min flow to daily mean with coverage rule."""
    cnt = series_15min.resample('D').count()
    mean = series_15min.resample('D').mean()
    good = cnt >= (FREQ_15MIN_PER_DAY * daily_min_frac)
    return mean.where(good)

def build_daily_swe(swe_df: pd.DataFrame) -> pd.Series:
    """Make a daily SWE series from hourly ERA5-Land. Uses daily max; then a light 3-day smooth."""
    ser = swe_df.set_index(pd.to_datetime(swe_df['date']))['snow_depth_water_equivalent'].sort_index()
    daily = ser.resample('D').max()
    return daily.rolling(3, center=True, min_periods=1).mean()

def wy_window(df_daily: pd.Series, wy: int, start_md: tuple, end_md: tuple) -> pd.Series:
    """Slice a daily series to the month-day window inside a given water year."""
    # Water year wy spans [Oct 1 (wy-1), Sep 30 (wy)]
    start_wy = pd.Timestamp(year=wy-1, month=10, day=1)
    end_wy   = pd.Timestamp(year=wy,   month=9,  day=30)
    s = df_daily.loc[start_wy:end_wy]
    if s.empty:
        return s
    # Build calendar-year specific bounds for window within wy
    y = wy  # spring window is within calendar year wy; winter window spans the same cal year wy
    start = pd.Timestamp(year=y, month=start_md[0], day=start_md[1])
    end   = pd.Timestamp(year=y, month=end_md[0],   day=end_md[1])
    return s.loc[start:end]

def wy_winter_window_for_peak(swe_daily: pd.Series, wy: int) -> pd.Series:
    """
    Winter peak window for a given water year:
      WY spans Oct 1 (wy-1) .. Sep 30 (wy)
      Winter window is Nov (wy-1) .. May (wy)
    """
    start = pd.Timestamp(year=wy-1, month=11, day=1)
    end   = pd.Timestamp(year=wy,   month=5,  day=31)
    return swe_daily.loc[start:end]

def wy_spring_window(q_daily: pd.Series, wy: int) -> pd.Series:
    """Spring melt window Apr 1 .. Jul 31 in the same WY calendar year."""
    start = pd.Timestamp(year=wy, month=4, day=1)
    end   = pd.Timestamp(year=wy, month=7, day=31)
    return q_daily.loc[start:end]

def assemble_wy_metrics(swe_daily: pd.Series, q_daily: pd.Series) -> pd.DataFrame:
    """
    Returns a WY-indexed DataFrame with:
      SWE_peak_m / SWE_peak_mm (Nov(wy-1)–May(wy))
      Q_AprJul_mean_cfs (mean over available days)
      Q_AprJul_peak1d_cfs (max daily mean over available days)
      Q_AprJul_peak3d_cfs (max 3-day mean over available days)
      Q_AprJul_total_acft_obs (sum over available days only)
      Q_AprJul_total_acft_scaled (obs * expected_days/present_days, for comparability)
      Q_days_present / Q_days_expected
    """
    swe_daily = swe_daily.sort_index()
    q_daily = q_daily.sort_index()

    # Overlapping WYs
    wys_swe = (swe_daily.index.year + (swe_daily.index.month >= 10)).unique()
    wys_q   = (q_daily.dropna().index.year + (q_daily.dropna().index.month >= 10)).unique()
    wys = sorted(set(wys_swe).intersection(set(wys_q)))

    rows = []
    for wy in wys:
        # Windows
        swe_win = swe_daily.loc[pd.Timestamp(wy-1, 11, 1): pd.Timestamp(wy, 5, 31)]
        q_start = pd.Timestamp(wy, 4, 1)
        q_end   = pd.Timestamp(wy, 7, 31)
        q_win   = q_daily.loc[q_start:q_end]

        if swe_win.empty or q_win.empty:
            continue

        # Coverage check in Apr–Jul
        expected = (q_end - q_start).days + 1
        present = int(q_win.notna().sum())
        if present < MIN_SPRING_DAY_FRAC * expected:
            continue

        # Metrics
        swe_peak_val  = float(swe_win.max())
        swe_peak_day  = swe_win.idxmax() if not swe_win.isna().all() else pd.NaT

        q_valid = q_win.dropna()
        q_mean  = float(q_valid.mean()) if not q_valid.empty else np.nan
        q_peak1 = float(q_valid.max()) if not q_valid.empty else np.nan
        q_peak3 = float(q_valid.rolling(3, center=True, min_periods=2).mean().max()) if not q_valid.empty else np.nan

        # Volume over available days (faithful) and scaled volume (comparable)
        q_total_obs_acft    = float((q_valid * CFS_DAY_TO_ACFT).sum()) if not q_valid.empty else np.nan
        scale = expected / present if present > 0 else np.nan
        q_total_scaled_acft = float(q_total_obs_acft * scale) if np.isfinite(scale) else np.nan

        rows.append({
            "WY": wy,
            "SWE_peak_m": swe_peak_val,
            "SWE_peak_mm": swe_peak_val * 1000.0,
            "SWE_peak_date": swe_peak_day,
            "Q_AprJul_mean_cfs": q_mean,
            "Q_AprJul_peak1d_cfs": q_peak1,
            "Q_AprJul_peak3d_cfs": q_peak3,
            "Q_AprJul_total_acft_obs": q_total_obs_acft,
            "Q_AprJul_total_acft_scaled": q_total_scaled_acft,
            "Q_days_present": present,
            "Q_days_expected": expected,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("assemble_wy_metrics: no qualifying years found. Check overlap or coverage thresholds.")
        if len(wys) == 0:
            print("  Diagnostic: no overlapping water years between SWE and Q.")
        return df
    return df.set_index("WY").sort_index()

def correlations_and_fit(df: pd.DataFrame, y_col: str, x_col: str):
    # Column check with a helpful message
    missing = [c for c in (x_col, y_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    d = df[[x_col, y_col]].dropna().copy()
    if d.empty:
        return None

    pearson = d[x_col].corr(d[y_col], method='pearson')
    spearman = d[x_col].corr(d[y_col], method='spearman')

    a, b = np.polyfit(d[x_col].values, d[y_col].values, 1)
    y_hat = a*d[x_col] + b
    r2_in = 1 - ((d[y_col] - y_hat).pow(2).sum() / ((d[y_col] - d[y_col].mean()).pow(2).sum()))

    # LOOCV
    y_cv = pd.Series(index=d.index, dtype=float)
    x = d[x_col].values; y = d[y_col].values
    for i in range(len(d)):
        m = np.ones(len(d), dtype=bool); m[i] = False
        a_i, b_i = np.polyfit(x[m], y[m], 1)
        y_cv.iloc[i] = a_i*x[i] + b_i
    rmse_cv = np.sqrt(((d[y_col] - y_cv)**2).mean())
    r2_cv = 1 - ((d[y_col] - y_cv).pow(2).sum() / ((d[y_col] - d[y_col].mean()).pow(2).sum()))

    return (
        {
            'pearson_r': float(pearson),
            'spearman_rho': float(spearman),
            'ols_slope': float(a),
            'ols_intercept': float(b),
            'r2_in_sample': float(r2_in),
            'r2_loocv': float(r2_cv),
            'rmse_loocv': float(rmse_cv),
            'n_years': int(len(d)),
        },
        y_hat, y_cv
    )


def annotate_wy_labels(ax, df, xcol, ycol, label_col=None, fontsize=8):
    """
    Add year labels to a scatter. Uses small staggered offsets to reduce overlaps.
    label_col defaults to the index (WY) if not provided.
    """
    d = df[[xcol, ycol]].copy()
    d['__label__'] = df.index if label_col is None else df[label_col].values

    # Sort by y so offsets alternate up/down in order
    d = d.sort_values(ycol)
    for i, (xv, yv, lab) in enumerate(zip(d[xcol], d[ycol], d['__label__'])):
        # Simple stagger pattern to reduce collisions
        dx = (-6, 6, 0)[i % 3]      # points
        dy = (8, -8)[(i // 3) % 2]  # points
        ax.annotate(
            f"{int(lab)}",
            xy=(xv, yv),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center", va="center",
            fontsize=fontsize, color="#264653",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ----------------------- CONFIG (tweak as needed)
FREQ_15MIN_PER_DAY = 96
DAILY_MIN_FRAC     = 0.80   # require >=80% of 15-min samples to accept a daily mean

WINDOW_START = (4, 1)       # analysis window: Apr 1 ...
WINDOW_END   = (7, 31)      # ... to Jul 31
CORE_START   = (5, 1)       # core SPE window: May 1 ...
CORE_END     = (6, 30)      # ... to Jun 30

SMOOTH_DAYS  = 3            # rolling mean smoothing on daily Q
THRESH_PCTL  = 0.90         # run threshold = 90th percentile of window flows
MIN_DUR_DAYS = 4            # minimum contiguous days above threshold
MAX_CV_TOP   = 0.20         # "flatness" requirement: CV of 5-day window around peak
REQ_CORE_PEAK = True        # require the event peak date to fall in core window

@dataclass
class SPEParams:
    thresh_pctl: float = THRESH_PCTL
    min_dur_days: int = MIN_DUR_DAYS
    max_cv_top: float = MAX_CV_TOP
    smooth_days: int = SMOOTH_DAYS
    require_core_peak: bool = REQ_CORE_PEAK

# ----------------------- Utilities
def build_daily_q(series_15min: pd.Series, daily_min_frac: float = DAILY_MIN_FRAC) -> pd.Series:
    """Resample 15-min series to daily mean with coverage rule."""
    series_15min = series_15min.sort_index()
    cnt = series_15min.resample('D').count()
    mean = series_15min.resample('D').mean()
    good = cnt >= (FREQ_15MIN_PER_DAY * daily_min_frac)
    return mean.where(good)

def water_year(ts: pd.Timestamp) -> int:
    return ts.year + (ts.month >= 10)

def wy_window(q_daily: pd.Series, wy: int, start_md: tuple, end_md: tuple) -> pd.Series:
    start = pd.Timestamp(wy, start_md[0], start_md[1])
    end   = pd.Timestamp(wy, end_md[0],   end_md[1])
    return q_daily.loc[start:end]

def core_contains(ts: pd.Timestamp) -> bool:
    md = (ts.month, ts.day)
    return (md >= CORE_START) and (md <= CORE_END)

def _runs_above(series: pd.Series, thr: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return list of (start, end) for contiguous runs where series >= thr."""
    mask = (series >= thr).astype('int')
    dif = mask.diff().fillna(0)
    starts = list(series.index[dif == 1])
    ends   = list(series.index[dif == -1] - pd.Timedelta(days=1))
    # handle run at the beginning
    if mask.iloc[0] == 1:
        starts = [series.index[0]] + starts
    # handle run at the end
    if mask.iloc[-1] == 1:
        ends = ends + [series.index[-1]]
    return list(zip(starts, ends))

# ----------------------- Detection logic
def detect_spe_in_wy(q_daily: pd.Series, wy: int, params: SPEParams = SPEParams()) -> dict | None:
    """
    Detect a Spring Peak Event in water year `wy`. Returns a dict of metrics (or None if not detected).
    """
    # Slice Apr–Jul window and smooth a bit
    s = wy_window(q_daily, wy, WINDOW_START, WINDOW_END)
    if s.empty:
        return None
    if params.smooth_days and params.smooth_days > 1:
        s_smooth = s.rolling(params.smooth_days, center=True, min_periods=1).mean()
    else:
        s_smooth = s

    # Threshold based on window percentile
    thr = float(np.nanpercentile(s_smooth.dropna().values, params.thresh_pctl * 100.0)) if s_smooth.notna().any() else np.nan
    if not np.isfinite(thr):
        return None

    # Find candidate runs above threshold
    runs = _runs_above(s_smooth.dropna(), thr)
    if not runs:
        return None

    # Score each run; pick the best by (duration, peak magnitude)
    best = None
    for start, end in runs:
        run = s_smooth.loc[start:end]
        dur = (end - start).days + 1
        peak_val = float(run.max())
        peak_day = run.idxmax()
        # Flatness near the peak: 5-day window CV
        win = s_smooth.loc[max(s_smooth.index.min(), peak_day - pd.Timedelta(days=2)):
                           min(s_smooth.index.max(), peak_day + pd.Timedelta(days=2))]
        cv_top = float(win.std() / win.mean()) if win.mean() > 0 else np.inf

        # Keep the best candidate by longest duration then higher peak
        score = (dur, peak_val)
        cand = dict(
            WY=wy, start=start, end=end, duration_days=dur, peak_cfs=peak_val, peak_day=peak_day,
            threshold_cfs=thr, cv_top=cv_top
        )
        if (best is None) or (score > (best['duration_days'], best['peak_cfs'])):
            best = cand

    if best is None:
        return None

    # Classification rules
    is_long_enough = best['duration_days'] >= params.min_dur_days
    is_flat_enough = best['cv_top'] <= params.max_cv_top
    in_core = (not params.require_core_peak) or core_contains(best['peak_day'])

    best['classified_SPE'] = bool(is_long_enough and is_flat_enough and in_core)
    best['reason'] = {
        'duration_ok': is_long_enough,
        'flatness_ok': is_flat_enough,
        'core_ok': in_core,
    }
    return best

def detect_spe_all_years(x, params=None, daily_min_frac: float = 0.80, prefer_col: str = "release_cfs") -> pd.DataFrame:
    """
    Accepts either a 15-min series OR a daily series/DataFrame.
    For DataFrame, picks `prefer_col` (default 'release_cfs').
    """
    if params is None:
        params = SPEParams()

    q_daily = _build_daily_q_generic(x, daily_min_frac=daily_min_frac, prefer_col=prefer_col)
    if q_daily.dropna().empty:
        return pd.DataFrame()

    wys = sorted({ts.year + (ts.month >= 10) for ts in q_daily.dropna().index})
    rows = []
    for wy in wys:
        res = detect_spe_in_wy(q_daily, wy, params=params)
        if res is None:
            rows.append({'WY': wy, 'classified_SPE': False, 'note': 'no-run'})
        else:
            res = res.copy(); res['WY'] = wy
            rows.append(res)
    df = pd.DataFrame(rows).set_index('WY').sort_index()
    if 'start' in df.columns:
        df['start'] = pd.to_datetime(df['start']); df['end'] = pd.to_datetime(df['end'])
        if 'peak_day' in df.columns:
            df['peak_day'] = pd.to_datetime(df['peak_day'])
    return df

# ----------------------- Quick plot helper (optional)
def plot_wy_spe(cfs_15min: pd.Series, wy: int, params: SPEParams = SPEParams()):
    """Plot Apr–Jul for a given WY with the detected event highlighted."""
    q_daily = build_daily_q(cfs_15min)
    s = wy_window(q_daily, wy, WINDOW_START, WINDOW_END)
    if s.empty:
        print(f"WY {wy}: no data")
        return
    s_smooth = s.rolling(params.smooth_days, center=True, min_periods=1).mean()

    res = detect_spe_in_wy(q_daily, wy, params=params)
    thr = float(np.nanpercentile(s_smooth.dropna().values, params.thresh_pctl * 100.0)) if s_smooth.notna().any() else np.nan

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(s.index, s.values, lw=1.0, alpha=0.35, label='Daily mean Q (raw)')
    ax.plot(s_smooth.index, s_smooth.values, lw=2.0, label=f'{params.smooth_days}-day smooth')
    if np.isfinite(thr):
        ax.axhline(thr, ls='--', alpha=0.5, label=f'{int(params.thresh_pctl*100)}th pct = {thr:.0f} cfs')

    if res and res.get('classified_SPE', False):
        ax.axvspan(res['start'], res['end'], color='#2a9d8f', alpha=0.15, label='Detected SPE window')
        ax.scatter([res['peak_day']], [res['peak_cfs']], zorder=5, s=40, color='#e76f51', edgecolor='white', linewidth=0.8)
        ax.annotate(f"peak {res['peak_cfs']:.0f} cfs\nCV={res['cv_top']:.2f}", 
                    (res['peak_day'], res['peak_cfs']), xytext=(6,6), textcoords='offset points')

    ax.set_title(f"SJ @ Archuleta – WY {wy} Apr–Jul")
    ax.set_ylabel("Q (cfs)")
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc='upper left')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def _to_daily(x, daily_min_frac: float = 0.80, prefer_col: str = "release_cfs") -> pd.Series:
    """
    Accept Series or DataFrame; returns a daily Series.
    - If sub-daily: resamples to daily mean with coverage rule (>= daily_min_frac * 96).
    - If already daily: aligns to 'D' without filling.
    """
    # pick series
    if isinstance(x, pd.DataFrame):
        if prefer_col in x.columns:
            s = x[prefer_col].copy()
        else:
            numcols = x.select_dtypes(include="number").columns
            if len(numcols) == 1:
                s = x[numcols[0]].copy()
            else:
                raise ValueError(f"plot_spe_timeline: provide a Series or a DataFrame with '{prefer_col}'.")
    elif isinstance(x, pd.Series):
        s = x.copy()
    else:
        raise TypeError("plot_spe_timeline: x must be a pandas Series or DataFrame.")

    # ensure datetime index
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    if s.empty:
        return s.asfreq("D")

    # cadence detect
    diffs = s.index.to_series().diff().dropna()
    med = diffs.median() if not diffs.empty else pd.Timedelta(days=1)

    if med < pd.Timedelta(hours=12):  # sub-daily -> resample with coverage rule
        cnt = s.resample("D").count()
        mean = s.resample("D").mean()
        good = cnt >= (FREQ_15MIN_PER_DAY * daily_min_frac)
        return mean.where(good)
    else:  # already daily
        return s.asfreq("D")


def _build_daily_q(series_15min: pd.Series, daily_min_frac: float = 0.80) -> pd.Series:
    s = series_15min.sort_index()
    cnt = s.resample('D').count()
    mean = s.resample('D').mean()
    good = cnt >= (FREQ_15MIN_PER_DAY * daily_min_frac)
    return mean.where(good)

def plot_spe_timeline(
    x,                        # Series or DataFrame (15-min OR daily)
    spe_df: pd.DataFrame,     # WY-indexed, with 'classified_SPE' True/False
    prefer_col: str = "release_cfs",
    daily_min_frac: float = 0.80,
    shade: bool = True,
    color_shade: str = "#e9c46a",
    shade_alpha: float = 0.22,
    success_color: str = "tab:blue",
    line_color: str = "#264653",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    title: str = "Spring Peak Event attempts"
):
    """
    Plots daily discharge from `x`; shades FULL water years (Oct1–Sep30) for successful SPE years.
    X-axis is calendar years (Jan 1 ticks) rotated 60°. Successful years’ tick labels colored blue.
    """
    q_daily = _to_daily(x, daily_min_frac=daily_min_frac, prefer_col=prefer_col)
    # subselect
    q_daily = q_daily.loc[
        pd.to_datetime(start) if start is not None else q_daily.index.min()
        : pd.to_datetime(end)   if end   is not None else q_daily.index.max()
    ]

    fig, ax = plt.subplots(figsize=(11.5, 4.8))

    # plot raw daily line (only if we have any finite values)
    if q_daily.dropna().empty:
        print("plot_spe_timeline: daily series is empty after preprocessing.")
    else:
        ax.plot(q_daily.index, q_daily.values, lw=1.1, color=line_color, alpha=0.85, label="Daily mean Q")

    # Shade full WYs
    if shade and spe_df is not None and not spe_df.empty and 'classified_SPE' in spe_df.columns:
        success_wys = spe_df.index[spe_df['classified_SPE'] == True].astype(int).tolist()
        for wy in success_wys:
            d0 = pd.Timestamp(wy-1, 10, 1)   # Oct 1 (prior cal year)
            d1 = pd.Timestamp(wy,   9, 30)   # Sep 30 (current cal year)
            left  = max(d0, q_daily.index.min()) if not q_daily.empty else d0
            right = min(d1, q_daily.index.max()) if not q_daily.empty else d1
            if right > left:
                ax.axvspan(left, right, color=color_shade, alpha=shade_alpha, zorder=0)

    # Year ticks at Jan 1, rotated 60°
    if not q_daily.empty:
        start_year = q_daily.index.min().year
        end_year   = q_daily.index.max().year
    else:
        # fall back to years spanned by shaded regions if series is empty
        if spe_df is not None and not spe_df.empty:
            start_year = int(spe_df.index.min()) - 1
            end_year   = int(spe_df.index.max())
        else:
            start_year = end_year = pd.Timestamp.today().year

    years = list(range(start_year, end_year + 1))
    ticks = [pd.Timestamp(y, 1, 1) for y in years]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(y) for y in years])
    plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

    # Color successful years' labels blue (WY == calendar year at Jan 1)
    success_set = set(int(y) for y in spe_df.index[spe_df['classified_SPE'] == True]) if spe_df is not None else set()
    for y, lbl in zip(years, ax.get_xticklabels()):
        lbl.set_color(success_color if y in success_set else "black")

    # Cosmetics
    ax.set_ylabel("Q (cfs)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # Legend for shading
    handles = []
    if shade:
        handles.append(Patch(facecolor=color_shade, edgecolor="none", alpha=shade_alpha, label="Successful year (shaded WY)"))
    if handles:
        ax.legend(handles=handles, frameon=False, loc="upper left")

    plt.tight_layout()
    plt.show()
    return ax


def load_usbr_navajo_daily(path_csv=r"X:\Research\DeepReservoir\Code\DeepReservoir\data\Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv"):
    """
    Returns a daily-indexed DataFrame with standardized columns:
      date (index), elev_ft, storage_af, evap_af, inflow_cfs, munreg_inflow_cfs, release_cfs
    Handles two-digit years like '7-Jun-67' by pivoting >2025 back 100 years.
    """
    df = pd.read_csv(path_csv)
    # Drop junk columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    # Standardize names
    colmap = {
        "Date": "date",
        "Elevation (feet)": "elev_ft",
        "Storage (af)": "storage_af",
        "Evaporation (af)": "evap_af",
        "Inflow** (cfs)": "inflow_cfs",
        "Modified Unregulated Inflow^ (cfs)": "munreg_inflow_cfs",
        "Total Release (cfs)": "release_cfs",
    }
    df = df.rename(columns=colmap)

    # Parse dates like '7-Jun-67' safely
    dt = pd.to_datetime(df["date"], format="%d-%b-%y", errors="coerce")
    # Fix 20xx mis-parses for 1967..1968 style years
    dt = dt.where(dt.dt.year <= 2025, dt - pd.DateOffset(years=100))
    df["date"] = dt

    # Coerce numeric columns
    for c in ("elev_ft","storage_af","evap_af","inflow_cfs","munreg_inflow_cfs","release_cfs"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
    # Ensure daily frequency (don’t up/downsample values)
    df = df[~df.index.duplicated(keep="first")]
    return df


def wy_index(idx: pd.DatetimeIndex) -> pd.Index:
    return idx.year + (idx.month >= 10)

# Helper: Apr–Jul mean by WY for any daily Q series
def aprjul_mean_by_wy(q_daily: pd.Series) -> pd.Series:
    q = q_daily.sort_index()
    wys = wy_index(q.dropna().index).unique()
    out = {}
    for wy in sorted(wys):
        s = q.loc[pd.Timestamp(wy,4,1): pd.Timestamp(wy,7,31)]
        if s.notna().sum() >= 0.6 * ((pd.Timestamp(wy,7,31) - pd.Timestamp(wy,4,1)).days + 1):
            out[wy] = float(s.mean())
    return pd.Series(out, name="Animas_AprJul_mean_cfs")

# Helper: “pre-SPR” storage by WY (pick one: 'mar_mean' or 'mar31')
def prespring_storage_by_wy(usbr_df: pd.DataFrame,
                            method: str = "feb_mean",   # "feb_mean" or "window"
                            window_days: int | None = None,
                            min_frac: float = 0.60) -> pd.Series:
    """
    Returns a Series indexed by WY with pre-SPR storage (acre-feet).

    method:
      - "feb_mean" : mean(storage) over full February of WY (Feb 1..Feb last day).
      - "window"   : mean(storage) over the last `window_days` days ending Feb last day.
    Rules:
      - Requires >= min_frac fraction of the expected days present in the window.
      - WY is the water year (Oct–Sep); February is in the same calendar year as the WY.
    """
    s = usbr_df["storage_af"].copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s = s.sort_index().asfreq("D")  # keep NaN gaps; no filling

    years = range(s.index.min().year, s.index.max().year + 1)
    out = {}
    for wy in years:
        feb_end = pd.Timestamp(wy, 3, 1) - pd.Timedelta(days=1)  # Feb 28 or 29
        feb_start = pd.Timestamp(wy, 2, 1)

        if method == "feb_mean":
            win_start, win_end = feb_start, feb_end
            expected = (win_end - win_start).days + 1
        elif method == "window":
            if not window_days or window_days < 1:
                raise ValueError("For method='window', provide window_days >= 1.")
            win_end = feb_end
            win_start = win_end - pd.Timedelta(days=window_days - 1)
            expected = window_days
        else:
            raise ValueError("method must be 'feb_mean' or 'window'.")

        win = s.loc[win_start:win_end].dropna()
        if win.empty:
            continue
        if win.shape[0] >= min_frac * expected:
            out[wy] = float(win.mean())

    return pd.Series(out, name="preSPR_storage_af").sort_index()


def _extract_series(x, prefer_col: str = "release_cfs") -> pd.Series:
    """Accept a Series or a DataFrame; return a single numeric Series."""
    if isinstance(x, pd.Series):
        s = x
    elif isinstance(x, pd.DataFrame):
        if prefer_col in x.columns:
            s = x[prefer_col]
        else:
            numcols = x.select_dtypes(include="number").columns
            if len(numcols) == 1:
                s = x[numcols[0]]
            else:
                raise ValueError(f"_extract_series: can't choose a column. "
                                 f"Provide a Series or include '{prefer_col}' in the DataFrame.")
    else:
        raise TypeError("_extract_series: expected Series or DataFrame.")
    # Ensure datetime index
    if not np.issubdtype(s.index.dtype, np.datetime64):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            s = s.copy()
            s.index = pd.to_datetime(s.index)
    return s.sort_index()

def _build_daily_q_generic(x, daily_min_frac: float = 0.80, prefer_col: str = "release_cfs") -> pd.Series:
    """
    If input is sub-daily -> resample to daily with coverage rule.
    If input is already daily -> return as-is (no coverage rule).
    """
    s = _extract_series(x, prefer_col=prefer_col)
    s = s[~s.index.duplicated(keep="first")]  # ensure one row per timestamp

    # detect cadence via median time step
    diffs = s.index.to_series().diff().dropna()
    med = diffs.median() if not diffs.empty else pd.Timedelta(days=1)

    if med < pd.Timedelta(hours=12):  # treat as sub-daily
        cnt = s.resample("D").count()
        mean = s.resample("D").mean()
        good = cnt >= (FREQ_15MIN_PER_DAY * daily_min_frac)
        q_daily = mean.where(good)
    else:
        # assume daily; just align to D without filling
        q_daily = s.asfreq("D")

    return pd.to_numeric(q_daily, errors="coerce").sort_index()


def wy_index(idx: pd.DatetimeIndex) -> pd.Index:
    return idx.year + (idx.month >= 10)

def aprjul_mean_by_wy(q_daily: pd.Series, min_frac: float = 0.60) -> pd.Series:
    q = q_daily.sort_index()
    wys = wy_index(q.dropna().index).unique()
    out = {}
    for wy in sorted(wys):
        s = q.loc[pd.Timestamp(wy,4,1): pd.Timestamp(wy,7,31)]
        if s.notna().sum() >= min_frac * ((pd.Timestamp(wy,7,31) - pd.Timestamp(wy,4,1)).days + 1):
            out[wy] = float(s.mean())
    return pd.Series(out, name="Animas_AprJul_mean_cfs")

def prespring_storage_by_wy(usbr_df: pd.DataFrame, method: str = "mar_mean") -> pd.Series:
    st = usbr_df["storage_af"].sort_index()
    years = range(st.index.min().year+1, st.index.max().year+1)
    out = {}
    for wy in years:
        mar = st.loc[pd.Timestamp(wy,3,1): pd.Timestamp(wy,3,31)]
        if mar.empty or mar.dropna().empty:
            continue
        if method == "mar_mean":
            out[wy] = float(mar.dropna().mean())
        elif method == "mar31":
            target = pd.Timestamp(wy,3,31)
            if target in st.index and pd.notna(st.loc[target]):
                out[wy] = float(st.loc[target])
            else:
                win = st.loc[target - pd.Timedelta(days=7): target + pd.Timedelta(days=7)].dropna()
                if not win.empty:
                    out[wy] = float(win.loc[(win.index - target).abs().argmin()])
        else:
            raise ValueError("method must be 'mar_mean' or 'mar31'")
    return pd.Series(out, name="preSPR_storage_af")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

def plot_animas_vs_prespring_storage(animas_q_daily: pd.Series,
                                     usbr_df: pd.DataFrame,
                                     spe_df: pd.DataFrame | None = None,
                                     storage_method: str = "feb_mean",   # "feb_mean" or "window"
                                     window_days: int | None = None,      # used if method="window"
                                     title: str | None = None,
                                     year_split: int = 2000,
                                     annotate_wy: bool = False,
                                     annotate_fontsize: int = 8,
                                     point_size: float = 72,
                                     colors: dict | None = None):
    """
    X = pre-SPR storage (Feb mean by default), Y = Animas Apr–Jul mean Q.
    Colors: WY<year_split -> gray; WY>=year_split & SPR=False -> orange; SPR=True -> blue.
    """
    if colors is None:
        colors = {"pre": "#9ca3af", "no": "#f59e0b", "spr": "#2563eb"}  # gray/amber/royal blue

    # Build Y (Apr–Jul mean by WY)
    q = animas_q_daily.sort_index()
    wys = (q.index.year + (q.index.month >= 10))
    out = {}
    for wy in sorted(set(wys)):
        s = q.loc[pd.Timestamp(wy,4,1): pd.Timestamp(wy,7,31)]
        exp = (pd.Timestamp(wy,7,31) - pd.Timestamp(wy,4,1)).days + 1
        if s.notna().sum() >= 0.60 * exp:
            out[wy] = float(s.mean())
    y_aprjul = pd.Series(out, name="Animas_AprJul_mean_cfs")

    # Build X (pre-SPR storage via February)
    x_storage = prespring_storage_by_wy(usbr_df, method=storage_method,
                                        window_days=window_days, min_frac=0.60)

    df = pd.concat([x_storage, y_aprjul], axis=1).dropna()
    if df.empty:
        print("No overlapping WYs between Animas flow and USBR storage (pre-SPR).")
        return

    # SPR flags
    if spe_df is not None and not spe_df.empty and "classified_SPE" in spe_df.columns:
        spr_flags = spe_df["classified_SPE"].reindex(df.index).fillna(False).astype(bool)
    else:
        spr_flags = pd.Series(False, index=df.index)

    df["cat"] = ["pre" if wy < year_split else ("spr" if bool(spr_flags.loc[wy]) else "no")
                 for wy in df.index]

    # Plot
    plt.figure(figsize=(7.8, 5.6))
    ax = plt.gca()
    for cat in ["pre", "no", "spr"]:
        dd = df[df["cat"] == cat]
        if dd.empty: continue
        ax.scatter(dd["preSPR_storage_af"], dd["Animas_AprJul_mean_cfs"],
                   s=point_size, alpha=0.9, color=colors[cat],
                   edgecolor="white", linewidth=1.0,
                   label=("WY < {0}".format(year_split) if cat=="pre"
                          else ("SPR attempted" if cat=="spr"
                                else "No SPR (WY ≥ {0})".format(year_split))))
        if annotate_wy:
            for wy, r in dd.iterrows():
                ax.annotate(str(int(wy)), (r["preSPR_storage_af"], r["Animas_AprJul_mean_cfs"]),
                            xytext=(6,6), textcoords="offset points",
                            fontsize=annotate_fontsize, color="#334155",
                            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                            zorder=5, annotation_clip=False)

    label_tag = "Feb mean" if storage_method=="feb_mean" else f"last {window_days} days ≤ Feb-end"
    ax.set_xlabel(f"Pre-Spring storage (af) [{label_tag}]")
    ax.set_ylabel("Animas Apr–Jul mean Q (cfs)")
    ax.set_title(title or "Apr–Jul mean Q vs pre-Spring storage (Feb-based)")
    ax.minorticks_on(); ax.grid(which="major", linestyle="-", alpha=0.25)
    ax.grid(which="minor", linestyle=":", alpha=0.12)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    handles, labels = ax.get_legend_handles_labels()
    order = []
    for want in [f"WY < {year_split}", f"No SPR (WY ≥ {year_split})", "SPR attempted"]:
        order += [i for i,lbl in enumerate(labels) if lbl == want]
    if order:
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  frameon=False, loc="best")

    plt.tight_layout(); plt.show()


def prespring_storage_by_wy(usbr_df: pd.DataFrame, method: str = "mar_mean") -> pd.Series:
    st = usbr_df["storage_af"].sort_index()
    years = range(st.index.min().year+1, st.index.max().year+1)
    out = {}
    for wy in years:
        mar = st.loc[pd.Timestamp(wy,3,1): pd.Timestamp(wy,3,31)]
        if mar.empty or mar.dropna().empty:
            continue
        if method == "mar_mean":
            out[wy] = float(mar.dropna().mean())
        elif method == "mar31":
            target = pd.Timestamp(wy,3,31)
            if target in st.index and pd.notna(st.loc[target]):
                out[wy] = float(st.loc[target])
            else:
                win = st.loc[target - pd.Timedelta(days=7): target + pd.Timedelta(days=7)].dropna()
                if not win.empty:
                    out[wy] = float(win.loc[(win.index - target).abs().argmin()])
        else:
            raise ValueError("method must be 'mar_mean' or 'mar31'")
    return pd.Series(out, name="preSPR_storage_af")

def plot_swe_vs_prespring_storage(wy_metrics: pd.DataFrame,
                                  usbr_df: pd.DataFrame,
                                  spe_df: pd.DataFrame | None = None,
                                  swe_col: str = "SWE_peak_mm",
                                  storage_method: str = "feb_mean",   # "feb_mean" or "window"
                                  window_days: int | None = None,
                                  title: str | None = None,
                                  year_split: int = 2000,
                                  annotate_wy: bool = False,
                                  annotate_fontsize: int = 8,
                                  point_size: float = 72,
                                  colors: dict | None = None):
    """
    X = pre-SPR storage (Feb mean or last N days to Feb-end), Y = wy_metrics[swe_col].
    Colors: WY<year_split -> gray; WY>=year_split & SPR=False -> orange; SPR=True -> blue.
    """
    if colors is None:
        colors = {"pre": "#9ca3af", "no": "#f59e0b", "spr": "#2563eb"}

    if swe_col not in wy_metrics.columns:
        raise KeyError(f"{swe_col!r} not in wy_metrics: {list(wy_metrics.columns)}")

    x_storage = prespring_storage_by_wy(usbr_df, method=storage_method,
                                        window_days=window_days, min_frac=0.60)
    y_swe = wy_metrics[swe_col].rename("SWE_y")
    df = pd.concat([x_storage, y_swe], axis=1).dropna()
    if df.empty:
        print("No overlapping WYs between SWE metrics and USBR storage (pre-SPR).")
        return

    if spe_df is not None and not spe_df.empty and "classified_SPE" in spe_df.columns:
        spr_flags = spe_df["classified_SPE"].reindex(df.index).fillna(False).astype(bool)
    else:
        spr_flags = pd.Series(False, index=df.index)

    df["cat"] = ["pre" if wy < year_split else ("spr" if bool(spr_flags.loc[wy]) else "no")
                 for wy in df.index]

    plt.figure(figsize=(7.8, 5.6))
    ax = plt.gca()
    for cat in ["pre", "no", "spr"]:
        dd = df[df["cat"] == cat]
        if dd.empty: continue
        ax.scatter(dd["preSPR_storage_af"], dd["SWE_y"],
                   s=point_size, alpha=0.9, color=colors[cat],
                   edgecolor="white", linewidth=1.0,
                   label=("WY < {0}".format(year_split) if cat=="pre"
                          else ("SPR attempted" if cat=="spr"
                                else "No SPR (WY ≥ {0})".format(year_split))))
        if annotate_wy:
            for wy, r in dd.iterrows():
                ax.annotate(str(int(wy)),
                            (r["preSPR_storage_af"], r["SWE_y"]),
                            xytext=(6,6), textcoords="offset points",
                            fontsize=annotate_fontsize, color="#334155",
                            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                            zorder=5, annotation_clip=False)

    label_tag = "Feb mean" if storage_method=="feb_mean" else f"last {window_days} days ≤ Feb-end"
    ax.set_xlabel(f"Pre-Spring storage (af) [{label_tag}]")
    ax.set_ylabel(swe_col.replace("_", " "))
    ax.set_title(title or f"{swe_col} vs pre-Spring storage (Feb-based)")
    ax.minorticks_on(); ax.grid(which="major", linestyle="-", alpha=0.25)
    ax.grid(which="minor", linestyle=":", alpha=0.12)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    handles, labels = ax.get_legend_handles_labels()
    order = []
    for want in [f"WY < {year_split}", f"No SPR (WY ≥ {year_split})", "SPR attempted"]:
        order += [i for i,lbl in enumerate(labels) if lbl == want]
    if order:
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  frameon=False, loc="best")

    plt.tight_layout(); plt.show()
