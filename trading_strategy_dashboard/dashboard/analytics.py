"""Core analytics engine for the trading strategy dashboard.

Provides time-series construction (equity, drawdown, returns), date-range
filtering, annualization heuristics, and the full performance-metrics
computation (Sharpe, Sortino, CAGR, Calmar, hit rate, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dashboard.config import RANGE_OPTIONS


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeriesPack:
    """Immutable bundle of derived time-series for a set of products.

    Attributes:
        dates:    Original date column from the source DataFrame.
        pnl:     Daily PnL (sum across selected products).
        equity:  Cumulative equity curve starting at *initial_capital*.
        hwm:     Running high-water mark of the equity curve.
        drawdown: Percentage drawdown from the high-water mark (0 at peaks,
                  negative during drawdowns).
        returns: Simple daily returns (PnL / previous-day equity).
    """

    dates: pd.Series
    pnl: pd.Series
    equity: pd.Series
    hwm: pd.Series
    drawdown: pd.Series
    returns: pd.Series


# ---------------------------------------------------------------------------
# Series construction
# ---------------------------------------------------------------------------

def compute_series(df: pd.DataFrame, products: List[str], initial_capital: float) -> SeriesPack:
    """Build equity, drawdown, and return series from raw PnL columns.

    Args:
        df:              DataFrame with a ``"date"`` column and PnL columns.
        products:        List of column names to aggregate.
        initial_capital: Starting capital for the equity curve.

    Returns:
        A populated :class:`SeriesPack`.  Missing ``"date"`` column or an empty
        frame yields an empty pack rather than raising.
    """
    if df is None or "date" not in df.columns:
        empty = pd.Series(dtype="float64")
        return SeriesPack(dates=empty, pnl=empty, equity=empty, hwm=empty, drawdown=empty, returns=empty)

    dates = df["date"]

    # Drop any product names that are not actual columns (stale UI state).
    products = [p for p in products if p in df.columns]

    if len(products) == 0:
        pnl = pd.Series(np.zeros(len(df)), index=df.index)
    else:
        pnl = df[products].sum(axis=1)

    equity = initial_capital + pnl.cumsum()
    hwm = equity.cummax()

    # BUG-FIX: Percentage drawdown -- divide by HWM, replacing 0 with NaN
    # to avoid division-by-zero.  Result is 0 at peaks, negative during drawdowns.
    dd = (equity / hwm.replace(0, np.nan)) - 1.0
    dd = dd.fillna(0.0)

    # Simple returns: daily PnL divided by previous day's equity.
    # Replace zero equity with NaN to prevent division-by-zero, then clamp
    # to [-1, 10] to suppress extreme outliers from tiny denominators.
    prev_eq = equity.shift(1).replace(0, np.nan)
    rets = (pnl / prev_eq).fillna(0.0)
    rets = rets.clip(-1.0, 10.0)

    return SeriesPack(dates=dates, pnl=pnl, equity=equity, hwm=hwm, drawdown=dd, returns=rets)


# ---------------------------------------------------------------------------
# Annualization helpers
# ---------------------------------------------------------------------------

def annualize_factor_from_dates(dates: pd.Series) -> int:
    """Infer the annualization factor from the cadence of a date series.

    Uses the median gap between consecutive dates:
      - <= 2 days  -> 252 (daily / business-day)
      - <= 8 days  -> 52  (weekly)
      - otherwise  -> 12  (monthly)

    Args:
        dates: Series of date-like values.

    Returns:
        Annualization factor (252, 52, or 12).
    """
    if len(dates) < 3:
        return 252
    d = pd.to_datetime(dates)
    deltas = d.diff().dropna().dt.days
    med = deltas.median()
    if pd.isna(med):
        return 252
    if med <= 2:
        return 252
    if med <= 8:
        return 52
    return 12


def padded_date_range(dates: pd.Series, pad_frac: float = 0.1) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return a padded (start, end) range for axis display.

    Adds *pad_frac* of the total span on each side so traces do not
    touch the plot borders.

    Args:
        dates:    Series of date-like values.
        pad_frac: Fraction of total date span to use as padding.

    Returns:
        ``(start - pad, end + pad)`` tuple, or ``None`` if no valid dates.
    """
    valid_dates = pd.to_datetime(dates).dropna()
    if valid_dates.empty:
        return None

    start = valid_dates.min()
    end = valid_dates.max()
    span = end - start
    pad = span * pad_frac if span > pd.Timedelta(0) else pd.Timedelta(days=30)
    return start - pad, end + pad


# ---------------------------------------------------------------------------
# Date-range filtering
# ---------------------------------------------------------------------------

def filter_df_by_range(df: pd.DataFrame, range_key: Optional[str]) -> pd.DataFrame:
    """Filter a DataFrame to a predefined date range.

    Supported *range_key* values: ``"1M"``, ``"3M"``, ``"YTD"``, ``"1Y"``,
    ``"All"`` (or ``None``).  Falls back to the full DataFrame when the
    filtered result would be empty.

    Args:
        df:        DataFrame with a ``"date"`` column.
        range_key: One of the :data:`RANGE_OPTIONS` or ``None``.

    Returns:
        Filtered (or original) DataFrame.
    """
    if df is None or df.empty:
        return df

    range_key = range_key or "All"
    if range_key == "All":
        return df

    dates = pd.to_datetime(df["date"], errors="coerce")
    end = dates.max()
    if pd.isna(end):
        return df

    if range_key == "1M":
        start = end - pd.DateOffset(months=1)
    elif range_key == "3M":
        start = end - pd.DateOffset(months=3)
    elif range_key == "YTD":
        start = pd.Timestamp(end.year, 1, 1)
    elif range_key == "1Y":
        start = end - pd.DateOffset(years=1)
    else:
        return df

    mask = (dates >= start) & (dates <= end)
    filtered = df.loc[mask].copy()
    return filtered if not filtered.empty else df


# ---------------------------------------------------------------------------
# Range-cycling helpers (for toolbar buttons)
# ---------------------------------------------------------------------------

def next_range_key(current: Optional[str]) -> str:
    """Cycle to the next date-range option.

    Args:
        current: The currently selected range key (or ``None``).

    Returns:
        The next range key in :data:`RANGE_OPTIONS`, wrapping around.
    """
    current_value = current or "All"
    try:
        idx = RANGE_OPTIONS.index(current_value)
    except ValueError:
        idx = len(RANGE_OPTIONS) - 1
    return RANGE_OPTIONS[(idx + 1) % len(RANGE_OPTIONS)]


def range_cycle_label(*ranges: Optional[str]) -> str:
    """Generate a label summarising the current range state of all panels.

    Args:
        *ranges: One range key per panel.

    Returns:
        ``"All Panels: <key>"`` when all panels share the same range,
        otherwise ``"All Panels: Mixed"``.
    """
    normalized = [value or "All" for value in ranges]
    unique = {value for value in normalized}
    if len(unique) == 1:
        return f"All Panels: {normalized[0]}"
    return "All Panels: Mixed"


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(sp: SeriesPack, rf_annual: float = 0.0) -> List[dict]:
    """Compute a comprehensive set of performance metrics from a SeriesPack.

    The returned list is ready to be rendered as rows in a Dash DataTable.

    Metrics include: Total PnL, Final Equity, CAGR, Volatility, Sharpe,
    Sortino, Max Drawdown, Calmar, Hit Rate, Profit Factor, Avg Win/Loss,
    Best/Worst Day PnL, Median Daily PnL, Std Daily PnL, Monthly stats,
    Skew, Kurtosis, Expectancy, Max DD Duration, and Annualization factor.

    Args:
        sp:        A :class:`SeriesPack` containing the derived series.
        rf_annual: Annual risk-free rate (decimal, e.g. 0.05 for 5%).

    Returns:
        List of ``{"Metric": ..., "Value": ...}`` dicts.
    """
    dates = pd.to_datetime(sp.dates)
    ann = annualize_factor_from_dates(dates)

    rets = sp.returns.copy()
    # Convert annual risk-free rate to per-period rate
    rf_period = (1.0 + rf_annual) ** (1.0 / ann) - 1.0
    ex = rets - rf_period

    eps = 1e-12
    mean_r = float(ex.mean())
    std_r = float(ex.std(ddof=1)) if len(ex) > 1 else 0.0

    # BUG-FIX: Sortino ratio -- downside deviation uses RMS of *negative*
    # excess returns (not std), which is the standard Sortino definition.
    negative_ex = ex[ex < 0.0]
    if len(negative_ex) > 0:
        downside_std = float(np.sqrt((negative_ex**2).mean()))
    else:
        downside_std = 0.0

    sharpe = (mean_r / (std_r + eps)) * math.sqrt(ann) if std_r > 0 else np.nan
    sortino = (mean_r / (downside_std + eps)) * math.sqrt(ann) if downside_std > 0 else np.nan

    max_dd = float(sp.drawdown.min()) if len(sp.drawdown) else np.nan
    total_pnl = float(sp.pnl.sum()) if len(sp.pnl) else 0.0

    init = float(sp.equity.iloc[0] - sp.pnl.iloc[0]) if len(sp.equity) else np.nan
    final_eq = float(sp.equity.iloc[-1]) if len(sp.equity) else np.nan

    # CAGR: compound annual growth rate over the observation period
    if init > 0 and final_eq > 0 and len(dates) > 1:
        years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
        cagr = (final_eq / init) ** (1.0 / max(years, 1e-9)) - 1.0 if years > 0.01 else np.nan
    else:
        cagr = np.nan

    calmar = (cagr / abs(max_dd)) if (np.isfinite(cagr) and np.isfinite(max_dd) and max_dd < 0) else np.nan

    hit_rate = float((sp.pnl > 0).mean()) if len(sp.pnl) else np.nan
    avg_win = float(sp.pnl[sp.pnl > 0].mean()) if (sp.pnl > 0).any() else np.nan
    avg_loss = float(sp.pnl[sp.pnl < 0].mean()) if (sp.pnl < 0).any() else np.nan

    sum_win = float(sp.pnl[sp.pnl > 0].sum()) if (sp.pnl > 0).any() else 0.0
    sum_loss = float(sp.pnl[sp.pnl < 0].sum()) if (sp.pnl < 0).any() else 0.0
    profit_factor = (sum_win / abs(sum_loss)) if sum_loss < 0 else np.nan

    # BUG-FIX: Volatility -- annualized from *raw* returns (not excess),
    # which is the standard definition of strategy volatility.
    raw_std = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    vol = raw_std * math.sqrt(ann) if raw_std > 0 else np.nan

    # Max drawdown duration: longest consecutive run where equity < HWM
    below = (sp.equity < sp.hwm).to_numpy()
    max_dur, cur = 0, 0
    for b in below:
        if b:
            cur += 1
            max_dur = max(max_dur, cur)
        else:
            cur = 0

    expectancy = float(sp.pnl.mean()) if len(sp.pnl) else np.nan

    # ---- Formatting helpers ----
    def fmt_pct(x: float) -> str:
        """Format a float as a percentage string, or em-dash if NaN."""
        return "\u2014" if (x is None or np.isnan(x)) else f"{x*100:,.2f}%"

    def fmt_num(x: float) -> str:
        """Format a float to 4 decimal places, or em-dash if NaN."""
        return "\u2014" if (x is None or np.isnan(x)) else f"{x:,.4f}"

    def fmt_cash(x: float) -> str:
        """Format a float as a cash value, or em-dash if NaN."""
        return "\u2014" if (x is None or np.isnan(x)) else f"{x:,.2f}"

    # Monthly return aggregation for summary stats
    eq_series = pd.Series(sp.equity.values, index=pd.to_datetime(sp.dates))
    monthly_ret = eq_series.resample("ME").last().pct_change(fill_method=None).dropna()

    rows = [
        {"Metric": "Total PnL", "Value": fmt_cash(total_pnl)},
        {"Metric": "Final Equity", "Value": fmt_cash(final_eq)},
        {"Metric": "CAGR", "Value": fmt_pct(cagr)},
        {"Metric": "Volatility", "Value": fmt_pct(vol)},
        {"Metric": "Sharpe", "Value": fmt_num(sharpe)},
        {"Metric": "Sortino", "Value": fmt_num(sortino)},
        {"Metric": "Max Drawdown", "Value": fmt_pct(max_dd)},
        {"Metric": "Calmar", "Value": fmt_num(calmar)},
        {"Metric": "Hit Rate", "Value": fmt_pct(hit_rate)},
        {"Metric": "Profit Factor", "Value": fmt_num(profit_factor)},
        {"Metric": "Avg Win", "Value": fmt_cash(avg_win)},
        {"Metric": "Avg Loss", "Value": fmt_cash(avg_loss)},
        {"Metric": "Best Day PnL", "Value": fmt_cash(float(sp.pnl.max()) if len(sp.pnl) else np.nan)},
        {"Metric": "Worst Day PnL", "Value": fmt_cash(float(sp.pnl.min()) if len(sp.pnl) else np.nan)},
        {"Metric": "Median Daily PnL", "Value": fmt_cash(float(sp.pnl.median()) if len(sp.pnl) else np.nan)},
        {"Metric": "Std Daily PnL", "Value": fmt_cash(float(sp.pnl.std(ddof=1)) if len(sp.pnl) > 1 else np.nan)},
        {"Metric": "Avg Monthly Return", "Value": fmt_pct(float(monthly_ret.mean()) if len(monthly_ret) else np.nan)},
        {
            "Metric": "Monthly Return Vol",
            "Value": fmt_pct(float(monthly_ret.std(ddof=1)) if len(monthly_ret) > 1 else np.nan),
        },
        {"Metric": "Skew (returns)", "Value": fmt_num(float(ex.skew()) if len(ex) else np.nan)},
        {"Metric": "Kurtosis (returns)", "Value": fmt_num(float(ex.kurt()) if len(ex) else np.nan)},
        {"Metric": "Expectancy (per day)", "Value": fmt_cash(expectancy)},
        {"Metric": "Max DD Duration (bars)", "Value": f"{max_dur:d}"},
        {"Metric": "Annualization", "Value": f"{ann:d}"},
    ]
    return rows
