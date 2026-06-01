"""Plotly figure factories for the trading strategy dashboard.

Each public function builds a fully-configured ``plotly.graph_objects.Figure``
ready to be rendered in a Dash ``dcc.Graph`` component.  Figures include:

* Equity curves (with optional drawdown overlay)
* Standalone drawdown charts
* Rolling pairwise correlation
* Rolling Sharpe ratio
* Monthly-return seasonality heatmap
* Placeholder / empty-state figures
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb

from dashboard.analytics import (
    SeriesPack,
    annualize_factor_from_dates,
    compute_series,
    padded_date_range,
)
from dashboard.config import BASE_BACKGROUND, BASE_FONT, BASE_HOVERLABEL, DEFAULT_INITIAL_CAPITAL
from dashboard.utils import format_product_label, valid_product_columns


# ---------------------------------------------------------------------------
# Shared layout helpers
# ---------------------------------------------------------------------------

def base_layout_kwargs(
    title: str, margin: dict, *, include_legend: bool = True, hovermode: str = "x unified"
) -> dict:
    """Return common Plotly layout keyword arguments.

    Args:
        title:          Figure title string.
        margin:         Dict of ``l``, ``r``, ``t``, ``b`` pixel margins.
        include_legend: Whether to include the horizontal legend bar.
        hovermode:      Plotly hover mode (default ``"x unified"``).

    Returns:
        Dict suitable for unpacking into ``fig.update_layout(**...)``.
    """
    layout: dict = {
        "template": "plotly_dark",
        "title": title,
        "margin": margin,
        "hovermode": hovermode,
        "dragmode": "zoom",
        "hoverlabel": BASE_HOVERLABEL,
        "paper_bgcolor": BASE_BACKGROUND,
        "plot_bgcolor": BASE_BACKGROUND,
        "font": BASE_FONT,
    }
    if include_legend:
        layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    return layout


def color_map(labels: List[str]) -> Dict[str, str]:
    """Map a list of labels to the Plotly qualitative color palette.

    Args:
        labels: Ordered list of trace labels.

    Returns:
        Dict mapping each label to a hex color string.
    """
    palette = px.colors.qualitative.Plotly
    return {label: palette[idx % len(palette)] for idx, label in enumerate(labels)}


def with_alpha(hex_color: str, alpha: float) -> str:
    """Convert a hex color to an ``rgba(...)`` string with the given alpha.

    Args:
        hex_color: CSS hex color (e.g. ``"#636EFA"``).
        alpha:     Opacity value between 0 and 1.

    Returns:
        RGBA color string.
    """
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


def decimate(x, y, max_points: int = 1000):
    """Downsample ``(x, y)`` for *display only*, preserving visual extremes.

    Plotting ~2,600+ daily points per SVG trace across several charts is the
    main source of UI lag (this browser has no WebGL fallback). Using min/max
    decimation -- keeping the smallest and largest sample of each contiguous
    bucket -- the point count drops to ~``max_points`` while spikes and troughs
    survive intact. Returns the inputs unchanged when already small enough.
    Never used for analytics/metrics -- only for plotted traces.

    Args:
        x:          Sequence of x values (dates).
        y:          Sequence of numeric y values.
        max_points: Approximate target number of plotted points (~2 per bucket).

    Returns:
        ``(x_ds, y_ds)`` reduced numpy arrays (or the originals if small).
    """
    y_arr = np.asarray(y, dtype="float64")
    n = y_arr.shape[0]
    if n <= max_points:
        return x, y
    x_arr = np.asarray(x)
    buckets = max(1, max_points // 2)
    keep: set[int] = {0, n - 1}  # always anchor the endpoints
    for bucket in np.array_split(np.arange(n), buckets):
        if bucket.size == 0:
            continue
        seg = y_arr[bucket]
        if np.all(np.isnan(seg)):
            keep.add(int(bucket[0]))
            continue
        keep.add(int(bucket[int(np.nanargmin(seg))]))
        keep.add(int(bucket[int(np.nanargmax(seg))]))
    idx = np.array(sorted(keep))
    return x_arr[idx], y_arr[idx]


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def equity_figure(
    series_by_label: Dict[str, SeriesPack],
    title: str,
    drawdown_series: Dict[str, SeriesPack] | None = None,
) -> go.Figure:
    """Build an equity-curve figure with optional drawdown overlay.

    When *drawdown_series* is provided, a secondary y-axis on the right
    shows the drawdown as a filled area chart.

    Args:
        series_by_label:  Mapping of display label to SeriesPack.
        title:            Figure title.
        drawdown_series:  Optional drawdown overlay (same keys as *series_by_label*).

    Returns:
        Configured Plotly ``Figure``.
    """
    fig = go.Figure()

    color_map_by_label = color_map(list(series_by_label.keys()))

    for label, sp in series_by_label.items():
        color = color_map_by_label[label]
        eq_x, eq_y = decimate(sp.dates, sp.equity)
        hwm_x, hwm_y = decimate(sp.dates, sp.hwm)
        fig.add_trace(
            go.Scatter(
                x=eq_x,
                y=eq_y,
                mode="lines",
                name=f"{label} Equity",
                hovertemplate="Equity: %{y:,.2f}<extra></extra>",
                line=dict(color=color),
                hoverlabel=dict(bgcolor=color, bordercolor=color),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hwm_x,
                y=hwm_y,
                mode="lines",
                name=f"{label} High Watermark",
                hovertemplate="HWM: %{y:,.2f}<extra></extra>",
                line=dict(width=1, dash="dot", color=color),
                hoverlabel=dict(bgcolor=color, bordercolor=color),
            )
        )

    # Optional drawdown overlay on secondary y-axis
    if drawdown_series:
        for label, sp in drawdown_series.items():
            color = color_map_by_label.get(label) or color_map([label])[label]
            dd_x, dd_y = decimate(sp.dates, sp.drawdown)
            fig.add_trace(
                go.Scatter(
                    x=dd_x,
                    y=dd_y,
                    mode="lines",
                    name=f"{label} Drawdown",
                    fill="tozeroy",
                    hovertemplate="DD: %{y:.2%}<extra></extra>",
                    line=dict(color=color, width=1),
                    fillcolor=with_alpha(color, 0.16),
                    hoverlabel=dict(bgcolor=color, bordercolor=color),
                    yaxis="y2",
                )
            )

    fig.update_layout(**base_layout_kwargs(title, margin=dict(l=14, r=14, t=40, b=22)))
    fig.update_xaxes(showgrid=False, showspikes=False, hoverformat="%Y-%m-%d")
    fig.update_yaxes(title="Equity", tickformat=",.0f")
    if drawdown_series:
        fig.update_layout(
            yaxis2=dict(
                title="Drawdown",
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=False,
                tickformat=".1%",
            )
        )
    return fig


# ---------------------------------------------------------------------------
# Drawdown chart
# ---------------------------------------------------------------------------

def drawdown_figure(series_by_label: Dict[str, SeriesPack], title: str) -> go.Figure:
    """Build a standalone drawdown area chart.

    Args:
        series_by_label: Mapping of display label to SeriesPack.
        title:           Figure title.

    Returns:
        Configured Plotly ``Figure``.
    """
    fig = go.Figure()

    color_map_by_label = color_map(list(series_by_label.keys()))

    for label, sp in series_by_label.items():
        color = color_map_by_label[label]
        dd_x, dd_y = decimate(sp.dates, sp.drawdown)
        fig.add_trace(
            go.Scatter(
                x=dd_x,
                y=dd_y,
                mode="lines",
                name=f"{label} Drawdown",
                fill="tozeroy",
                hovertemplate="DD: %{y:.2%}<extra></extra>",
                line=dict(color=color),
                fillcolor=with_alpha(color, 0.2),
                hoverlabel=dict(bgcolor=color, bordercolor=color),
            )
        )

    fig.update_layout(
        **base_layout_kwargs(title, margin=dict(l=14, r=14, t=40, b=22)),
        yaxis=dict(title="Drawdown", tickformat=".1%"),
    )
    fig.update_xaxes(showgrid=False, showspikes=False, hoverformat="%Y-%m-%d")
    return fig


# ---------------------------------------------------------------------------
# Rolling correlation
# ---------------------------------------------------------------------------

def rolling_correlation_figure(
    df: pd.DataFrame, products: List[str], initial_capital: float, window: int, title: str
) -> go.Figure:
    """Build a rolling pairwise correlation chart for selected products.

    Requires at least two products; returns an empty figure otherwise.

    Args:
        df:              DataFrame with ``"date"`` and PnL columns.
        products:        Product columns to correlate.
        initial_capital: Starting capital (used for per-product series).
        window:          Rolling window size (in bars).
        title:           Figure title.

    Returns:
        Configured Plotly ``Figure``.
    """
    fig = go.Figure()

    if len(products) < 2:
        fig.update_layout(
            **base_layout_kwargs(title, margin=dict(l=18, r=18, t=48, b=56)),
        )
        return fig

    # Compute per-product PnL series for correlation
    returns_by_product = {}
    for p in products:
        sp = compute_series(df, [p], initial_capital)
        returns_by_product[p] = sp.pnl

    returns_df = pd.DataFrame(returns_by_product)
    dates = pd.to_datetime(df["date"])

    # Build labels for each unique pair
    corr_labels = []
    for i, p1 in enumerate(products):
        for p2 in products[i + 1 :]:
            corr_labels.append(f"{format_product_label(p1)} vs {format_product_label(p2)}")

    color_map_by_label = color_map(corr_labels)

    for i, p1 in enumerate(products):
        for p2 in products[i + 1 :]:
            roll_corr = returns_df[p1].rolling(window).corr(returns_df[p2])
            label = f"{format_product_label(p1)} vs {format_product_label(p2)}"
            # Many pairs (e.g. 15 for a 6-leg portfolio) -> decimate harder to keep
            # the payload/render light.
            corr_x, corr_y = decimate(dates, roll_corr, max_points=500)
            fig.add_trace(
                go.Scatter(
                    x=corr_x,
                    y=corr_y,
                    mode="lines",
                    name=label,
                    hovertemplate="Roll Corr: %{y:.2f}<extra></extra>",
                    line=dict(color=color_map_by_label[label]),
                    hoverlabel=dict(bgcolor=color_map_by_label[label], bordercolor=color_map_by_label[label]),
                )
            )

    fig.update_layout(
        **base_layout_kwargs(title, margin=dict(l=18, r=18, t=48, b=56)),
    )
    padded_range = padded_date_range(df["date"])
    fig.update_yaxes(title="Rolling Corr", range=[-1, 1])
    xaxis_kwargs: dict = dict(showgrid=False, showspikes=False, hoverformat="%Y-%m-%d")
    if padded_range is not None:
        xaxis_kwargs["range"] = padded_range
    fig.update_xaxes(**xaxis_kwargs)
    return fig


# ---------------------------------------------------------------------------
# Rolling Sharpe
# ---------------------------------------------------------------------------

def rolling_sharpe_figure(
    df: pd.DataFrame,
    products: List[str],
    initial_capital: float,
    window: int,
    title: str,
    *,
    include_individuals: bool = True,
    include_aggregate: bool = True,
) -> go.Figure:
    """Build a rolling Sharpe ratio chart.

    Can display individual product Sharpe traces, an aggregate trace, or both.

    Args:
        df:                  DataFrame with ``"date"`` and PnL columns.
        products:            Product columns to include.
        initial_capital:     Starting capital for series computation.
        window:              Rolling window size (in bars).
        title:               Figure title.
        include_individuals: Show one trace per product.
        include_aggregate:   Show an aggregate ("ALL") trace when >1 product.

    Returns:
        Configured Plotly ``Figure``.
    """
    fig = go.Figure()
    ann = annualize_factor_from_dates(df["date"])

    # Determine trace ordering for consistent color assignment
    label_order: List[str] = []
    if include_individuals:
        label_order.extend([format_product_label(p) for p in products])
    if include_aggregate and len(products) > 1:
        label_order.append("ALL (agg)")

    color_map_by_label = color_map(label_order)

    # Individual product rolling Sharpe
    if include_individuals:
        for p in products:
            sp = compute_series(df, [p], initial_capital)
            r = sp.pnl
            roll = (r.rolling(window).mean() / (r.rolling(window).std(ddof=1) + 1e-12)) * math.sqrt(ann)
            roll_x, roll_y = decimate(sp.dates, roll)
            fig.add_trace(
                go.Scatter(
                    x=roll_x,
                    y=roll_y,
                    mode="lines",
                    name=format_product_label(p),
                    hovertemplate="Roll Sharpe: %{y:.2f}<extra></extra>",
                    line=dict(color=color_map_by_label[format_product_label(p)]),
                    hoverlabel=dict(
                        bgcolor=color_map_by_label[format_product_label(p)],
                        bordercolor=color_map_by_label[format_product_label(p)],
                    ),
                )
            )

    # Aggregate rolling Sharpe (all products combined)
    if include_aggregate and len(products) > 1:
        sp_all = compute_series(df, products, initial_capital)
        r = sp_all.pnl
        roll = (r.rolling(window).mean() / (r.rolling(window).std(ddof=1) + 1e-12)) * math.sqrt(ann)
        # Guard against a missing key returning None (invalid color for Plotly).
        agg_color = color_map_by_label.get("ALL (agg)") or color_map(["ALL (agg)"])["ALL (agg)"]
        agg_x, agg_y = decimate(sp_all.dates, roll)
        fig.add_trace(
            go.Scatter(
                x=agg_x,
                y=agg_y,
                mode="lines",
                name="ALL (agg)",
                line=dict(width=2, color=agg_color),
                hovertemplate="Roll Sharpe: %{y:.2f}<extra></extra>",
                hoverlabel=dict(bgcolor=agg_color, bordercolor=agg_color),
            )
        )

    fig.update_layout(
        **base_layout_kwargs(title, margin=dict(l=18, r=18, t=48, b=56)),
    )
    padded_range = padded_date_range(df["date"])
    xaxis_kwargs: dict = dict(showgrid=False, showspikes=False, hoverformat="%Y-%m-%d")
    if padded_range is not None:
        xaxis_kwargs["range"] = padded_range
    fig.update_xaxes(**xaxis_kwargs)
    fig.update_yaxes(title="Sharpe")
    return fig


# ---------------------------------------------------------------------------
# Seasonality heatmap
# ---------------------------------------------------------------------------

def seasonality_figure(df: pd.DataFrame, products: List[str], title: str) -> go.Figure:
    """Build a monthly-return seasonality heatmap.

    Aggregates PnL across *products*, constructs an equity curve, then
    computes month-over-month returns pivoted by year x month.

    Args:
        df:       DataFrame with ``"date"`` and PnL columns.
        products: Product columns to include.
        title:    Figure title.

    Returns:
        Configured Plotly ``Figure`` with a year x month heatmap.
    """
    valid_products = valid_product_columns(df, products) or valid_product_columns(df)

    tmp = df[["date"] + valid_products].copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    pnl = tmp[valid_products].sum(axis=1)
    equity = DEFAULT_INITIAL_CAPITAL + pnl.cumsum()
    eq_series = pd.Series(equity.values, index=tmp["date"])

    # Proper month-over-month return: each month's ending equity relative to
    # the prior month's ending equity.  The first month is seeded against the
    # initial capital so it is not dropped.  Using prior-month close (rather
    # than the first value inside the month) is the standard return definition.
    month_end_eq = eq_series.resample("ME").last()
    prior_eq = month_end_eq.shift(1)
    if len(prior_eq) > 0:
        prior_eq.iloc[0] = DEFAULT_INITIAL_CAPITAL
    monthly_returns = (month_end_eq - prior_eq) / prior_eq.abs()

    heatmap_df = monthly_returns.to_frame(name="ret")
    heatmap_df["year"] = heatmap_df.index.year
    heatmap_df["month"] = heatmap_df.index.month
    pivot = heatmap_df.pivot(index="year", columns="month", values="ret").sort_index()
    pivot = pivot.reindex(columns=range(1, 13))

    # Robust symmetric color limits using 10th/90th percentiles
    z_values = pivot.values.flatten()
    z_values = z_values[~np.isnan(z_values)]

    if len(z_values) > 0:
        z_low, z_high = np.percentile(z_values, [10, 90])
        z_abs = max(abs(z_low), abs(z_high))  # keep symmetry around 0
    else:
        z_abs = 0.1

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            y=pivot.index,
            colorscale=[[0, "#6b0f1a"], [0.5, "#1b1f2a"], [1.0, "#2ed47a"]],
            zmid=0,
            zmin=-z_abs,
            zmax=z_abs,
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>",
            colorbar=dict(
                thickness=8,
                len=0.82,
                y=0.5,
                yanchor="middle",
                x=1.02,
                xanchor="left",
                tickfont=dict(color="#e6e6e6", size=10),
                outlinewidth=0,
            ),
        )
    )
    fig.update_layout(
        **base_layout_kwargs(title, margin=dict(l=28, r=28, t=40, b=40), hovermode="closest"),
    )
    fig.update_xaxes(title=None, constrain="domain", automargin=True)
    fig.update_yaxes(title=None, scaleanchor="x", scaleratio=1, automargin=True)
    return fig


# ---------------------------------------------------------------------------
# Placeholder / empty state
# ---------------------------------------------------------------------------

def placeholder_figure(title: str, subtitle: str | None = None) -> go.Figure:
    """Create a blank figure with a centered message.

    Used when no data is available or a panel has nothing to display.

    Args:
        title:    Primary message text (rendered bold).
        subtitle: Optional smaller secondary text.

    Returns:
        Configured Plotly ``Figure``.
    """
    fig = go.Figure()
    subtitle_html = f"<br><span style='font-size:12px; color:#9ca8b8;'>{subtitle}</span>" if subtitle else ""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_visible=False,
        yaxis_visible=False,
        annotations=[
            dict(
                text=f"<b>{title}</b>{subtitle_html}",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="center",
                font=dict(color="#e6e6e6", size=14),
            )
        ],
        margin=dict(l=14, r=14, t=40, b=22),
    )
    return fig
