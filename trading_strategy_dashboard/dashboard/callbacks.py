"""Dash callback registration for the trading strategy dashboard.

Contains all interactive logic: sidebar toggles, strategy/product selection,
theme switching, panel visibility, range cycling, modal open/close,
CSV explorer, and the main ``update_dashboard`` callback that rebuilds
all figures and metric tables in response to user input.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import dash
from dash import Input, Output, State, callback_context, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from dashboard.analytics import (
    compute_metrics,
    compute_series,
    filter_df_by_range,
    next_range_key,
    range_cycle_label,
)
from dashboard.config import BASE_FONT, DEFAULT_RF, DEFAULT_ROLL_WINDOW, PANEL_KEYS
from dashboard.table_styles import metrics_cell_style, metrics_header_style
from dashboard.data import portfolio_dataframe, products_for_strategy
from dashboard.figures import (
    correlation_matrix_figure,
    drawdown_figure,
    equity_figure,
    placeholder_figure,
    rolling_correlation_figure,
    rolling_sharpe_figure,
    seasonality_figure,
)
from dashboard.utils import format_product_label
from dashboard.var_callbacks import register_var_callbacks
from dashboard.var_scaling import (
    compute_var_scaled_frame,
    portfolio_effective_dataframe,
    validate_var_config,
)


# ---------------------------------------------------------------------------
# Metrics-table builders
# ---------------------------------------------------------------------------

def _build_metric_rows(
    df: pd.DataFrame,
    display_columns: List[Tuple[str, List[str]]],
    empty_label: str,
) -> Tuple[List[dict], List[dict]]:
    """Shared core for the metrics DataTables.

    Computes one metrics column per ``(label, columns)`` entry and pivots them
    into row-per-metric form.  When *display_columns* is empty (nothing
    selected), returns a single informational row instead of a silent blank.

    Args:
        df:              Source DataFrame with ``"date"`` and value columns.
        display_columns: Ordered ``(column_label, source_columns)`` pairs.
        empty_label:     Message shown when no columns are selected.

    Returns:
        ``(columns, rows)`` tuple suitable for a ``dash_table.DataTable``.
    """
    if not display_columns:
        return (
            [{"name": "Metric", "id": "Metric"}, {"name": "", "id": "Value"}],
            [{"Metric": empty_label, "Value": "—"}],
        )

    columns = [{"name": "Metric", "id": "Metric"}] + [
        {"name": label, "id": label} for label, _ in display_columns
    ]

    metrics_by_column: Dict[str, List[dict]] = {}
    for label, source_columns in display_columns:
        sp = compute_series(df, source_columns)
        metrics_by_column[label] = compute_metrics(sp, rf_annual=DEFAULT_RF)

    metric_order = [row["Metric"] for row in next(iter(metrics_by_column.values()), [])]
    rows: List[dict] = []
    for metric in metric_order:
        row = {"Metric": metric}
        for label, metrics in metrics_by_column.items():
            row[label] = next((r["Value"] for r in metrics if r["Metric"] == metric), "—")
        rows.append(row)

    return columns, rows


def build_metrics_table(
    df: pd.DataFrame,
    selected_products: List[str],
    all_products: List[str],
) -> Tuple[List[dict], List[dict]]:
    """Build columns and row data for the per-product metrics DataTable.

    When ``"ALL"`` is in *selected_products*, a single aggregate column is
    shown; otherwise one column per selected product.

    Args:
        df:                Source DataFrame with ``"date"`` and PnL columns.
        selected_products: Currently selected product keys (may include ``"ALL"``).
        all_products:      Full list of available product column names.

    Returns:
        ``(columns, rows)`` tuple suitable for a ``dash_table.DataTable``.
    """
    if "ALL" in selected_products:
        display_columns = [("All", all_products)]
    else:
        display_columns = [(format_product_label(p), [p]) for p in selected_products if p in all_products]

    return _build_metric_rows(df, display_columns, empty_label="No products selected")


def build_portfolio_metrics_table(
    selected_strategies: List[str],
    all_strategies: List[str],
    portfolio_df: pd.DataFrame,
) -> Tuple[List[dict], List[dict]]:
    """Build columns and row data for the portfolio-level metrics DataTable.

    Similar to :func:`build_metrics_table` but operates on strategy-level
    aggregation rather than individual products.

    Args:
        selected_strategies: Currently selected strategy keys (may include ``"ALL"``).
        all_strategies:      Full list of available strategy names.
        portfolio_df:        Merged portfolio DataFrame from :func:`portfolio_dataframe`.

    Returns:
        ``(columns, rows)`` tuple suitable for a ``dash_table.DataTable``.
    """
    if "ALL" in selected_strategies:
        display_columns = [("All Strategies", all_strategies)]
    else:
        display_columns = [(s, [s]) for s in selected_strategies if s in all_strategies]

    return _build_metric_rows(portfolio_df, display_columns, empty_label="No strategies selected")


def register_callbacks(
    app: dash.Dash,
    strategies: Dict[str, pd.DataFrame],
    returns: Dict[str, pd.DataFrame] | None = None,
) -> None:
    """Register all Dash callbacks on the given app instance.

    This is the central wiring function that connects every UI control to
    its corresponding update logic.  It defines inner helper functions and
    roughly 20 ``@app.callback`` decorated functions covering:

    * Sidebar and settings-panel toggles
    * Theme selection and button styling
    * Panel visibility and layout mode
    * Date-range cycling (per-panel and global)
    * Modal open/close for equity, drawdown, metrics, custom analytics, CSV
    * Strategy and product selection
    * The main ``update_dashboard`` callback that recomputes all figures

    VaR-scaling callbacks are registered separately via
    :func:`dashboard.var_callbacks.register_var_callbacks`.

    Args:
        app:        The Dash application instance.
        strategies: Mapping of strategy name to its loaded PnL DataFrame.
        returns:    Optional mapping of strategy name to its daily-returns
                    DataFrame (for VaR scaling). Absent strategies simply have
                    VaR scaling unavailable.
    """
    strategy_names = list(strategies.keys())
    returns = returns or {}

    def default_strategy_name() -> str:
        """Return the default strategy key shown on initial load."""
        return "Portfolio"

    def product_button(label: str, value: str) -> dbc.Button:
        """Create a product-filter pill button (callback-internal version).

        Args:
            label: Display text.
            value: Internal value for pattern-matching callbacks.

        Returns:
            Configured ``dbc.Button``.
        """
        return dbc.Button(
            label,
            id={"type": "product-btn", "value": value},
            color="secondary",
            outline=True,
            className="product-pill",
            n_clicks=0,
            active=False,
        )

    # ----- Theme configuration: Plotly color palettes per theme key -----
    theme_plotly = {
        "deep": {
            "font_color": "#e6e9f0",
            "axis_color": "#cfd6e4",
            "grid_color": "rgba(92, 110, 140, 0.25)",
            "hover_bg": "#151c2c",
            "hover_border": "#6f7c95",
            "template": "plotly_dark",
        },
        "light": {
            "font_color": "#1f2a3a",
            "axis_color": "#4a5a72",
            "grid_color": "rgba(150, 168, 196, 0.35)",
            "hover_bg": "#eef2f8",
            "hover_border": "#9aaac2",
            "template": "plotly_white",
        },
    }

    def apply_plotly_theme(fig: go.Figure, theme: str | None) -> None:
        """Apply the selected colour theme to a Plotly figure in-place.

        Updates font colours, grid colours, hover label styling, and
        annotation/colorbar text to match the chosen theme.

        Args:
            fig:   The Plotly figure to mutate.
            theme: Theme key (e.g. ``"deep"``, ``"light"``); defaults to ``"deep"``.
        """
        theme_key = theme or "deep"
        theme_values = theme_plotly.get(theme_key, theme_plotly["deep"])
        font_color = theme_values["font_color"]
        axis_color = theme_values["axis_color"]
        grid_color = theme_values["grid_color"]
        fig.update_layout(
            template=theme_values["template"],
            font=dict(color=font_color, family=BASE_FONT["family"]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hoverlabel=dict(
                bgcolor=theme_values["hover_bg"],
                bordercolor=theme_values["hover_border"],
                font=dict(color=font_color, family=BASE_FONT["family"], size=12),
            ),
            legend=dict(font=dict(color=font_color, family=BASE_FONT["family"])),
        )
        fig.update_xaxes(color=axis_color, gridcolor=grid_color, zerolinecolor=grid_color)
        fig.update_yaxes(color=axis_color, gridcolor=grid_color, zerolinecolor=grid_color)
        if fig.layout.annotations:
            for annotation in fig.layout.annotations:
                annotation.font = dict(color=font_color, family=BASE_FONT["family"], size=14)
        for trace in fig.data:
            if getattr(trace, "colorbar", None):
                trace.colorbar.tickfont = dict(color=font_color, family=BASE_FONT["family"], size=10)

    # ================================================================
    # Sidebar toggles
    # ================================================================

    @app.callback(
        Output("sidebar", "is_open"),
        Input("btn-open-sidebar", "n_clicks"),
        Input("store-selected-strategy", "data"),
        State("sidebar", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_sidebar(n, selected_strategy, is_open):
        """Open/close the navigation sidebar; auto-close on strategy change."""
        ctx = callback_context
        if not ctx.triggered:
            return is_open

        trig_id = ctx.triggered_id
        if trig_id == "btn-open-sidebar":
            return not is_open

        if trig_id == "store-selected-strategy":
            return False

        return is_open

    @app.callback(
        Output("settings-sidebar", "is_open"),
        Input("btn-open-settings", "n_clicks"),
        Input("btn-reset-layout", "n_clicks"),
        Input({"type": "csv-open-btn", "value": dash.ALL}, "n_clicks"),
        State("settings-sidebar", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_settings_sidebar(open_clicks, reset_clicks, csv_clicks, is_open):
        """Toggle the settings sidebar; auto-close on reset or CSV open."""
        ctx = callback_context
        if not ctx.triggered:
            return is_open

        trig_id = ctx.triggered_id
        if trig_id == "btn-open-settings":
            return not is_open

        if trig_id == "btn-reset-layout":
            return False

        if isinstance(trig_id, dict) and trig_id.get("type") == "csv-open-btn":
            return False

        return is_open

    # ================================================================
    # Root CSS class management (theme + dimming overlays)
    # ================================================================

    @app.callback(
        Output("app-root", "className"),
        Input("sidebar", "is_open"),
        Input("settings-sidebar", "is_open"),
        Input("store-equity-modal-open", "data"),
        Input("store-drawdown-modal-open", "data"),
        Input("store-metrics-modal-open", "data"),
        Input("store-custom-analytics-modal-open", "data"),
        Input("store-csv-modal-open", "data"),
        Input("store-theme", "data"),
        Input("store-var-modal-open", "data"),
        Input("store-var-expand-open", "data"),
    )
    def dim_background(
        is_open,
        settings_open,
        equity_modal_open,
        drawdown_modal_open,
        metrics_modal_open,
        custom_analytics_open,
        csv_modal_open,
        theme,
        var_modal_open,
        var_expand_open,
    ):
        """Build the root element CSS class list based on active state."""
        base_class = "app-root"
        classes = [base_class]
        theme_class = f"theme-{theme or 'deep'}"
        classes.append(theme_class)
        if is_open or settings_open:
            classes.append("sidebar-visible")
        if equity_modal_open:
            classes.append("equity-modal-visible")
        if drawdown_modal_open:
            classes.append("drawdown-modal-visible")
        if metrics_modal_open:
            classes.append("metrics-modal-visible")
        if custom_analytics_open:
            classes.append("custom-analytics-visible")
        if csv_modal_open:
            classes.append("csv-modal-visible")
        if var_modal_open:
            classes.append("var-modal-visible")
        if var_expand_open:
            classes.append("var-expand-visible")
        return " ".join(classes)

    # ================================================================
    # Theme selection
    # ================================================================

    @app.callback(
        Output("store-theme", "data"),
        Input({"type": "theme-btn", "value": dash.ALL}, "n_clicks"),
        State({"type": "theme-btn", "value": dash.ALL}, "id"),
        State("store-theme", "data"),
        prevent_initial_call=True,
    )
    def set_theme(n_clicks_list, ids, current_theme):
        """Persist the selected theme key into the store."""
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
        trig_id = ctx.triggered_id
        if isinstance(trig_id, dict) and trig_id.get("type") == "theme-btn":
            return trig_id.get("value") or current_theme or "deep"
        return dash.no_update

    @app.callback(
        Output({"type": "theme-btn", "value": dash.ALL}, "className"),
        Output({"type": "theme-btn", "value": dash.ALL}, "outline"),
        Input("store-theme", "data"),
        State({"type": "theme-btn", "value": dash.ALL}, "id"),
    )
    def update_theme_button_styles(theme, ids):
        """Highlight the active theme button and dim the others."""
        if not ids:
            return [], []
        current = theme or "deep"
        classes = []
        outlines = []
        for component_id in ids:
            is_active = component_id.get("value") == current
            classes.append("settings-option-btn active" if is_active else "settings-option-btn")
            outlines.append(not is_active)
        return classes, outlines

    @app.callback(
        Output("metrics-table", "style_cell"),
        Output("metrics-table", "style_header"),
        Output("metrics-modal-table", "style_cell"),
        Output("metrics-modal-table", "style_header"),
        Output("csv-modal-table", "style_cell"),
        Output("csv-modal-table", "style_header"),
        Input("store-theme", "data"),
    )
    def theme_table_styles(theme):
        """Recolour DataTable text/headers per theme.

        DataTable inline styles beat CSS, so light-mode legibility is driven here
        rather than via fragile ``!important`` overrides. Dark output is identical
        to the original inline dicts.
        """
        return (
            metrics_cell_style(theme),
            metrics_header_style(theme),
            metrics_cell_style(theme, padding="12px 14px", min_width="140px"),
            metrics_header_style(theme, header_bg_dark="rgba(255,255,255,0.05)"),
            metrics_cell_style(theme),
            metrics_header_style(theme),
        )

    # ================================================================
    # Panel visibility and layout mode
    # ================================================================

    @app.callback(
        Output("panel-visibility", "value"),
        Output("store-last-deselected", "data"),
        Input("panel-visibility", "value"),
        Input("btn-reset-layout", "n_clicks"),
        State("store-last-deselected", "data"),
        prevent_initial_call=True,
    )
    def enforce_panel_visibility(visible_panels, reset_clicks, last_deselected):
        """Ensure at least 3 of 4 panels remain visible at all times."""
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update, last_deselected

        trig_id = ctx.triggered_id
        if trig_id == "btn-reset-layout":
            return PANEL_KEYS.copy(), None

        visible = visible_panels or []
        visible_set = set(visible)
        missing = [panel for panel in PANEL_KEYS if panel not in visible_set]

        if len(missing) <= 1:
            return visible, missing[0] if missing else None

        if last_deselected in missing:
            newly_deselected = next(panel for panel in missing if panel != last_deselected)
        else:
            newly_deselected = missing[-1]

        corrected_visible = [panel for panel in PANEL_KEYS if panel != newly_deselected]
        return corrected_visible, newly_deselected

    @app.callback(
        Output("body-grid", "className"),
        Input("layout-radio", "value"),
        Input("panel-visibility", "value"),
    )
    def update_layout_class(layout_value, visible_panels):
        """Compute the CSS grid class based on layout mode and hidden panels."""
        layout_value = layout_value or "default"
        if layout_value == "focused":
            return "body-grid layout-focused"
        if layout_value != "default":
            return f"body-grid layout-{layout_value}"

        visible_set = set(visible_panels or PANEL_KEYS)
        hidden = [panel for panel in PANEL_KEYS if panel not in visible_set]
        classes = [f"body-grid layout-{layout_value}"]
        if len(hidden) == 1:
            classes.append(f"panel-hidden-{hidden[0]}")
        return " ".join(classes)

    @app.callback(
        Output("layout-radio", "value"),
        Input("btn-reset-layout", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_layout_option(n_clicks):
        """Reset layout radio to default when the reset button is pressed."""
        if not n_clicks:
            return dash.no_update
        return "default"

    @app.callback(
        Output("panel-equity", "style"),
        Output("panel-custom", "style"),
        Output("panel-drawdown", "style"),
        Output("panel-metrics", "style"),
        Input("panel-visibility", "value"),
        Input("layout-radio", "value"),
    )
    def update_panel_visibility(visible_panels, layout_value):
        """Set display style on each panel based on visibility checklist and layout."""
        visible_set = set(visible_panels or [])
        if layout_value == "focused":
            visible_set = {"equity", "metrics"}
        elif layout_value != "default":
            visible_set = set(PANEL_KEYS)

        def style_for(panel_key: str) -> dict:
            return {} if panel_key in visible_set else {"display": "none"}

        return (
            style_for("equity"),
            style_for("custom"),
            style_for("drawdown"),
            style_for("metrics"),
        )

    # ================================================================
    # Date-range cycling
    # ================================================================

    @app.callback(
        Output("store-equity-range", "data"),
        Output("equity-range-btn", "children"),
        Output("store-drawdown-range", "data"),
        Output("drawdown-range-btn", "children"),
        Output("store-metrics-range", "data"),
        Output("metrics-range-btn", "children"),
        Output("store-custom-range", "data"),
        Output("custom-range-btn", "children"),
        Output("btn-cycle-range", "children"),
        Input("equity-range-btn", "n_clicks"),
        Input("drawdown-range-btn", "n_clicks"),
        Input("metrics-range-btn", "n_clicks"),
        Input("custom-range-btn", "n_clicks"),
        Input("btn-cycle-range", "n_clicks"),
        Input("btn-reset-layout", "n_clicks"),
        State("store-equity-range", "data"),
        State("store-drawdown-range", "data"),
        State("store-metrics-range", "data"),
        State("store-custom-range", "data"),
        prevent_initial_call=True,
    )
    def update_range_controls(
        equity_clicks,
        drawdown_clicks,
        metrics_clicks,
        custom_clicks,
        cycle_clicks,
        reset_clicks,
        equity_range,
        drawdown_range,
        metrics_range,
        custom_range,
    ):
        """Cycle individual or all panel date ranges and update button labels."""
        ctx = callback_context
        if not ctx.triggered:
            cycle_label = range_cycle_label(equity_range, drawdown_range, metrics_range, custom_range)
            return (
                equity_range or "All",
                equity_range or "All",
                drawdown_range or "All",
                drawdown_range or "All",
                metrics_range or "All",
                metrics_range or "All",
                custom_range or "All",
                custom_range or "All",
                cycle_label,
            )

        trig_id = ctx.triggered_id
        if trig_id == "btn-reset-layout":
            return ("All", "All", "All", "All", "All", "All", "All", "All", "All Panels: All")

        # Only write the store(s) that actually changed -- returning ``no_update``
        # for the rest prevents Dash from re-firing every panel's figure callback
        # when a single panel's range is cycled (the main source of UI lag).
        nu = dash.no_update
        if trig_id == "equity-range-btn":
            equity_range = next_range_key(equity_range)
            label = range_cycle_label(equity_range, drawdown_range, metrics_range, custom_range)
            return (equity_range, equity_range, nu, nu, nu, nu, nu, nu, label)
        if trig_id == "drawdown-range-btn":
            drawdown_range = next_range_key(drawdown_range)
            label = range_cycle_label(equity_range, drawdown_range, metrics_range, custom_range)
            return (nu, nu, drawdown_range, drawdown_range, nu, nu, nu, nu, label)
        if trig_id == "metrics-range-btn":
            metrics_range = next_range_key(metrics_range)
            label = range_cycle_label(equity_range, drawdown_range, metrics_range, custom_range)
            return (nu, nu, nu, nu, metrics_range, metrics_range, nu, nu, label)
        if trig_id == "custom-range-btn":
            custom_range = next_range_key(custom_range)
            label = range_cycle_label(equity_range, drawdown_range, metrics_range, custom_range)
            return (nu, nu, nu, nu, nu, nu, custom_range, custom_range, label)
        if trig_id == "btn-cycle-range":
            r = next_range_key(equity_range)
            label = range_cycle_label(r, r, r, r)
            return (r, r, r, r, r, r, r, r, label)

        return (nu, nu, nu, nu, nu, nu, nu, nu, nu)

    # ================================================================
    # Modal open / close callbacks
    # ================================================================

    @app.callback(
        Output("store-equity-modal-open", "data"),
        Input("open-equity-modal", "n_clicks"),
        Input("close-equity-modal", "n_clicks"),
        Input("equity-modal-backdrop", "n_clicks"),
        State("store-equity-modal-open", "data"),
    )
    def toggle_equity_modal(open_clicks, close_clicks, backdrop_clicks, is_open):
        """Toggle the full-screen equity modal."""
        ctx = callback_context
        if not ctx.triggered:
            return is_open or False

        trig_id = ctx.triggered_id
        if trig_id == "open-equity-modal":
            return True
        if trig_id in ("close-equity-modal", "equity-modal-backdrop"):
            return False

        return is_open or False

    @app.callback(
        Output("store-drawdown-modal-open", "data"),
        Input("open-drawdown-modal", "n_clicks"),
        Input("close-drawdown-modal", "n_clicks"),
        Input("drawdown-modal-backdrop", "n_clicks"),
        State("store-drawdown-modal-open", "data"),
    )
    def toggle_drawdown_modal(open_clicks, close_clicks, backdrop_clicks, is_open):
        """Toggle the full-screen drawdown modal."""
        ctx = callback_context
        if not ctx.triggered:
            return is_open or False

        trig_id = ctx.triggered_id
        if trig_id == "open-drawdown-modal":
            return True
        if trig_id in ("close-drawdown-modal", "drawdown-modal-backdrop"):
            return False

        return is_open or False

    @app.callback(
        Output("store-metrics-modal-open", "data"),
        Input("open-metrics-modal", "n_clicks"),
        Input("close-metrics-modal", "n_clicks"),
        Input("metrics-modal-backdrop", "n_clicks"),
        State("store-metrics-modal-open", "data"),
    )
    def toggle_metrics_modal(open_clicks, close_clicks, backdrop_clicks, is_open):
        """Toggle the full-screen metrics modal."""
        ctx = callback_context
        if not ctx.triggered:
            return is_open or False

        trig_id = ctx.triggered_id
        if trig_id == "open-metrics-modal":
            return True
        if trig_id in ("close-metrics-modal", "metrics-modal-backdrop"):
            return False

        return is_open or False

    @app.callback(
        Output("store-custom-analytics-modal-open", "data"),
        Input("open-custom-analytics-modal", "n_clicks"),
        Input("close-custom-analytics-modal", "n_clicks"),
        Input("custom-analytics-modal-backdrop", "n_clicks"),
        State("store-custom-analytics-modal-open", "data"),
    )
    def toggle_custom_analytics_modal(open_clicks, close_clicks, backdrop_clicks, is_open):
        """Toggle the full-screen custom-analytics modal."""
        ctx = callback_context
        if not ctx.triggered:
            return is_open or False

        trig_id = ctx.triggered_id
        if trig_id == "open-custom-analytics-modal":
            return True
        if trig_id in ("close-custom-analytics-modal", "custom-analytics-modal-backdrop"):
            return False

        return is_open or False

    # ================================================================
    # Modal overlay CSS class callbacks
    # ================================================================

    @app.callback(
        Output("equity-modal-overlay", "className"),
        Input("store-equity-modal-open", "data"),
    )
    def set_equity_modal_class(is_open):
        """Add/remove 'open' class on the equity modal overlay."""
        base = "equity-modal-overlay"
        return f"{base} open" if is_open else base

    @app.callback(
        Output("drawdown-modal-overlay", "className"),
        Input("store-drawdown-modal-open", "data"),
    )
    def set_drawdown_modal_class(is_open):
        """Add/remove 'open' class on the drawdown modal overlay."""
        base = "drawdown-modal-overlay"
        return f"{base} open" if is_open else base

    @app.callback(
        Output("metrics-modal-overlay", "className"),
        Input("store-metrics-modal-open", "data"),
    )
    def set_metrics_modal_class(is_open):
        """Add/remove 'open' class on the metrics modal overlay."""
        base = "metrics-modal-overlay"
        return f"{base} open" if is_open else base

    @app.callback(
        Output("custom-analytics-modal-overlay", "className"),
        Input("store-custom-analytics-modal-open", "data"),
    )
    def set_custom_analytics_modal_class(is_open):
        """Add/remove 'open' class on the custom-analytics modal overlay."""
        base = "custom-analytics-modal-overlay"
        return f"{base} open" if is_open else base

    # ================================================================
    # CSV explorer modal
    # ================================================================

    @app.callback(
        Output("store-csv-modal-open", "data"),
        Output("store-selected-csv", "data"),
        Input({"type": "csv-open-btn", "value": dash.ALL}, "n_clicks"),
        Input("close-csv-modal", "n_clicks"),
        Input("csv-modal-backdrop", "n_clicks"),
        State("store-csv-modal-open", "data"),
        State("store-selected-csv", "data"),
        prevent_initial_call=True,
    )
    def toggle_csv_modal(open_clicks, close_clicks, backdrop_clicks, is_open, selected_csv):
        """Open the CSV explorer for a specific strategy or close it."""
        ctx = callback_context
        if not ctx.triggered:
            return is_open or False, selected_csv

        trig_id = ctx.triggered_id
        if isinstance(trig_id, dict) and trig_id.get("type") == "csv-open-btn":
            return True, trig_id.get("value")
        if trig_id in ("close-csv-modal", "csv-modal-backdrop"):
            return False, selected_csv

        return is_open or False, selected_csv

    @app.callback(
        Output("csv-modal-overlay", "className"),
        Input("store-csv-modal-open", "data"),
    )
    def set_csv_modal_class(is_open):
        """Add/remove 'open' class on the CSV modal overlay."""
        base = "csv-modal-overlay"
        return f"{base} open" if is_open else base

    @app.callback(
        Output("csv-modal-table", "columns"),
        Output("csv-modal-table", "data"),
        Output("csv-modal-title", "children"),
        Output("csv-modal-meta", "children"),
        Input("store-selected-csv", "data"),
    )
    def update_csv_modal(selected):
        """Populate the CSV explorer table with the selected strategy data."""
        if not selected or selected not in strategies:
            return [], [], "CSV Explorer", "Select a CSV to view."

        df = strategies[selected]
        display_df = df.copy()
        if "date" in display_df.columns:
            display_df["date"] = (
                pd.to_datetime(display_df["date"], errors="coerce")
                .dt.strftime("%Y-%m-%d")
                .fillna("")
            )

        column_map = {c: c.replace("_", " ").strip().title() for c in display_df.columns}
        display_df = display_df.rename(columns=column_map)
        columns = [{"name": column_map[c], "id": column_map[c]} for c in df.columns]
        data = display_df.to_dict("records")

        title = selected
        return columns, data, title, ""

    # ================================================================
    # Strategy and product selection
    # ================================================================

    @app.callback(
        Output("store-selected-strategy", "data"),
        Output("strategy-radio", "value"),
        Output("home-radio", "value"),
        Output("store-selected-products", "data", allow_duplicate=True),
        Input("strategy-radio", "value"),
        Input("home-radio", "value"),
        Input("btn-reset-layout", "n_clicks"),
        State("store-selected-strategy", "data"),
        prevent_initial_call=True,
    )
    def set_strategy(strategy, home_strategy, reset_clicks, current):
        """Synchronise the selected strategy between sidebar radios and store.

        Also resets the product selection to ALL *in the same callback* when the
        strategy changes, so the panel callbacks rebuild once rather than twice
        (a separate products-reset wave was a major source of lag).
        """
        ctx = callback_context
        selected = current or default_strategy_name()
        if ctx.triggered:
            trig_id = ctx.triggered_id
            if trig_id == "btn-reset-layout":
                selected = "Portfolio"
            if trig_id == "home-radio":
                selected = home_strategy or selected
            elif trig_id == "strategy-radio":
                selected = strategy or selected

        strategy_value = selected if selected in strategy_names else None
        home_value = "Portfolio" if selected == "Portfolio" else None
        products_value = ["ALL"] if selected != current else dash.no_update
        return selected, strategy_value, home_value, products_value

    @app.callback(
        Output("product-buttons", "children"),
        Input("store-selected-strategy", "data"),
    )
    def render_product_buttons(strategy):
        """Regenerate the product pill buttons when the strategy changes."""
        if strategy == "Portfolio":
            if not strategy_names:
                return [
                    html.Div(
                        "No strategies are available for this view.",
                        className="text-muted small fst-italic py-2",
                    )
                ]
            buttons = [product_button("All Strategies", "ALL")]
            buttons.extend([product_button(s, s) for s in strategy_names])
            return buttons

        products = products_for_strategy(strategies, strategy)

        if not products:
            return [
                html.Div(
                    "No product filters available for this view.",
                    className="text-muted small fst-italic py-2",
                )
            ]

        buttons = [product_button("All", "ALL")]
        buttons.extend([product_button(format_product_label(p), p) for p in products])
        return buttons

    @app.callback(
        Output({"type": "product-btn", "value": dash.ALL}, "color"),
        Output({"type": "product-btn", "value": dash.ALL}, "outline"),
        Output({"type": "product-btn", "value": dash.ALL}, "className"),
        Output({"type": "product-btn", "value": dash.ALL}, "active"),
        Input("store-selected-products", "data"),
        Input("store-selected-strategy", "data"),
        State({"type": "product-btn", "value": dash.ALL}, "id"),
    )
    def update_product_button_styles(selected, strategy, ids):
        """Style product buttons based on current selection state."""
        if not ids:
            return [], [], [], []

        selected_set = set(selected or [])

        colors = []
        outlines = []
        classes = []
        actives = []

        for component_id in ids:
            value = component_id.get("value")
            is_active = (value == "ALL" and "ALL" in selected_set) or (value in selected_set)

            colors.append("info" if is_active else "secondary")
            outlines.append(not is_active)
            classes.append("product-pill" + (" active" if is_active else ""))
            actives.append(is_active)

        return colors, outlines, classes, actives

    @app.callback(
        Output("store-selected-products", "data"),
        Input({"type": "product-btn", "value": dash.ALL}, "n_clicks"),
        State({"type": "product-btn", "value": dash.ALL}, "id"),
        State("store-selected-products", "data"),
        prevent_initial_call=True,
    )
    def set_products(n_clicks_list, ids, current):
        """Handle product pill clicks: toggle individual or select ALL.

        Strategy changes reset products via :func:`set_strategy` (same wave), so
        this callback only handles genuine pill clicks. Buttons are recreated on
        every strategy change with ``n_clicks=0``; ignore those non-clicks.
        """
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update

        trig = ctx.triggered[0]
        if not trig.get("value"):  # n_clicks == 0/None -> (re)created, not clicked
            return dash.no_update

        trig_id = ctx.triggered_id
        if isinstance(trig_id, dict) and trig_id.get("type") == "product-btn":
            clicked = trig_id.get("value")
        else:
            return dash.no_update

        if clicked == "ALL":
            return ["ALL"]

        selected = set([p for p in current if p != "ALL"])
        if clicked in selected:
            selected.remove(clicked)
        else:
            selected.add(clicked)

        if len(selected) == 0:
            return ["ALL"]
        return sorted(selected)

    # ================================================================
    # Main dashboard update -- split per panel
    # ----------------------------------------------------------------
    # Previously a single callback rebuilt every figure + metrics table on
    # ANY input change, shipping ~600 KB per interaction. Splitting it so each
    # panel only recomputes when ITS inputs change means a tab switch or a
    # single panel's range cycle rebuilds just that panel. A small memo shares
    # the resolved frame across the panel callbacks, so a strategy/product
    # change resolves it once rather than five times.
    # ================================================================

    _WIN = DEFAULT_ROLL_WINDOW
    _resolve_cache: Dict[tuple, dict] = {}

    def _panel_titles(layout_value: str) -> Tuple[str, str, str]:
        if layout_value == "analytics":
            return "Rolling Sharpe", "Seasonality", "Rolling Correlation"
        if layout_value == "focused":
            return "Equity & Drawdown Curve", "Custom Analytics", "Drawdown"
        return "Equity Curve", "Custom Analytics", "Drawdown"

    def _resolve_context(strategy, selected_products, var_config) -> dict:
        """Resolve the frame + grouping for the current selection (memoized).

        Returns a dict describing how to build any panel: the (VaR-aware) frame,
        the columns to plot, whether to aggregate, labelling, and the header.
        """
        key = (strategy, tuple(selected_products or []), json.dumps(var_config or {}, sort_keys=True))
        cached = _resolve_cache.get(key)
        if cached is not None:
            return cached

        if strategy == "Portfolio":
            sel = selected_products or ["ALL"]
            all_strategies = [s for s in strategy_names if s in strategies]
            selected = all_strategies if "ALL" in sel else [s for s in sel if s in strategies]
            if not selected:
                selected = all_strategies
            df, _breakdown = portfolio_effective_dataframe(strategies, returns, selected, var_config)
            ctx = {
                "empty": df.empty,
                "is_portfolio": True,
                "df": df,
                "cols": selected,
                "all_cols": all_strategies,
                "use_agg": "ALL" in sel,
                "agg_label": "All Strategies",
                "label_fn": (lambda c: c),
                "header": "Portfolio",
                "selected_products": sel,
            }
        else:
            df = strategies.get(strategy)
            all_products = products_for_strategy(strategies, strategy)
            sel = selected_products or ["ALL"]
            var_scaled = False
            if df is not None:
                var_cfg = (var_config or {}).get(strategy) or {}
                returns_df = returns.get(strategy)
                if bool(var_cfg.get("active")) and returns_df is not None:
                    verdict = validate_var_config(
                        var_cfg.get("total_var"), var_cfg.get("allocations", {}), all_products
                    )
                    if verdict["ok"]:
                        scaled_df, _ = compute_var_scaled_frame(
                            df, returns_df, all_products,
                            var_cfg.get("total_var"), var_cfg.get("allocations", {}),
                        )
                        if not scaled_df.empty:
                            df = scaled_df
                            all_products = [c for c in scaled_df.columns if c != "date"]
                            var_scaled = True
            products = all_products if ("ALL" in sel) else [p for p in sel if p in all_products]
            ctx = {
                "empty": df is None,
                "is_portfolio": False,
                "df": df,
                "cols": products,
                "all_cols": all_products,
                "use_agg": "ALL" in sel,
                "agg_label": "All Products",
                "label_fn": format_product_label,
                "header": f"{strategy} Trading Strategy" + (" (VaR-scaled)" if var_scaled else ""),
                "selected_products": sel,
            }

        if len(_resolve_cache) > 64:
            _resolve_cache.clear()
        _resolve_cache[key] = ctx
        return ctx

    def _series(ctx, range_key):
        fdf = filter_df_by_range(ctx["df"], range_key)
        if ctx["use_agg"]:
            return {ctx["agg_label"]: compute_series(fdf, ctx["cols"])}
        return {ctx["label_fn"](c): compute_series(fdf, [c]) for c in ctx["cols"]}

    def _finalize(fig, theme):
        fig.update_layout(autosize=True, margin=dict(l=14, r=14, t=40, b=22))
        apply_plotly_theme(fig, theme)
        return fig

    # ---- Panel titles (cheap; no figure work) ----
    @app.callback(
        Output("header-title", "children"),
        Output("equity-panel-title", "children"),
        Output("custom-panel-title", "children"),
        Output("drawdown-panel-title", "children"),
        Output("equity-modal-title", "children"),
        Output("custom-analytics-modal-title", "children"),
        Output("drawdown-modal-title", "children"),
        Input("store-selected-strategy", "data"),
        Input("store-selected-products", "data"),
        Input("layout-radio", "value"),
        Input("store-var-config", "data"),
    )
    def update_titles(strategy, selected_products, layout_value, var_config):
        eq_t, cu_t, dd_t = _panel_titles(layout_value or "default")
        ctx = _resolve_context(strategy, selected_products, var_config)
        return ctx["header"], eq_t, cu_t, dd_t, eq_t, cu_t, dd_t

    # ---- Equity panel ----
    @app.callback(
        Output("equity-graph", "figure"),
        Input("store-selected-strategy", "data"),
        Input("store-selected-products", "data"),
        Input("layout-radio", "value"),
        Input("store-equity-range", "data"),
        Input("store-theme", "data"),
        Input("store-var-config", "data"),
    )
    def update_equity(strategy, selected_products, layout_value, equity_range, theme, var_config):
        layout_value = layout_value or "default"
        ctx = _resolve_context(strategy, selected_products, var_config)
        if ctx["empty"]:
            return _finalize(
                placeholder_figure("Portfolio view", "Add your portfolio data to explore performance and analytics."),
                theme,
            )
        if layout_value == "analytics":
            fdf = filter_df_by_range(ctx["df"], equity_range)
            fig = rolling_sharpe_figure(
                fdf, ctx["cols"], _WIN, "",
                include_individuals=not ctx["use_agg"], include_aggregate=ctx["use_agg"],
            )
        else:
            combine = layout_value == "focused"
            eq_series = _series(ctx, equity_range)
            dd_series = _series(ctx, equity_range) if combine else None
            fig = equity_figure(eq_series, "", drawdown_series=dd_series)
        return _finalize(fig, theme)

    # ---- Drawdown panel ----
    @app.callback(
        Output("drawdown-graph", "figure"),
        Input("store-selected-strategy", "data"),
        Input("store-selected-products", "data"),
        Input("layout-radio", "value"),
        Input("store-drawdown-range", "data"),
        Input("store-theme", "data"),
        Input("store-var-config", "data"),
    )
    def update_drawdown(strategy, selected_products, layout_value, drawdown_range, theme, var_config):
        layout_value = layout_value or "default"
        ctx = _resolve_context(strategy, selected_products, var_config)
        if ctx["empty"]:
            return _finalize(placeholder_figure("Drawdowns will display once data is connected."), theme)
        if layout_value == "analytics":
            fdf = filter_df_by_range(ctx["df"], drawdown_range)
            fig = rolling_correlation_figure(fdf, ctx["cols"], _WIN, "")
        else:
            fig = drawdown_figure(_series(ctx, drawdown_range), "")
        return _finalize(fig, theme)

    # ---- Custom analytics panel ----
    @app.callback(
        Output("custom-graph", "figure"),
        Input("store-selected-strategy", "data"),
        Input("store-selected-products", "data"),
        Input("custom-tabs", "active_tab"),
        Input("layout-radio", "value"),
        Input("store-custom-range", "data"),
        Input("store-theme", "data"),
        Input("store-var-config", "data"),
        Input("store-corr-mode", "data"),
        Input("corr-range-slider", "value"),
    )
    def update_custom(
        strategy, selected_products, active_tab, layout_value, custom_range,
        theme, var_config, corr_mode, corr_range,
    ):
        layout_value = layout_value or "default"
        ctx = _resolve_context(strategy, selected_products, var_config)
        if ctx["empty"]:
            return _finalize(placeholder_figure("Custom analytics will appear here."), theme)
        # On the VaR tab the graph is hidden (summary table shown) -> skip the heavy build.
        if active_tab == "tab-var" and layout_value != "analytics":
            return _finalize(placeholder_figure("VaR Scaling", "Configure VaR scaling for this strategy."), theme)
        fdf = filter_df_by_range(ctx["df"], custom_range)
        if layout_value == "analytics" or active_tab == "tab-season":
            fig = seasonality_figure(fdf, ctx["cols"], "")
        elif active_tab == "tab-roll":
            fig = rolling_sharpe_figure(
                fdf, ctx["cols"], _WIN, "",
                include_individuals=not ctx["use_agg"], include_aggregate=ctx["use_agg"],
            )
        else:  # tab-corr -> static matrix (date-slider range) or rolling chart
            if (corr_mode or "matrix") == "matrix":
                full = ctx["df"]
                n = len(full)
                rng = corr_range or [0, n - 1]
                lo = max(0, min(int(rng[0]), max(0, n - 1)))
                hi = max(lo, min(int(rng[1]), max(0, n - 1)))
                sub = full.iloc[lo:hi + 1] if n else full
                fig = correlation_matrix_figure(sub, ctx["cols"], "")
            else:
                fig = rolling_correlation_figure(fdf, ctx["cols"], _WIN, "")
        return _finalize(fig, theme)

    # ---- Key metrics (inline + modal share the same data) ----
    @app.callback(
        Output("metrics-table", "columns"),
        Output("metrics-table", "data"),
        Output("metrics-modal-table", "columns"),
        Output("metrics-modal-table", "data"),
        Input("store-selected-strategy", "data"),
        Input("store-selected-products", "data"),
        Input("store-metrics-range", "data"),
        Input("store-var-config", "data"),
    )
    def update_metrics(strategy, selected_products, metrics_range, var_config):
        ctx = _resolve_context(strategy, selected_products, var_config)
        if ctx["empty"]:
            return [], [], [], []
        fdf = filter_df_by_range(ctx["df"], metrics_range)
        if ctx["is_portfolio"]:
            cols, data = build_portfolio_metrics_table(ctx["selected_products"], ctx["all_cols"], fdf)
        else:
            cols, data = build_metrics_table(fdf, ctx["selected_products"], ctx["all_cols"])
        return cols, data, cols, data

    # ================================================================
    # Correlation tab: Matrix/Rolling toggle + static-matrix date slider
    # ================================================================

    def _corr_dates(ctx):
        df = ctx.get("df")
        if df is None or "date" not in getattr(df, "columns", []) or len(df) == 0:
            return None
        return pd.to_datetime(df["date"]).reset_index(drop=True)

    def _corr_slider_config(ctx):
        """Return (min, max, marks, value) for the date-range slider over ctx's dates."""
        dates = _corr_dates(ctx)
        if dates is None or len(dates) < 2:
            return 0, 1, {}, [0, 1]
        n = len(dates)
        marks: Dict[int, str] = {}
        seen = set()
        for i, d in enumerate(dates):
            y = int(d.year)
            if y not in seen:
                seen.add(y)
                marks[i] = str(y)
        return 0, n - 1, marks, [0, n - 1]

    @app.callback(
        Output("store-corr-mode", "data"),
        Input("corr-mode-matrix-btn", "n_clicks"),
        Input("corr-mode-rolling-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def set_corr_mode(_matrix_clicks, _rolling_clicks):
        """Toggle the correlation view mode from the segmented buttons."""
        return "rolling" if callback_context.triggered_id == "corr-mode-rolling-btn" else "matrix"

    @app.callback(
        Output("corr-mode-matrix-btn", "className"),
        Output("corr-mode-rolling-btn", "className"),
        Input("store-corr-mode", "data"),
    )
    def style_corr_mode_buttons(mode):
        """Highlight the active segment of the Matrix/Rolling toggle."""
        matrix_active = (mode or "matrix") == "matrix"
        return (
            "corr-mode-btn active" if matrix_active else "corr-mode-btn",
            "corr-mode-btn" if matrix_active else "corr-mode-btn active",
        )

    @app.callback(
        Output("correlation-controls", "style"),
        Output("corr-range-wrapper", "style"),
        Input("custom-tabs", "active_tab"),
        Input("layout-radio", "value"),
        Input("store-corr-mode", "data"),
    )
    def correlation_controls_visibility(active_tab, layout_value, mode):
        """Show the controls only on the Correlation tab; the slider only in Matrix mode."""
        on_corr_tab = active_tab == "tab-corr" and (layout_value or "default") != "analytics"
        controls = {"display": "flex"} if on_corr_tab else {"display": "none"}
        slider = {"display": "flex"} if (on_corr_tab and (mode or "matrix") == "matrix") else {"display": "none"}
        return controls, slider

    @app.callback(
        Output("corr-range-slider", "min"),
        Output("corr-range-slider", "max"),
        Output("corr-range-slider", "marks"),
        Output("corr-range-slider", "value"),
        Input("store-selected-strategy", "data"),
        Input("store-selected-products", "data"),
        Input("store-var-config", "data"),
        Input("custom-tabs", "active_tab"),
    )
    def init_corr_slider(strategy, selected_products, var_config, active_tab):
        """Configure the slider bounds/marks for the current frame.

        The full-range *value* is only (re)set when the Correlation tab is active,
        so switching strategies on other tabs does not re-trigger ``update_custom``.
        """
        ctx = _resolve_context(strategy, selected_products, var_config)
        lo, hi, marks, value = _corr_slider_config(ctx)
        if active_tab != "tab-corr":
            return lo, hi, marks, dash.no_update
        return lo, hi, marks, value

    @app.callback(
        Output("corr-range-label", "children"),
        Input("corr-range-slider", "value"),
        State("store-selected-strategy", "data"),
        State("store-selected-products", "data"),
        State("store-var-config", "data"),
    )
    def corr_range_label(value, strategy, selected_products, var_config):
        """Show the selected date range as 'Mon YYYY – Mon YYYY'."""
        ctx = _resolve_context(strategy, selected_products, var_config)
        dates = _corr_dates(ctx)
        if dates is None or not value:
            return ""
        n = len(dates)
        lo = max(0, min(int(value[0]), n - 1))
        hi = max(lo, min(int(value[1]), n - 1))
        return f"{dates.iloc[lo]:%b %Y} – {dates.iloc[hi]:%b %Y}"

    # ================================================================
    # Lazy modal-graph population
    # ----------------------------------------------------------------
    # The expand modals mirror the inline graphs. Rendering all three
    # (Plotly) modal graphs on every dashboard update doubled the render
    # cost for no benefit while they were closed. The dashboard controls
    # are blocked while a modal is open (backdrop), so the inline figure
    # cannot change mid-view -- copying the figure once on open is enough.
    # ================================================================

    def _mirror_on_open(modal_store_id: str, source_graph_id: str, target_graph_id: str) -> None:
        @app.callback(
            Output(target_graph_id, "figure"),
            Input(modal_store_id, "data"),
            State(source_graph_id, "figure"),
            prevent_initial_call=True,
        )
        def _mirror(is_open, figure):
            return figure if is_open else dash.no_update

    _mirror_on_open("store-equity-modal-open", "equity-graph", "equity-modal-graph")
    _mirror_on_open("store-drawdown-modal-open", "drawdown-graph", "drawdown-modal-graph")
    _mirror_on_open("store-custom-analytics-modal-open", "custom-graph", "custom-analytics-modal-graph")

    # ================================================================
    # VaR scaling callbacks (modals, validation, per-strategy memory,
    # summary tables) -- registered from a dedicated module.
    # ================================================================
    register_var_callbacks(app, strategies, returns)
