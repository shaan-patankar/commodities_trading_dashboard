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
from dashboard.table_styles import (
    LOWER_IS_BETTER,
    metrics_cell_style,
    metrics_header_style,
    table_data_conditional,
)
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
    effective_scaled_frame,
    portfolio_effective_dataframe,
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
    # Value heat-map overlay (Settings toggle)
    # ================================================================

    @app.callback(
        Output("store-table-heatmap", "data"),
        Output("btn-toggle-heatmap", "children"),
        Output("btn-toggle-heatmap", "className"),
        Input("btn-toggle-heatmap", "n_clicks"),
        Input("btn-reset-layout", "n_clicks"),
        State("store-table-heatmap", "data"),
        prevent_initial_call=True,
    )
    def toggle_table_heatmap(_toggle_clicks, _reset_clicks, is_on):
        """Flip the value heat-map overlay; reset turns it back off."""
        if callback_context.triggered_id == "btn-reset-layout":
            on = False
        else:
            on = not bool(is_on)
        label = "Value Heatmap: On" if on else "Value Heatmap: Off"
        cls = "settings-option-btn active" if on else "settings-option-btn"
        return on, label, cls

    @app.callback(
        Output("metrics-table", "style_data_conditional"),
        Output("metrics-modal-table", "style_data_conditional"),
        Input("metrics-table", "data"),
        Input("metrics-table", "columns"),
        Input("store-table-heatmap", "data"),
    )
    def style_metrics_heatmap(data, columns, heatmap_on):
        """Tint the Key Metrics cells (inline + modal) when the overlay is on.

        Normalised per metric row so each metric's columns are compared against
        each other (green = best); single-column views fall back to sign.
        """
        conditional = table_data_conditional(
            data, columns,
            label_ids={"Metric"}, orient="row",
            lower_better=LOWER_IS_BETTER, heatmap_on=bool(heatmap_on),
        )
        return conditional, conditional

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
            return "Equity & Drawdown Curve", "Analytics", "Drawdown"
        return "Equity Curve", "Analytics", "Drawdown"

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
            suffix = ""
            if df is not None:
                var_cfg = (var_config or {}).get(strategy) or {}
                returns_df = returns.get(strategy)
                scaled_df, suffix = effective_scaled_frame(df, returns_df, all_products, var_cfg)
                if not scaled_df.empty:
                    df = scaled_df
                    all_products = [c for c in scaled_df.columns if c != "date"]
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
                "header": f"{strategy} Trading Strategy" + suffix,
                "selected_products": sel,
            }

        if len(_resolve_cache) > 64:
            _resolve_cache.clear()
        _resolve_cache[key] = ctx
        return ctx

    def _series(ctx, range_key):
        fdf = filter_df_by_range(ctx["df"], range_key)
        return _series_from(ctx, fdf)

    def _series_from(ctx, fdf):
        """Build the label->SeriesPack mapping from an already-windowed frame."""
        if ctx["use_agg"]:
            return {ctx["agg_label"]: compute_series(fdf, ctx["cols"])}
        return {ctx["label_fn"](c): compute_series(fdf, [c]) for c in ctx["cols"]}

    def _snap_month_window(dates, lo, hi):
        """Snap a ``[lo, hi]`` index window to whole calendar months.

        FROM (``lo``) moves to the first row of ``dates[lo]``'s month (≈ the
        month start); TO (``hi``) moves to the last row of ``dates[hi]``'s month
        (≈ the month end), so a window picked mid-month covers those months in
        full. Purely index-based, so the result stays valid for the shared frame.
        """
        if dates is None or len(dates) == 0:
            return lo, hi
        n = len(dates)
        lo = max(0, min(int(lo), n - 1))
        hi = max(lo, min(int(hi), n - 1))
        periods = dates.dt.to_period("M")
        # dates carries a 0..n-1 RangeIndex, so idxmax returns positional indices.
        new_lo = int((periods == periods.iloc[lo]).idxmax())
        new_hi = int((periods == periods.iloc[hi]).iloc[::-1].idxmax())
        return new_lo, new_hi

    def _window_df(ctx, cycle_key, cal_value):
        """Resolve a panel's working frame.

        A calendar window ([lo, hi] row indices) takes precedence over the cycle
        preset; a window spanning the full extent (or ``None``) defers to the
        preset. Index-based slicing matches the shared resolved frame, so a
        global window means the same dates across every panel. The window is
        snapped to whole months so plots and metrics always start/end on month
        boundaries.
        """
        df = ctx["df"]
        if df is None:
            return df
        n = len(df)
        if cal_value and n:
            lo = max(0, min(int(cal_value[0]), n - 1))
            hi = max(lo, min(int(cal_value[1]), n - 1))
            if "date" in getattr(df, "columns", []):
                dates = pd.to_datetime(df["date"]).reset_index(drop=True)
                lo, hi = _snap_month_window(dates, lo, hi)
            if not (lo <= 0 and hi >= n - 1):
                return df.iloc[lo:hi + 1]
        return filter_df_by_range(df, cycle_key)

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
        Input({"type": "cal-range", "panel": "equity"}, "data"),
    )
    def update_equity(strategy, selected_products, layout_value, equity_range, theme, var_config, cal_value):
        layout_value = layout_value or "default"
        ctx = _resolve_context(strategy, selected_products, var_config)
        if ctx["empty"]:
            return _finalize(
                placeholder_figure("Portfolio view", "Add your portfolio data to explore performance and analytics."),
                theme,
            )
        fdf = _window_df(ctx, equity_range, cal_value)
        if layout_value == "analytics":
            fig = rolling_sharpe_figure(
                fdf, ctx["cols"], _WIN, "",
                include_individuals=not ctx["use_agg"], include_aggregate=ctx["use_agg"],
            )
        else:
            combine = layout_value == "focused"
            eq_series = _series_from(ctx, fdf)
            dd_series = _series_from(ctx, fdf) if combine else None
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
        Input({"type": "cal-range", "panel": "drawdown"}, "data"),
    )
    def update_drawdown(strategy, selected_products, layout_value, drawdown_range, theme, var_config, cal_value):
        layout_value = layout_value or "default"
        ctx = _resolve_context(strategy, selected_products, var_config)
        if ctx["empty"]:
            return _finalize(placeholder_figure("Drawdowns will display once data is connected."), theme)
        fdf = _window_df(ctx, drawdown_range, cal_value)
        if layout_value == "analytics":
            fig = rolling_correlation_figure(fdf, ctx["cols"], _WIN, "")
        else:
            fig = drawdown_figure(_series_from(ctx, fdf), "")
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
        Input({"type": "cal-range", "panel": "custom"}, "data"),
    )
    def update_custom(
        strategy, selected_products, active_tab, layout_value, custom_range,
        theme, var_config, corr_mode, cal_value,
    ):
        layout_value = layout_value or "default"
        ctx = _resolve_context(strategy, selected_products, var_config)
        if ctx["empty"]:
            return _finalize(placeholder_figure("Custom analytics will appear here."), theme)
        # On the VaR tab the graph is hidden (summary table shown) -> skip the heavy build.
        if active_tab == "tab-var" and layout_value != "analytics":
            return _finalize(placeholder_figure("VaR Scaling", "Configure VaR scaling for this strategy."), theme)
        fdf = _window_df(ctx, custom_range, cal_value)
        if layout_value == "analytics" or active_tab == "tab-season":
            fig = seasonality_figure(fdf, ctx["cols"], "")
        elif active_tab == "tab-roll":
            fig = rolling_sharpe_figure(
                fdf, ctx["cols"], _WIN, "",
                include_individuals=not ctx["use_agg"], include_aggregate=ctx["use_agg"],
            )
        else:  # tab-corr -> static matrix or rolling chart (both over the windowed frame)
            if (corr_mode or "matrix") == "matrix":
                fig = correlation_matrix_figure(fdf, ctx["cols"], "")
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
        Input({"type": "cal-range", "panel": "metrics"}, "data"),
    )
    def update_metrics(strategy, selected_products, metrics_range, var_config, cal_value):
        ctx = _resolve_context(strategy, selected_products, var_config)
        if ctx["empty"]:
            return [], [], [], []
        fdf = _window_df(ctx, metrics_range, cal_value)
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

    # Correlation view mode (static matrix <-> rolling) is toggled by
    # double-clicking the "Correlation" tab itself — no separate buttons. A
    # clientside listener flips store-corr-mode via set_props; default = matrix.
    # Output goes to a benign prop (no_update) to avoid a store self-cycle.
    app.clientside_callback(
        """
        function(activeTab) {
            if (window.__corrMode === undefined) window.__corrMode = 'matrix';
            if (!window.__corrDblBound) {
                const tabs = document.querySelectorAll('#custom-tabs a, .custom-tabs a');
                let bound = false;
                tabs.forEach(function(a) {
                    if (a.textContent.trim() === 'Correlation' && !a.__corrDbl) {
                        a.__corrDbl = true;
                        bound = true;
                        a.addEventListener('dblclick', function(e) {
                            e.preventDefault();
                            const next = (window.__corrMode === 'matrix') ? 'rolling' : 'matrix';
                            window.__corrMode = next;
                            window.dash_clientside.set_props('store-corr-mode', {data: next});
                        });
                    }
                });
                if (bound) window.__corrDblBound = true;
            }
            // Close an open calendar bubble on any click outside it (and outside
            // the calendar button, which manages its own toggle).
            if (!window.__calOutsideBound) {
                window.__calOutsideBound = true;
                document.addEventListener('click', function(e) {
                    const openPop = document.querySelector('.cal-popover.open');
                    if (!openPop) return;
                    if (openPop.contains(e.target)) return;
                    if (e.target.closest && e.target.closest('.cal-btn')) return;
                    const closeBtn = openPop.querySelector('.cal-pop-close');
                    if (closeBtn) closeBtn.click();
                });
            }
            // Debounced window-resize tick -> drives the VaR table's dynamic
            // blank-row padding to re-fit when the viewport size changes.
            if (!window.__winResizeBound) {
                window.__winResizeBound = true;
                let rt;
                window.addEventListener('resize', function() {
                    clearTimeout(rt);
                    rt = setTimeout(function() {
                        window.dash_clientside.set_props('store-win-tick',
                            {data: (window.__winTick = (window.__winTick || 0) + 1)});
                        if (window.__evalProductOverflow) window.__evalProductOverflow();
                    }, 180);
                });
            }

            // ---- Product-bar overflow + shared arrow-key scrolling ----
            if (!window.__scrollHelpers) {
                window.__scrollHelpers = true;
                window.__SCROLL_STEP = 150;   // px per arrow press
                window.__OVF_TOL = 4;         // overflow tolerance (sub-pixel)

                // When the product pills overflow, switch the bar to flex-start so
                // the first pills stay reachable by the arrow keys (flex-end pushes
                // the overflow off-screen left, where scrollLeft can't reach it).
                // Measure the CONTENT width vs the bar — the bar's own scrollWidth is
                // unreliable under justify-content:flex-end.
                window.__evalProductOverflow = function() {
                    var bar = document.querySelector('.topbar-products');
                    var content = document.getElementById('product-buttons');
                    if (!bar || !content) return;
                    var overflowing = (content.scrollWidth - bar.clientWidth) > window.__OVF_TOL;
                    bar.classList.toggle('is-overflowing', overflowing);
                };

                // First element in *root* (self or descendant) that can scroll on
                // *axis* ('x' or 'y').
                window.__findScrollable = function(root, axis) {
                    if (!root) return null;
                    var els = [root];
                    var kids = root.querySelectorAll ? root.querySelectorAll('*') : [];
                    for (var k = 0; k < kids.length; k++) els.push(kids[k]);
                    for (var i = 0; i < els.length; i++) {
                        var el = els[i], cs = getComputedStyle(el);
                        if (axis === 'x' && (el.scrollWidth - el.clientWidth) > window.__OVF_TOL
                            && (cs.overflowX === 'auto' || cs.overflowX === 'scroll')) return el;
                        if (axis === 'y' && (el.scrollHeight - el.clientHeight) > window.__OVF_TOL
                            && (cs.overflowY === 'auto' || cs.overflowY === 'scroll')) return el;
                    }
                    return null;
                };
            }

            // Bind the overflow evaluator once; re-run when the product pills change.
            if (!window.__productEvalBound) {
                if (document.querySelector('.topbar-products')) {
                    window.__productEvalBound = true;
                    window.__evalProductOverflow();
                }
            }
            if (!window.__productObsBound) {
                var pbtns = document.getElementById('product-buttons');
                if (pbtns) {
                    window.__productObsBound = true;
                    var mo = new MutationObserver(function() {
                        var bar = document.querySelector('.topbar-products');
                        if (bar) bar.scrollLeft = 0;
                        window.requestAnimationFrame(function() {
                            window.requestAnimationFrame(window.__evalProductOverflow);
                        });
                    });
                    mo.observe(pbtns, { childList: true, subtree: false });
                }
            }

            // Shared keyboard scrolling for the product bar and the metrics / VaR
            // tables. Left/Right scroll horizontally; Up/Down scroll vertically;
            // Home/End jump to the extremes. A fullscreen modal (Key Metrics or VaR
            // Scaling) always owns the keys while open; otherwise the keys act on the
            // scrollable region under the mouse (product bar or a table). For tables,
            // Home -> top + far-left, End -> bottom + far-left; for the product bar,
            // Home -> far-left, End -> far-right.
            if (!window.__arrowKeysBound) {
                window.__arrowKeysBound = true;
                window.__lastHovered = null;
                document.addEventListener('mousemove', function(e) {
                    window.__lastHovered = (e.target && e.target.closest)
                        ? e.target.closest('.topbar-products, .dash-table-container')
                        : null;
                });
                document.addEventListener('keydown', function(e) {
                    var k = e.key;
                    var isArrow = (k === 'ArrowLeft' || k === 'ArrowRight' ||
                                   k === 'ArrowUp' || k === 'ArrowDown');
                    var isHomeEnd = (k === 'Home' || k === 'End');
                    if (!isArrow && !isHomeEnd) return;
                    var ae = document.activeElement;
                    if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA' ||
                               ae.isContentEditable)) return;
                    var modal = document.querySelector(
                        '.metrics-modal-overlay.open, .var-expand-overlay.open');
                    var root = modal ? modal : window.__lastHovered;
                    if (!root) return;
                    var xt = window.__findScrollable(root, 'x');
                    var yt = window.__findScrollable(root, 'y');
                    if (!xt && !yt) return;
                    var step = window.__SCROLL_STEP;

                    if (isHomeEnd) {
                        var pureHoriz = xt && !yt;   // product bar: single horizontal axis
                        if (pureHoriz) {
                            var left = (k === 'Home') ? 0 : (xt.scrollWidth - xt.clientWidth);
                            xt.scrollTo({ left: left, behavior: 'smooth' });
                        } else {
                            if (xt) xt.scrollTo({ left: 0, behavior: 'smooth' });   // far-left
                            if (yt) {
                                var top = (k === 'Home') ? 0 : (yt.scrollHeight - yt.clientHeight);
                                yt.scrollTo({ top: top, behavior: 'smooth' });
                            }
                        }
                        e.preventDefault();
                        return;
                    }

                    var horiz = (k === 'ArrowLeft' || k === 'ArrowRight');
                    var target = horiz ? xt : yt;
                    if (!target) return;
                    if (horiz) {
                        target.scrollBy({ left: (k === 'ArrowRight' ? 1 : -1) * step, behavior: 'smooth' });
                    } else {
                        target.scrollBy({ top: (k === 'ArrowDown' ? 1 : -1) * step, behavior: 'smooth' });
                    }
                    e.preventDefault();
                });
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("custom-tabs", "className"),
        Input("custom-tabs", "active_tab"),
    )

    # ================================================================
    # Per-panel calendar date-range popovers (local / global scope)
    # ----------------------------------------------------------------
    # Each panel header has a calendar button opening a small bubble with a
    # two-handle date slider. A window slices that panel's frame by row index;
    # in Global scope a window on any panel drives every panel (same dates,
    # since all panels share the resolved frame).
    # ================================================================

    @app.callback(
        Output("store-daterange-scope", "data"),
        Output("btn-daterange-scope", "children"),
        Output("btn-daterange-scope", "className"),
        Input("btn-daterange-scope", "n_clicks"),
        Input("btn-reset-layout", "n_clicks"),
        State("store-daterange-scope", "data"),
        prevent_initial_call=True,
    )
    def toggle_daterange_scope(_toggle, _reset, scope):
        """Flip local/global date-range scope; reset returns to local."""
        if callback_context.triggered_id == "btn-reset-layout":
            scope = "local"
        else:
            scope = "global" if (scope or "local") == "local" else "local"
        label = "Date Range: Global" if scope == "global" else "Date Range: Local"
        cls = "settings-option-btn active" if scope == "global" else "settings-option-btn"
        return scope, label, cls

    @app.callback(
        Output({"type": "cal-open", "panel": dash.ALL}, "data"),
        Input({"type": "cal-btn", "panel": dash.ALL}, "n_clicks"),
        Input({"type": "cal-close", "panel": dash.ALL}, "n_clicks"),
        State({"type": "cal-open", "panel": dash.ALL}, "data"),
        State({"type": "cal-open", "panel": dash.ALL}, "id"),
        prevent_initial_call=True,
    )
    def toggle_cal_popovers(_btns, _closes, opens, ids):
        """Open the clicked panel's bubble (toggling), closing all others."""
        trig = callback_context.triggered_id
        result = []
        for i, cid in enumerate(ids):
            same = isinstance(trig, dict) and trig.get("panel") == cid["panel"]
            if same and trig.get("type") == "cal-btn":
                result.append(not bool(opens[i]))
            else:
                result.append(False)
        return result

    @app.callback(
        Output({"type": "cal-pop", "panel": dash.ALL}, "className"),
        Input({"type": "cal-open", "panel": dash.ALL}, "data"),
    )
    def cal_popover_class(opens):
        return ["cal-popover open" if o else "cal-popover" for o in opens]

    @app.callback(
        Output({"type": "cal-btn", "panel": dash.ALL}, "className"),
        Input({"type": "cal-open", "panel": dash.ALL}, "data"),
    )
    def cal_btn_class(opens):
        """Highlight the calendar button only while its bubble is open (hover is CSS)."""
        return ["cal-btn active" if o else "cal-btn" for o in opens]

    @app.callback(
        Output({"type": "cal-slider", "panel": dash.ALL}, "min"),
        Output({"type": "cal-slider", "panel": dash.ALL}, "max"),
        Output({"type": "cal-slider", "panel": dash.ALL}, "marks"),
        Output({"type": "cal-slider", "panel": dash.ALL}, "value"),
        Output({"type": "cal-range", "panel": dash.ALL}, "data"),
        Input("store-selected-strategy", "data"),
        Input("store-selected-products", "data"),
        Input("store-var-config", "data"),
        State({"type": "cal-slider", "panel": dash.ALL}, "id"),
    )
    def init_cal_sliders(strategy, selected_products, var_config, ids):
        """Re-bound every slider to the current frame and clear any stale window.

        The year tick labels are hidden via CSS — the live From/To readouts
        (driven by ``drag_value``) show the current dates while dragging. We still
        pass a tiny explicit ``marks`` dict (the two endpoints) rather than ``{}``,
        because dcc treats a falsy ``marks`` as "auto-generate" and would emit one
        mark per step (thousands of hidden DOM nodes).
        """
        ctx = _resolve_context(strategy, selected_products, var_config)
        lo, hi, _marks, value = _corr_slider_config(ctx)
        marks = {lo: "", hi: ""}
        k = len(ids)
        return [lo] * k, [hi] * k, [marks] * k, [value] * k, [None] * k

    @app.callback(
        Output({"type": "cal-range", "panel": dash.ALL}, "data", allow_duplicate=True),
        Output({"type": "cal-slider", "panel": dash.ALL}, "value", allow_duplicate=True),
        Input({"type": "cal-slider", "panel": dash.ALL}, "value"),
        State({"type": "cal-slider", "panel": dash.ALL}, "id"),
        State({"type": "cal-slider", "panel": dash.ALL}, "min"),
        State({"type": "cal-slider", "panel": dash.ALL}, "max"),
        State("store-daterange-scope", "data"),
        prevent_initial_call=True,
    )
    def mirror_cal_ranges(values, ids, mins, maxs, scope):
        """Write the dragged slider into its window store; mirror all when global.

        A window spanning the full extent is stored as ``None`` (no filter).
        """
        trig = callback_context.triggered_id
        k = len(ids)
        idx = next(
            (i for i, cid in enumerate(ids) if isinstance(trig, dict) and cid["panel"] == trig.get("panel")),
            None,
        )
        if idx is None:
            return [dash.no_update] * k, [dash.no_update] * k

        v = values[idx]
        mn, mx = mins[idx], maxs[idx]
        if not v:
            window = None
        else:
            lo, hi = int(v[0]), int(v[1])
            window = None if (lo <= mn and hi >= mx) else [lo, hi]

        if (scope or "local") == "global":
            return [window] * k, [v] * k

        ranges = [dash.no_update] * k
        ranges[idx] = window
        return ranges, [dash.no_update] * k

    @app.callback(
        Output({"type": "cal-from", "panel": dash.ALL}, "children"),
        Output({"type": "cal-to", "panel": dash.ALL}, "children"),
        Input({"type": "cal-slider", "panel": dash.ALL}, "value"),
        Input({"type": "cal-slider", "panel": dash.ALL}, "drag_value"),
        State("store-selected-strategy", "data"),
        State("store-selected-products", "data"),
        State("store-var-config", "data"),
    )
    def cal_labels(values, drag_values, strategy, selected_products, var_config):
        """Feed each bubble's From / To readouts from its slider window.

        While dragging, ``drag_value`` updates continuously (the committed
        ``value`` — and hence the chart — only changes on release), so the
        readouts track the handles live. On release/init the committed value is
        used. Only the open bubble is visible, so stale drag values on the other
        (hidden) panels don't matter.
        """
        trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
        dragging = trig.endswith(".drag_value")
        ctx = _resolve_context(strategy, selected_products, var_config)
        dates = _corr_dates(ctx)
        froms, tos = [], []
        for v, dv in zip(values, drag_values):
            use = dv if (dragging and dv) else v
            if dates is None or not use:
                froms.append("—")
                tos.append("—")
                continue
            n = len(dates)
            lo = max(0, min(int(use[0]), n - 1))
            hi = max(lo, min(int(use[1]), n - 1))
            froms.append(f"{dates.iloc[lo]:%b %Y}")
            tos.append(f"{dates.iloc[hi]:%b %Y}")
        return froms, tos

    # Double-click the calendar button -> reset that panel's window to full
    # (clientside so the rapid second click is caught by wall-clock timing).
    app.clientside_callback(
        """
        function(nClicks, ids, mins, maxs, vals) {
            const noup = (vals || []).map(function(){ return window.dash_clientside.no_update; });
            const ctx = window.dash_clientside.callback_context;
            if (!ctx || !ctx.triggered || !ctx.triggered.length) return noup;
            const propId = ctx.triggered[0].prop_id;
            let panel = null;
            try { panel = JSON.parse(propId.split('.n_clicks')[0]).panel; } catch (e) { return noup; }
            const now = Date.now();
            window._calClick = window._calClick || {};
            const last = window._calClick[panel] || 0;
            window._calClick[panel] = now;
            if (now - last < 350) {
                const idx = ids.findIndex(function(id){ return id.panel === panel; });
                if (idx >= 0) noup[idx] = [mins[idx], maxs[idx]];
            }
            return noup;
        }
        """,
        Output({"type": "cal-slider", "panel": dash.ALL}, "value", allow_duplicate=True),
        Input({"type": "cal-btn", "panel": dash.ALL}, "n_clicks"),
        State({"type": "cal-slider", "panel": dash.ALL}, "id"),
        State({"type": "cal-slider", "panel": dash.ALL}, "min"),
        State({"type": "cal-slider", "panel": dash.ALL}, "max"),
        State({"type": "cal-slider", "panel": dash.ALL}, "value"),
        prevent_initial_call=True,
    )

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
