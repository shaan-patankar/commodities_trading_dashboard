"""Dash callback registration for the trading strategy dashboard.

Contains all interactive logic: sidebar toggles, strategy/product selection,
theme switching, panel visibility, range cycling, modal open/close,
CSV explorer, and the main ``update_dashboard`` callback that rebuilds
all figures and metric tables in response to user input.
"""

from __future__ import annotations

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
from dashboard.config import BASE_FONT, DEFAULT_INITIAL_CAPITAL, DEFAULT_RF, DEFAULT_ROLL_WINDOW, PANEL_KEYS
from dashboard.data import portfolio_dataframe, products_for_strategy
from dashboard.figures import (
    drawdown_figure,
    equity_figure,
    placeholder_figure,
    rolling_correlation_figure,
    rolling_sharpe_figure,
    seasonality_figure,
)
from dashboard.utils import format_product_label


# ---------------------------------------------------------------------------
# Metrics-table builders
# ---------------------------------------------------------------------------

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

    columns = [{"name": "Metric", "id": "Metric"}] + [
        {"name": label, "id": label} for label, _ in display_columns
    ]

    metrics_by_column: Dict[str, List[dict]] = {}
    for label, product_list in display_columns:
        sp = compute_series(df, product_list, DEFAULT_INITIAL_CAPITAL)
        metrics_by_column[label] = compute_metrics(sp, rf_annual=DEFAULT_RF)

    metric_order = [row["Metric"] for row in next(iter(metrics_by_column.values()), [])]
    rows: List[dict] = []
    for metric in metric_order:
        row = {"Metric": metric}
        for label, metrics in metrics_by_column.items():
            value = next((r["Value"] for r in metrics if r["Metric"] == metric), "—")
            row[label] = value
        rows.append(row)

    return columns, rows


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
        display_strategies = all_strategies
        display_columns = [("All Strategies", all_strategies)]
    else:
        display_strategies = [s for s in selected_strategies if s in all_strategies]
        display_columns = [(s, [s]) for s in display_strategies]

    columns = [{"name": "Metric", "id": "Metric"}] + [
        {"name": label, "id": label} for label, _ in display_columns
    ]

    metrics_by_column: Dict[str, List[dict]] = {}
    for label, strategy_list in display_columns:
        sp = compute_series(portfolio_df, strategy_list, DEFAULT_INITIAL_CAPITAL)
        metrics_by_column[label] = compute_metrics(sp, rf_annual=DEFAULT_RF)

    metric_order = [row["Metric"] for row in next(iter(metrics_by_column.values()), [])]
    rows: List[dict] = []
    for metric in metric_order:
        row = {"Metric": metric}
        for label, metrics in metrics_by_column.items():
            value = next((r["Value"] for r in metrics if r["Metric"] == metric), "—")
            row[label] = value
        rows.append(row)

    return columns, rows


def register_callbacks(app: dash.Dash, strategies: Dict[str, pd.DataFrame]) -> None:
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

    Args:
        app:        The Dash application instance.
        strategies: Mapping of strategy name to its loaded PnL DataFrame.
    """
    strategy_names = list(strategies.keys())

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

        if trig_id == "equity-range-btn":
            equity_range = next_range_key(equity_range)
        elif trig_id == "drawdown-range-btn":
            drawdown_range = next_range_key(drawdown_range)
        elif trig_id == "metrics-range-btn":
            metrics_range = next_range_key(metrics_range)
        elif trig_id == "custom-range-btn":
            custom_range = next_range_key(custom_range)
        elif trig_id == "btn-cycle-range":
            next_range = next_range_key(equity_range)
            equity_range = next_range
            drawdown_range = next_range
            metrics_range = next_range
            custom_range = next_range

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
        Input("strategy-radio", "value"),
        Input("home-radio", "value"),
        Input("btn-reset-layout", "n_clicks"),
        State("store-selected-strategy", "data"),
    )
    def set_strategy(strategy, home_strategy, reset_clicks, current):
        """Synchronise the selected strategy between sidebar radios and store."""
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
        return selected, strategy_value, home_value

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
        Input("store-selected-strategy", "data"),
        State({"type": "product-btn", "value": dash.ALL}, "id"),
        State("store-selected-products", "data"),
        prevent_initial_call=True,
    )
    def set_products(n_clicks_list, strategy, ids, current):
        """Handle product pill clicks: toggle individual or select ALL."""
        if strategy not in strategies and strategy != "Portfolio":
            return ["ALL"]

        ctx = callback_context
        if not ctx.triggered:
            return current

        trig_id = ctx.triggered_id
        if trig_id == "store-selected-strategy":
            return ["ALL"]

        if isinstance(trig_id, dict) and trig_id.get("type") == "product-btn":
            clicked = trig_id.get("value")
        else:
            return current

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
    # Main dashboard update (figures + metrics)
    # ================================================================

    @app.callback(
        Output("header-title", "children"),
        Output("equity-panel-title", "children"),
        Output("custom-panel-title", "children"),
        Output("drawdown-panel-title", "children"),
        Output("equity-modal-title", "children"),
        Output("custom-analytics-modal-title", "children"),
        Output("drawdown-modal-title", "children"),
        Output("equity-graph", "figure"),
        Output("equity-modal-graph", "figure"),
        Output("drawdown-graph", "figure"),
        Output("drawdown-modal-graph", "figure"),
        Output("metrics-table", "columns"),
        Output("metrics-table", "data"),
        Output("metrics-modal-table", "columns"),
        Output("metrics-modal-table", "data"),
        Output("custom-graph", "figure"),
        Output("custom-analytics-modal-graph", "figure"),
        Input("store-selected-strategy", "data"),
        Input("store-selected-products", "data"),
        Input("custom-tabs", "active_tab"),
        Input("layout-radio", "value"),
        Input("store-equity-range", "data"),
        Input("store-drawdown-range", "data"),
        Input("store-metrics-range", "data"),
        Input("store-custom-range", "data"),
        Input("store-theme", "data"),
    )
    def update_dashboard(
        strategy,
        selected_products,
        active_tab,
        layout_value,
        equity_range,
        drawdown_range,
        metrics_range,
        custom_range,
        theme,
    ):
        """Recompute all figures, titles, and metric tables for the dashboard.

        This is the primary callback -- it fires whenever the strategy,
        product selection, tab, layout mode, date range, or theme changes.
        It returns 17 outputs covering panel titles, graph figures, and
        DataTable columns/data for both inline and modal views.
        """
        layout_value = layout_value or "default"
        combine_drawdown = layout_value == "focused"
        if layout_value == "analytics":
            equity_title = "Rolling Sharpe"
            custom_title = "Seasonality"
            drawdown_title = "Rolling Correlation"
        elif layout_value == "focused":
            equity_title = "Equity & Drawdown Curve"
            custom_title = "Custom Analytics"
            drawdown_title = "Drawdown"
        else:
            equity_title = "Equity Curve"
            custom_title = "Custom Analytics"
            drawdown_title = "Drawdown"

        if strategy == "Portfolio":
            selected_products = selected_products or ["ALL"]
            all_strategies = [s for s in strategy_names if s in strategies]
            selected_strategies = (
                all_strategies if "ALL" in selected_products else [s for s in selected_products if s in strategies]
            )
            if not selected_strategies:
                selected_strategies = all_strategies

            portfolio_df = portfolio_dataframe(strategies, selected_strategies)
            if portfolio_df.empty:
                placeholder_title = "Portfolio view"
                subtitle = "Add your portfolio data to explore performance and analytics."
                eq_fig = placeholder_figure(placeholder_title, subtitle)
                dd_fig = placeholder_figure("Drawdowns will display once data is connected.")
                custom_fig = placeholder_figure("Custom analytics will appear here.")
                apply_plotly_theme(eq_fig, theme)
                apply_plotly_theme(dd_fig, theme)
                apply_plotly_theme(custom_fig, theme)
                return (
                    "Portfolio",
                    equity_title,
                    custom_title,
                    drawdown_title,
                    equity_title,
                    custom_title,
                    drawdown_title,
                    eq_fig,
                    eq_fig,
                    dd_fig,
                    dd_fig,
                    [],
                    [],
                    [],
                    [],
                    custom_fig,
                    custom_fig,
                )

            equity_df = filter_df_by_range(portfolio_df, equity_range)
            drawdown_df = filter_df_by_range(portfolio_df, drawdown_range)
            drawdown_df_for_equity = equity_df if combine_drawdown else drawdown_df
            metrics_df = filter_df_by_range(portfolio_df, metrics_range)
            custom_df = filter_df_by_range(portfolio_df, custom_range)

            if "ALL" in selected_products:
                equity_series = {
                    "All Strategies": compute_series(
                        equity_df,
                        selected_strategies,
                        DEFAULT_INITIAL_CAPITAL,
                    )
                }
                drawdown_series = {
                    "All Strategies": compute_series(
                        drawdown_df,
                        selected_strategies,
                        DEFAULT_INITIAL_CAPITAL,
                    )
                }
                equity_drawdown_series = {
                    "All Strategies": compute_series(
                        drawdown_df_for_equity,
                        selected_strategies,
                        DEFAULT_INITIAL_CAPITAL,
                    )
                }
            else:
                equity_series = {
                    s: compute_series(equity_df, [s], DEFAULT_INITIAL_CAPITAL) for s in selected_strategies
                }
                drawdown_series = {
                    s: compute_series(drawdown_df, [s], DEFAULT_INITIAL_CAPITAL) for s in selected_strategies
                }
                equity_drawdown_series = {
                    s: compute_series(drawdown_df_for_equity, [s], DEFAULT_INITIAL_CAPITAL)
                    for s in selected_strategies
                }

            eq_fig = equity_figure(
                equity_series,
                title="",
                drawdown_series=equity_drawdown_series if combine_drawdown else None,
            )
            dd_fig = drawdown_figure(drawdown_series, title="")

            table_columns, table_data = build_portfolio_metrics_table(
                selected_products,
                all_strategies,
                metrics_df,
            )

            show_individuals = "ALL" not in selected_products
            show_aggregate = "ALL" in selected_products
            if layout_value == "analytics":
                eq_fig = rolling_sharpe_figure(
                    equity_df,
                    selected_strategies,
                    DEFAULT_INITIAL_CAPITAL,
                    DEFAULT_ROLL_WINDOW,
                    title="",
                    include_individuals=show_individuals,
                    include_aggregate=show_aggregate,
                )
                custom_fig = seasonality_figure(custom_df, selected_strategies, title="")
                dd_fig = rolling_correlation_figure(
                    drawdown_df,
                    selected_strategies,
                    DEFAULT_INITIAL_CAPITAL,
                    DEFAULT_ROLL_WINDOW,
                    title="",
                )
            elif active_tab == "tab-roll":
                custom_fig = rolling_sharpe_figure(
                    custom_df,
                    selected_strategies,
                    DEFAULT_INITIAL_CAPITAL,
                    DEFAULT_ROLL_WINDOW,
                    title="",
                    include_individuals=show_individuals,
                    include_aggregate=show_aggregate,
                )
            elif active_tab == "tab-season":
                custom_fig = seasonality_figure(custom_df, selected_strategies, title="")
            else:
                custom_fig = rolling_correlation_figure(
                    custom_df,
                    selected_strategies,
                    DEFAULT_INITIAL_CAPITAL,
                    DEFAULT_ROLL_WINDOW,
                    title="",
                )

            for fig in [eq_fig, dd_fig, custom_fig]:
                fig.update_layout(
                    autosize=True,
                    margin=dict(l=14, r=14, t=40, b=22),
                )
                apply_plotly_theme(fig, theme)

            return (
                "Portfolio",
                equity_title,
                custom_title,
                drawdown_title,
                equity_title,
                custom_title,
                drawdown_title,
                eq_fig,
                eq_fig,
                dd_fig,
                dd_fig,
                table_columns,
                table_data,
                table_columns,
                table_data,
                custom_fig,
                custom_fig,
            )

        df = strategies.get(strategy)
        all_products = products_for_strategy(strategies, strategy)

        if df is None:
            placeholder_title = "Portfolio view"
            subtitle = "Add your portfolio data to explore performance and analytics."
            eq_fig = placeholder_figure(placeholder_title, subtitle)
            dd_fig = placeholder_figure("Drawdowns will display once data is connected.")
            custom_fig = placeholder_figure("Custom analytics will appear here.")
            apply_plotly_theme(eq_fig, theme)
            apply_plotly_theme(dd_fig, theme)
            apply_plotly_theme(custom_fig, theme)
            return (
                "Portfolio",
                equity_title,
                custom_title,
                drawdown_title,
                equity_title,
                custom_title,
                drawdown_title,
                eq_fig,
                eq_fig,
                dd_fig,
                dd_fig,
                [],
                [],
                [],
                [],
                custom_fig,
                custom_fig,
            )

        selected_products = selected_products or ["ALL"]
        products = all_products if ("ALL" in selected_products) else [p for p in selected_products if p in all_products]

        equity_df = filter_df_by_range(df, equity_range)
        drawdown_df = filter_df_by_range(df, drawdown_range)
        drawdown_df_for_equity = equity_df if combine_drawdown else drawdown_df
        metrics_df = filter_df_by_range(df, metrics_range)
        custom_df = filter_df_by_range(df, custom_range)

        if "ALL" in selected_products:
            equity_series = {"All Products": compute_series(equity_df, products, DEFAULT_INITIAL_CAPITAL)}
            drawdown_series = {"All Products": compute_series(drawdown_df, products, DEFAULT_INITIAL_CAPITAL)}
            equity_drawdown_series = {
                "All Products": compute_series(drawdown_df_for_equity, products, DEFAULT_INITIAL_CAPITAL)
            }
        else:
            equity_series = {
                format_product_label(p): compute_series(equity_df, [p], DEFAULT_INITIAL_CAPITAL)
                for p in products
            }
            drawdown_series = {
                format_product_label(p): compute_series(drawdown_df, [p], DEFAULT_INITIAL_CAPITAL)
                for p in products
            }
            equity_drawdown_series = {
                format_product_label(p): compute_series(drawdown_df_for_equity, [p], DEFAULT_INITIAL_CAPITAL)
                for p in products
            }
        title_text = f"{strategy} Trading Strategy"

        eq_fig = equity_figure(
            equity_series,
            title="",
            drawdown_series=equity_drawdown_series if combine_drawdown else None,
        )
        dd_fig = drawdown_figure(drawdown_series, title="")

        table_columns, table_data = build_metrics_table(metrics_df, selected_products, all_products)

        show_individuals = "ALL" not in selected_products
        show_aggregate = "ALL" in selected_products
        if layout_value == "analytics":
            eq_fig = rolling_sharpe_figure(
                equity_df,
                products,
                DEFAULT_INITIAL_CAPITAL,
                DEFAULT_ROLL_WINDOW,
                title="",
                include_individuals=show_individuals,
                include_aggregate=show_aggregate,
            )
            custom_fig = seasonality_figure(custom_df, products, title="")
            dd_fig = rolling_correlation_figure(
                drawdown_df,
                products,
                DEFAULT_INITIAL_CAPITAL,
                DEFAULT_ROLL_WINDOW,
                title="",
            )
        elif active_tab == "tab-roll":
            custom_fig = rolling_sharpe_figure(
                custom_df,
                products,
                DEFAULT_INITIAL_CAPITAL,
                DEFAULT_ROLL_WINDOW,
                title="",
                include_individuals=show_individuals,
                include_aggregate=show_aggregate,
            )
        elif active_tab == "tab-season":
            custom_fig = seasonality_figure(custom_df, products, title="")
        else:
            custom_fig = rolling_correlation_figure(
                custom_df,
                products,
                DEFAULT_INITIAL_CAPITAL,
                DEFAULT_ROLL_WINDOW,
                title="",
            )

        for fig in [eq_fig, dd_fig, custom_fig]:
            fig.update_layout(
                autosize=True,
                margin=dict(l=14, r=14, t=40, b=22),
            )
            apply_plotly_theme(fig, theme)

        return (
            title_text,
            equity_title,
            custom_title,
            drawdown_title,
            equity_title,
            custom_title,
            drawdown_title,
            eq_fig,
            eq_fig,
            dd_fig,
            dd_fig,
            table_columns,
            table_data,
            table_columns,
            table_data,
            custom_fig,
            custom_fig,
        )
