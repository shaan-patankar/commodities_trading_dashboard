"""Dash layout definition for the trading strategy dashboard.

Constructs the full component tree including the top bar, navigation
sidebar, settings sidebar, the four main panels (equity, custom analytics,
drawdown, key metrics), and all full-screen modal overlays.

This module is purely declarative -- it contains no callback logic.
"""

from __future__ import annotations

from typing import List

from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc

from dashboard.config import GRAPH_CONFIG, LAYOUT_OPTIONS, PANEL_KEYS, SETTINGS_GEAR_SVG
from dashboard.table_styles import metrics_cell_style, metrics_header_style
from dashboard.utils import format_product_label


# ---------------------------------------------------------------------------
# Reusable component helpers
# ---------------------------------------------------------------------------

def product_button(label: str, value: str) -> dbc.Button:
    """Create a product-filter pill button for the top bar.

    Args:
        label: Display text for the button.
        value: Internal value used in pattern-matching callbacks.

    Returns:
        A ``dbc.Button`` configured as a toggle pill.
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


def var_alloc_row(product: str, value: float | None = None) -> html.Div:
    """Build one product allocation row for the VaR config modal.

    Args:
        product: Product column name (pattern-matched into the input id).
        value:   Pre-seeded percentage (per-strategy memory), or ``None``.

    Returns:
        A row with a label, a numeric ``%`` input, and an inline error slot.
    """
    return html.Div(
        className="var-alloc-row",
        children=[
            html.Span(format_product_label(product), className="var-alloc-label"),
            html.Div(
                className="var-alloc-input-group",
                children=[
                    dcc.Input(
                        id={"type": "var-alloc-input", "product": product},
                        type="number",
                        min=0,
                        max=100,
                        step="any",
                        value=value,
                        className="var-alloc-input",
                        debounce=True,
                        placeholder="0",
                    ),
                    html.Span("%", className="var-alloc-pct"),
                ],
            ),
            html.Span("", id={"type": "var-alloc-error", "product": product}, className="var-row-error"),
        ],
    )


def build_layout(strategy_names: List[str], default_strategy: str, products: List[str]) -> html.Div:
    """Construct the complete Dash layout tree.

    Args:
        strategy_names:   List of available strategy display names.
        default_strategy: Strategy to select on initial load.
        products:         Product columns for the default strategy.

    Returns:
        The root ``html.Div`` containing all stores, sidebars, panels,
        and modal overlays.
    """
    home_options = [{"label": "Portfolio", "value": "Portfolio"}]
    strategy_options = [{"label": s, "value": s} for s in strategy_names]
    strategy_value = default_strategy if default_strategy in strategy_names else None
    home_value = "Portfolio" if default_strategy == "Portfolio" else None
    default_csv = strategy_names[0] if strategy_names else None
    theme_options = [
        ("deep", "Dark Mode"),
        ("light", "Light Mode"),
    ]
    csv_buttons = [
        dbc.Button(
            label,
            id={"type": "csv-open-btn", "value": label},
            color="secondary",
            outline=True,
            className="settings-option-btn",
            n_clicks=0,
        )
        for label in strategy_names
    ]

    return html.Div(
        id="app-root",
        className="app-root",
        children=[
            dcc.Store(id="store-selected-strategy", data=default_strategy),
            dcc.Store(id="store-selected-products", data=["ALL"]),
            dcc.Store(id="store-equity-modal-open", data=False),
            dcc.Store(id="store-drawdown-modal-open", data=False),
            dcc.Store(id="store-metrics-modal-open", data=False),
            dcc.Store(id="store-custom-analytics-modal-open", data=False),
            dcc.Store(id="store-equity-range", data="All"),
            dcc.Store(id="store-drawdown-range", data="All"),
            dcc.Store(id="store-metrics-range", data="All"),
            dcc.Store(id="store-custom-range", data="All"),
            dcc.Store(id="store-last-deselected", data=None),
            dcc.Store(id="store-selected-csv", data=default_csv),
            dcc.Store(id="store-csv-modal-open", data=False),
            dcc.Store(id="store-theme", data="deep"),
            # VaR scaling: config keyed per strategy -> {strategy: {total_var, allocations, active}}
            dcc.Store(id="store-var-config", data={}),
            dcc.Store(id="store-var-modal-open", data=False),
            dcc.Store(id="store-var-expand-open", data=False),
            html.Div(
                className="topbar",
                children=[
                    dbc.Button(
                        "☰",
                        id="btn-open-sidebar",
                        color="dark",
                        className="me-2 sidebar-toggle-btn",
                        n_clicks=0,
                    ),
                    html.Div(
                        className="topbar-title",
                        children=[
                            html.Div(
                                f"{default_strategy} Trading Strategy",
                                id="header-title",
                                className="h5 fw-bold m-0",
                            ),
                        ],
                    ),
                    html.Div(className="topbar-spacer"),
                    html.Div(
                        className="topbar-products",
                        children=[
                            html.Div(
                                id="product-buttons",
                                children=[product_button("All", "ALL")]
                                + [product_button(format_product_label(p), p) for p in products],
                            ),
                        ],
                    ),
                    dbc.Button(
                        html.Span(
                            html.Img(
                                src=SETTINGS_GEAR_SVG,
                                className="settings-gear-svg",
                                alt="Settings",
                            ),
                            className="settings-gear-icon",
                        ),
                        id="btn-open-settings",
                        color="dark",
                        className="sidebar-toggle-btn settings-toggle-btn",
                        n_clicks=0,
                    ),
                ],
            ),
            dbc.Offcanvas(
                id="sidebar",
                title=html.Div("Trading Strategy Dashboard", className="fw-bold m-0"),
                is_open=False,
                placement="start",
                backdrop=True,
                close_button=False,
                className="sidebar",
                children=[
                    html.Div(
                        className="sidebar-body",
                        children=[
                            html.Div(className="sidebar-spacer"),
                            html.Div(
                                className="sidebar-section",
                                children=[
                                    html.Div("Home", className="sidebar-section-title"),
                                    dbc.RadioItems(
                                        id="home-radio",
                                        options=home_options,
                                        value=home_value,
                                        className="strategy-radio",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="sidebar-section",
                                children=[
                                    html.Div("Strategies", className="sidebar-section-title"),
                                    dbc.RadioItems(
                                        id="strategy-radio",
                                        options=strategy_options,
                                        value=strategy_value,
                                        className="strategy-radio",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dbc.Offcanvas(
                id="settings-sidebar",
                title=html.Div("Settings", className="fw-bold m-0"),
                is_open=False,
                placement="end",
                backdrop=True,
                close_button=False,
                className="sidebar settings-sidebar",
                children=[
                    html.Div(
                        className="sidebar-body",
                        children=[
                            html.Div(className="sidebar-spacer"),
                            html.Div(
                                className="sidebar-section",
                                children=[
                                    html.Div("Layout", className="sidebar-section-title"),
                                    dbc.RadioItems(
                                        id="layout-radio",
                                        options=LAYOUT_OPTIONS,
                                        value="default",
                                        className="settings-radio",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="sidebar-section",
                                children=[
                                    html.Div("Panels", className="sidebar-section-title"),
                                    dbc.Checklist(
                                        id="panel-visibility",
                                        options=[
                                            {"label": "Equity Curve", "value": "equity"},
                                            {"label": "Custom Analytics", "value": "custom"},
                                            {"label": "Drawdown", "value": "drawdown"},
                                            {"label": "Key Metrics", "value": "metrics"},
                                        ],
                                        value=PANEL_KEYS.copy(),
                                        className="settings-checklist",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="sidebar-section",
                                children=[
                                    html.Div("Range", className="sidebar-section-title"),
                                    dbc.Button(
                                        "All Panels: All",
                                        id="btn-cycle-range",
                                        color="secondary",
                                        outline=True,
                                        className="settings-option-btn",
                                        n_clicks=0,
                                    ),
                                ],
                            ),
                            html.Div(
                                className="sidebar-section",
                                children=[
                                    html.Div("Theme", className="sidebar-section-title"),
                                    html.Div(
                                        className="settings-csv-buttons",
                                        children=[
                                            dbc.Button(
                                                label,
                                                id={"type": "theme-btn", "value": value},
                                                color="secondary",
                                                outline=True,
                                                className="settings-option-btn",
                                                n_clicks=0,
                                            )
                                            for value, label in theme_options
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="sidebar-section",
                                children=[
                                    html.Div("CSV Explorer", className="sidebar-section-title"),
                                    html.Div(className="settings-csv-buttons", children=csv_buttons),
                                ],
                            ),
                            html.Div(
                                className="sidebar-section",
                                children=[
                                    html.Div("Reset", className="sidebar-section-title"),
                                    dbc.Button(
                                        "Reset Layout",
                                        id="btn-reset-layout",
                                        color="secondary",
                                        outline=True,
                                        className="settings-option-btn",
                                        n_clicks=0,
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="body-grid",
                className="body-grid",
                children=[
                    html.Div(
                        id="panel-equity",
                        className="panel panel-big",
                        children=[
                            dbc.Card(
                                className="card-dark",
                                children=[
                                    dbc.CardHeader(
                                        className="card-header-dark",
                                        children=[
                                            html.Div(
                                                "Equity Curve",
                                                id="equity-panel-title",
                                                className="fw-semibold",
                                            ),
                                            html.Div(
                                                className="card-header-actions",
                                                children=[
                                                    dbc.Button(
                                                        "All",
                                                        id="equity-range-btn",
                                                        color="secondary",
                                                        outline=True,
                                                        className="range-toggle-btn",
                                                        n_clicks=0,
                                                        size="sm",
                                                    ),
                                                    dbc.Button(
                                                        "⛶",
                                                        id="open-equity-modal",
                                                        color="secondary",
                                                        outline=True,
                                                        className="equity-expand-btn",
                                                        n_clicks=0,
                                                        size="sm",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    dbc.CardBody(
                                        className="card-body-tight",
                                        children=[dcc.Graph(id="equity-graph", config=GRAPH_CONFIG, className="graph")],
                                    ),
                                ],
                            )
                        ],
                    ),
                    html.Div(
                        id="panel-custom",
                        className="panel panel-small",
                        children=[
                            dbc.Card(
                                className="card-dark",
                                children=[
                                    dbc.CardHeader(
                                        className="card-header-dark",
                                        children=[
                                            html.Div(
                                                "Custom Analytics",
                                                id="custom-panel-title",
                                                className="fw-semibold",
                                            ),
                                            html.Div(
                                                className="card-header-actions",
                                                children=[
                                                    dbc.Button(
                                                        "All",
                                                        id="custom-range-btn",
                                                        color="secondary",
                                                        outline=True,
                                                        className="range-toggle-btn",
                                                        n_clicks=0,
                                                        size="sm",
                                                    ),
                                                    dbc.Button(
                                                        "⛶",
                                                        id="open-custom-analytics-modal",
                                                        color="secondary",
                                                        outline=True,
                                                        className="custom-analytics-expand-btn",
                                                        n_clicks=0,
                                                        size="sm",
                                                    ),
                                                    dbc.Button(
                                                        "⛶",
                                                        id="open-var-expand-modal",
                                                        color="secondary",
                                                        outline=True,
                                                        className="custom-analytics-expand-btn",
                                                        n_clicks=0,
                                                        size="sm",
                                                        style={"display": "none"},
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    dbc.CardBody(
                                        className="card-body-tight",
                                        children=[
                                            html.Div(
                                                className="custom-tabs-container",
                                                children=[
                                                    dbc.Tabs(
                                                        id="custom-tabs",
                                                        active_tab="tab-roll",
                                                        className="custom-tabs",
                                                        children=[
                                                            dbc.Tab(label="Rolling Sharpe", tab_id="tab-roll"),
                                                            dbc.Tab(label="Seasonality", tab_id="tab-season"),
                                                            dbc.Tab(label="Rolling Correlation", tab_id="tab-corr"),
                                                            dbc.Tab(label="VaR Scaling", tab_id="tab-var"),
                                                        ],
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                id="custom-graph-wrapper",
                                                className="custom-graph-wrapper mt-2",
                                                children=[
                                                    dcc.Graph(
                                                        id="custom-graph",
                                                        config=GRAPH_CONFIG,
                                                        className="graph",
                                                        clear_on_unhover=True,
                                                    )
                                                ],
                                            ),
                                            html.Div(
                                                id="var-summary-wrapper",
                                                className="var-summary-wrapper mt-2",
                                                style={"display": "none"},
                                                children=[
                                                    html.Div(id="var-notice", className="var-notice"),
                                                    html.Div(
                                                        className="metrics-table-wrapper var-summary-table-wrapper",
                                                        children=[
                                                            dash_table.DataTable(
                                                                id="var-summary-table",
                                                                columns=[],
                                                                data=[],
                                                                style_as_list_view=True,
                                                                fill_width=True,
                                                                style_table={
                                                                    "maxHeight": "100%",
                                                                    "minHeight": "0",
                                                                    "width": "100%",
                                                                    "overflowY": "auto",
                                                                    "overflowX": "auto",
                                                                    "border": "0px",
                                                                    "borderRadius": "14px",
                                                                },
                                                                style_cell=metrics_cell_style(),
                                                                style_header=metrics_header_style(),
                                                                page_action="none",
                                                            )
                                                        ],
                                                    ),
                                                    dbc.Button(
                                                        "Configure VaR…",
                                                        id="btn-var-open-from-tab",
                                                        color="secondary",
                                                        outline=True,
                                                        className="settings-option-btn var-configure-btn",
                                                        n_clicks=0,
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                    html.Div(
                        id="panel-drawdown",
                        className="panel panel-big",
                        children=[
                            dbc.Card(
                                className="card-dark",
                                children=[
                                    dbc.CardHeader(
                                        className="card-header-dark",
                                        children=[
                                            html.Div(
                                                "Drawdown",
                                                id="drawdown-panel-title",
                                                className="fw-semibold",
                                            ),
                                            html.Div(
                                                className="card-header-actions",
                                                children=[
                                                    dbc.Button(
                                                        "All",
                                                        id="drawdown-range-btn",
                                                        color="secondary",
                                                        outline=True,
                                                        className="range-toggle-btn",
                                                        n_clicks=0,
                                                        size="sm",
                                                    ),
                                                    dbc.Button(
                                                        "⛶",
                                                        id="open-drawdown-modal",
                                                        color="secondary",
                                                        outline=True,
                                                        className="drawdown-expand-btn",
                                                        n_clicks=0,
                                                        size="sm",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    dbc.CardBody(
                                        className="card-body-tight",
                                        children=[
                                            dcc.Graph(
                                                id="drawdown-graph",
                                                config=GRAPH_CONFIG,
                                                className="graph",
                                                clear_on_unhover=True,
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                    html.Div(
                        id="panel-metrics",
                        className="panel panel-small",
                        children=[
                            dbc.Card(
                                className="card-dark",
                                children=[
                                    dbc.CardHeader(
                                        className="card-header-dark",
                                        children=[
                                            html.Div("Key Metrics", className="fw-semibold"),
                                            html.Div(
                                                className="card-header-actions",
                                                children=[
                                                    dbc.Button(
                                                        "All",
                                                        id="metrics-range-btn",
                                                        color="secondary",
                                                        outline=True,
                                                        className="range-toggle-btn",
                                                        n_clicks=0,
                                                        size="sm",
                                                    ),
                                                    dbc.Button(
                                                        "⛶",
                                                        id="open-metrics-modal",
                                                        color="secondary",
                                                        outline=True,
                                                        className="metrics-expand-btn",
                                                        n_clicks=0,
                                                        size="sm",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    dbc.CardBody(
                                        className="card-body-tight",
                                        children=[
                                            html.Div(
                                                className="metrics-table-wrapper",
                                                children=[
                                                    dash_table.DataTable(
                                                        id="metrics-table",
                                                        columns=[{"name": "Metric", "id": "Metric"}],
                                                        data=[],
                                                        style_as_list_view=True,
                                                        fixed_rows={"headers": True},
                                                        fill_width=True,
                                                        style_table={
                                                            "maxHeight": "100%",
                                                            "minHeight": "0",
                                                            "width": "100%",
                                                            "minWidth": "max-content",
                                                            "backgroundColor": "transparent",
                                                            "overflowY": "auto",
                                                            "overflowX": "auto",
                                                            "border": "0px",
                                                            "borderRadius": "14px",
                                                        },
                                                        style_cell=metrics_cell_style(),
                                                        style_cell_conditional=[
                                                            {
                                                                "if": {"column_id": "Metric"},
                                                                "textAlign": "left",
                                                                "fontFamily": "'Inter', 'Segoe UI', system-ui",
                                                                "width": "32%",
                                                                "minWidth": "180px",
                                                                "maxWidth": "320px",
                                                            },
                                                        ],
                                                        style_data_conditional=[
                                                            {"if": {"column_id": "Metric"}, "textAlign": "left"},
                                                            {
                                                                "if": {"state": "active"},
                                                                "backgroundColor": "transparent",
                                                                "border": "0px",
                                                                "borderBottom": "0px",
                                                                "boxShadow": "none",
                                                            },
                                                            {
                                                                "if": {"state": "selected"},
                                                                "backgroundColor": "transparent",
                                                                "border": "0px",
                                                                "borderBottom": "0px",
                                                                "boxShadow": "none",
                                                            },
                                                        ],
                                                        style_header=metrics_header_style(),
                                                        style_header_conditional=[
                                                            {"if": {"column_id": "Metric"}, "textAlign": "left"},
                                                        ],
                                                        page_action="none",
                                                    )
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
            html.Div(
                id="equity-modal-overlay",
                className="equity-modal-overlay",
                children=[
                    html.Div(
                        id="equity-modal-backdrop",
                        className="modal-backdrop",
                        n_clicks=0,
                    ),
                    html.Div(
                        className="equity-modal",
                        children=[
                            html.Div(
                                className="equity-modal-header",
                                children=[
                                    html.Div(
                                        "Equity Curve",
                                        id="equity-modal-title",
                                        className="fw-semibold",
                                    ),
                                    dbc.Button(
                                        "×",
                                        id="close-equity-modal",
                                        color="secondary",
                                        outline=True,
                                        className="equity-close-btn",
                                        n_clicks=0,
                                        size="sm",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="equity-modal-body",
                                children=[
                                    dcc.Graph(
                                        id="equity-modal-graph",
                                        config=GRAPH_CONFIG,
                                        className="graph",
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="drawdown-modal-overlay",
                className="drawdown-modal-overlay",
                children=[
                    html.Div(
                        id="drawdown-modal-backdrop",
                        className="modal-backdrop",
                        n_clicks=0,
                    ),
                    html.Div(
                        className="drawdown-modal",
                        children=[
                            html.Div(
                                className="drawdown-modal-header",
                                children=[
                                    html.Div(
                                        "Drawdown",
                                        id="drawdown-modal-title",
                                        className="fw-semibold",
                                    ),
                                    dbc.Button(
                                        "×",
                                        id="close-drawdown-modal",
                                        color="secondary",
                                        outline=True,
                                        className="drawdown-close-btn",
                                        n_clicks=0,
                                        size="sm",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="drawdown-modal-body",
                                children=[
                                    dcc.Graph(
                                        id="drawdown-modal-graph",
                                        config=GRAPH_CONFIG,
                                        className="graph",
                                        clear_on_unhover=True,
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="metrics-modal-overlay",
                className="metrics-modal-overlay",
                children=[
                    html.Div(
                        id="metrics-modal-backdrop",
                        className="modal-backdrop",
                        n_clicks=0,
                    ),
                    html.Div(
                        className="metrics-modal",
                        children=[
                            html.Div(
                                className="metrics-modal-header",
                                children=[
                                    html.Div("Key Metrics", className="fw-semibold"),
                                    dbc.Button(
                                        "×",
                                        id="close-metrics-modal",
                                        color="secondary",
                                        outline=True,
                                        className="metrics-close-btn",
                                        n_clicks=0,
                                        size="sm",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="metrics-modal-body",
                                children=[
                                    html.Div(
                                        className="metrics-table-wrapper metrics-modal-table",
                                        children=[
                                            dash_table.DataTable(
                                                id="metrics-modal-table",
                                                columns=[{"name": "Metric", "id": "Metric"}],
                                                data=[],
                                                style_as_list_view=True,
                                                fixed_rows={"headers": True},
                                                fill_width=True,
                                                style_table={
                                                    "maxHeight": "100%",
                                                    "minHeight": "0",
                                                    "width": "100%",
                                                    "minWidth": "max-content",
                                                    "backgroundColor": "transparent",
                                                    "overflowY": "auto",
                                                    "overflowX": "auto",
                                                    "border": "0px",
                                                    "borderRadius": "18px",
                                                },
                                                style_cell=metrics_cell_style(padding="12px 14px", min_width="140px"),
                                                style_cell_conditional=[
                                                    {
                                                        "if": {"column_id": "Metric"},
                                                        "textAlign": "left",
                                                        "fontFamily": "'Inter', 'Segoe UI', system-ui",
                                                        "width": "32%",
                                                        "minWidth": "200px",
                                                        "maxWidth": "360px",
                                                    },
                                                ],
                                                style_data_conditional=[
                                                    {"if": {"column_id": "Metric"}, "textAlign": "left"},
                                                    {
                                                        "if": {"state": "active"},
                                                        "backgroundColor": "transparent",
                                                        "border": "0px",
                                                        "borderBottom": "0px",
                                                        "boxShadow": "none",
                                                    },
                                                    {
                                                        "if": {"state": "selected"},
                                                        "backgroundColor": "transparent",
                                                        "border": "0px",
                                                        "borderBottom": "0px",
                                                        "boxShadow": "none",
                                                    },
                                                ],
                                                style_header=metrics_header_style(header_bg_dark="rgba(255,255,255,0.05)"),
                                                style_header_conditional=[
                                                    {"if": {"column_id": "Metric"}, "textAlign": "left"},
                                                ],
                                                page_action="none",
                                            )
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="custom-analytics-modal-overlay",
                className="custom-analytics-modal-overlay",
                children=[
                    html.Div(
                        id="custom-analytics-modal-backdrop",
                        className="modal-backdrop",
                        n_clicks=0,
                    ),
                    html.Div(
                        className="custom-analytics-modal",
                        children=[
                            html.Div(
                                className="custom-analytics-modal-header",
                                children=[
                                    html.Div(
                                        "Custom Analytics",
                                        id="custom-analytics-modal-title",
                                        className="fw-semibold",
                                    ),
                                    dbc.Button(
                                        "×",
                                        id="close-custom-analytics-modal",
                                        color="secondary",
                                        outline=True,
                                        className="custom-analytics-close-btn",
                                        n_clicks=0,
                                        size="sm",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="custom-analytics-modal-body",
                                children=[
                                    dcc.Graph(
                                        id="custom-analytics-modal-graph",
                                        config=GRAPH_CONFIG,
                                        className="graph",
                                        clear_on_unhover=True,
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="csv-modal-overlay",
                className="csv-modal-overlay",
                children=[
                    html.Div(
                        id="csv-modal-backdrop",
                        className="modal-backdrop",
                        n_clicks=0,
                    ),
                    html.Div(
                        className="csv-modal",
                        children=[
                            html.Div(
                                className="csv-modal-header",
                                children=[
                                    html.Div(id="csv-modal-title", className="fw-semibold"),
                                    dbc.Button(
                                        "×",
                                        id="close-csv-modal",
                                        color="secondary",
                                        outline=True,
                                        className="csv-close-btn",
                                        n_clicks=0,
                                        size="sm",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="csv-modal-meta text-muted small",
                                id="csv-modal-meta",
                            ),
                            html.Div(
                                className="csv-modal-body",
                                children=[
                                    html.Div(
                                        className="metrics-table-wrapper metrics-modal-table",
                                        children=[
                                            dash_table.DataTable(
                                                id="csv-modal-table",
                                                columns=[],
                                                data=[],
                                                style_as_list_view=True,
                                                fixed_rows={"headers": True},
                                                fill_width=True,
                                                style_table={
                                                    "maxHeight": "100%",
                                                    "minHeight": "0",
                                                    "width": "100%",
                                                    "minWidth": "max-content",
                                                    "backgroundColor": "transparent",
                                                    "overflowY": "auto",
                                                    "overflowX": "auto",
                                                    "border": "0px",
                                                    "borderRadius": "14px",
                                                },
                                                style_cell=metrics_cell_style(),
                                                style_cell_conditional=[
                                                    {
                                                        "if": {"column_id": "Date"},
                                                        "textAlign": "left",
                                                        "fontFamily": "'Inter', 'Segoe UI', system-ui",
                                                        "minWidth": "160px",
                                                    },
                                                ],
                                                style_data_conditional=[
                                                    {"if": {"column_id": "Date"}, "textAlign": "left"},
                                                    {
                                                        "if": {"state": "active"},
                                                        "backgroundColor": "transparent",
                                                        "border": "0px",
                                                        "borderBottom": "0px",
                                                        "boxShadow": "none",
                                                    },
                                                    {
                                                        "if": {"state": "selected"},
                                                        "backgroundColor": "transparent",
                                                        "border": "0px",
                                                        "borderBottom": "0px",
                                                        "boxShadow": "none",
                                                    },
                                                ],
                                                style_header=metrics_header_style(),
                                                style_header_conditional=[
                                                    {"if": {"column_id": "Date"}, "textAlign": "left"},
                                                ],
                                                page_action="none",
                                            )
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            # ---- VaR Scaling: configuration popup ----
            html.Div(
                id="var-modal-overlay",
                className="var-modal-overlay",
                children=[
                    html.Div(id="var-modal-backdrop", className="modal-backdrop", n_clicks=0),
                    html.Div(
                        className="var-modal",
                        children=[
                            html.Div(
                                className="var-modal-header",
                                children=[
                                    html.Div("VaR Scaling", id="var-modal-title", className="fw-semibold"),
                                    dbc.Button(
                                        "×",
                                        id="close-var-modal",
                                        color="secondary",
                                        outline=True,
                                        className="custom-analytics-close-btn",
                                        n_clicks=0,
                                        size="sm",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="var-modal-body",
                                children=[
                                    html.Div(
                                        "Size each product so its daily 95% VaR matches its share of "
                                        "the total budget. Sizing uses VaR ÷ (1.645 × 20-day return σ).",
                                        className="var-modal-subtitle",
                                    ),
                                    html.Div(
                                        className="var-total-group",
                                        children=[
                                            html.Label("Total VaR Allocation", className="var-total-label"),
                                            # Input is rendered here (seeded per strategy) to keep the
                                            # config<->value sync acyclic.
                                            html.Div(id="var-total-container", className="var-total-container"),
                                            html.Span("", id="var-total-error", className="var-total-error"),
                                        ],
                                    ),
                                    html.Div("Allocation by product", className="var-section-label"),
                                    html.Div(id="var-alloc-rows", className="var-alloc-rows"),
                                    html.Div(id="var-alloc-total", className="var-alloc-total"),
                                    html.Div(id="var-validation-msg", className="var-validation-msg"),
                                    html.Div(
                                        className="var-modal-actions",
                                        children=[
                                            dbc.Button(
                                                "Equal weight",
                                                id="btn-var-equal",
                                                color="secondary",
                                                outline=True,
                                                className="var-action-btn",
                                                n_clicks=0,
                                                size="sm",
                                            ),
                                            dbc.Button(
                                                "Normalize to 100%",
                                                id="btn-var-normalize",
                                                color="secondary",
                                                outline=True,
                                                className="var-action-btn",
                                                n_clicks=0,
                                                size="sm",
                                            ),
                                            dbc.Button(
                                                "Reset",
                                                id="btn-var-reset",
                                                color="secondary",
                                                outline=True,
                                                className="var-action-btn",
                                                n_clicks=0,
                                                size="sm",
                                            ),
                                            dbc.Button(
                                                "Apply",
                                                id="btn-var-apply",
                                                color="primary",
                                                className="var-apply-btn",
                                                n_clicks=0,
                                                size="sm",
                                                disabled=True,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="var-active-row",
                                        children=[
                                            # Switch is rendered here (seeded per strategy) to keep the
                                            # config<->value sync acyclic.
                                            html.Div(id="var-switch-container", className="var-switch-container"),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            # ---- VaR Scaling: full-screen results popup ----
            html.Div(
                id="var-expand-overlay",
                className="var-expand-overlay",
                children=[
                    html.Div(id="var-expand-backdrop", className="modal-backdrop", n_clicks=0),
                    html.Div(
                        className="var-expand-modal",
                        children=[
                            html.Div(
                                className="var-expand-modal-header",
                                children=[
                                    html.Div(
                                        "VaR Scaling — Results",
                                        id="var-expand-title",
                                        className="fw-semibold",
                                    ),
                                    dbc.Button(
                                        "×",
                                        id="close-var-expand-modal",
                                        color="secondary",
                                        outline=True,
                                        className="custom-analytics-close-btn",
                                        n_clicks=0,
                                        size="sm",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="var-expand-modal-body",
                                children=[
                                    html.Div(id="var-expand-notice", className="var-notice"),
                                    html.Div(
                                        className="metrics-table-wrapper var-summary-table-wrapper",
                                        children=[
                                            dash_table.DataTable(
                                                id="var-expand-table",
                                                columns=[],
                                                data=[],
                                                style_as_list_view=True,
                                                fill_width=True,
                                                style_table={
                                                    "maxHeight": "100%",
                                                    "minHeight": "0",
                                                    "width": "100%",
                                                    "overflowY": "auto",
                                                    "overflowX": "auto",
                                                    "border": "0px",
                                                    "borderRadius": "18px",
                                                },
                                                style_cell=metrics_cell_style(padding="12px 14px", min_width="140px"),
                                                style_header=metrics_header_style(header_bg_dark="rgba(255,255,255,0.05)"),
                                                page_action="none",
                                            )
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
