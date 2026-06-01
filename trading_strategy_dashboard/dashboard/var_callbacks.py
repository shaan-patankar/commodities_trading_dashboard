"""Dash callbacks for the VaR Scaling feature.

Kept separate from :mod:`dashboard.callbacks` so the new feature is
self-contained. Covers: the config popup and full-screen expand popup,
per-strategy memory of the config, live input validation with inline error
indication, the equal-weight / normalize helpers, persistence into
``store-var-config`` (keyed per strategy), and the VaR summary tables.

The actual dashboard rescaling lives in ``update_dashboard`` (which reads
``store-var-config``); this module handles everything around the inputs and the
summary/visibility of the VaR tab.
"""

from __future__ import annotations

import math
from typing import Dict, List

import dash
from dash import ALL, Input, Output, State, callback_context, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd

from dashboard.config import DEFAULT_TOTAL_VAR, RETURNS_FILES, VAR_ALLOC_TOLERANCE
from dashboard.layout import var_alloc_row
from dashboard.table_styles import metrics_cell_style, metrics_header_style
from dashboard.utils import format_product_label, valid_product_columns
from dashboard.var_scaling import (
    compute_var_scaled_frame,
    equal_weight_allocations,
    normalize_allocations,
    validate_var_config,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:,.2f}%"


def _fmt_cash(x: float) -> str:
    if x is None:
        return "—"
    try:
        if math.isnan(float(x)):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return f"{float(x):,.0f}"


_STRATEGY_COLUMNS = [
    {"name": "Product", "id": "product"},
    {"name": "σ (20d)", "id": "sigma"},
    {"name": "VaR Budget", "id": "budget"},
    {"name": "Notional", "id": "notional"},
    {"name": "Weight", "id": "weight"},
]

_PORTFOLIO_COLUMNS = [
    {"name": "Strategy", "id": "strategy"},
    {"name": "VaR", "id": "varon"},
    {"name": "Total VaR", "id": "total"},
    {"name": "Products", "id": "nprod"},
]


def _returns_filename(strategy: str) -> str:
    path = RETURNS_FILES.get(strategy)
    return path.name if path is not None else f"{strategy}_returns.csv"


def _total_input(value, disabled: bool = False) -> dcc.Input:
    """Build the Total VaR numeric input (rendered, so it can be seeded acyclically)."""
    return dcc.Input(
        id="var-total-input",
        type="number",
        min=0,
        step="any",
        value=value,
        disabled=disabled,
        className="var-total-input",
        debounce=True,
        placeholder="e.g. 10000",
    )


def _active_switch(value: bool, disabled: bool = False) -> dbc.Switch:
    """Build the VaR on/off switch (rendered, so it can be seeded acyclically)."""
    return dbc.Switch(
        id="var-active-switch",
        label="VaR scaling on",
        value=bool(value),
        disabled=disabled,
        className="var-active-switch",
    )


def _build_strategy_summary(strategy, strategies, returns, var_config):
    """Return (columns, rows, notice) for an individual strategy's VaR tab."""
    df = strategies.get(strategy)
    products = valid_product_columns(df) if df is not None else []
    cfg = (var_config or {}).get(strategy) or {}
    returns_df = returns.get(strategy)

    if df is None:
        return _STRATEGY_COLUMNS, [], "No data for this strategy."
    if returns_df is None:
        return (
            _STRATEGY_COLUMNS,
            [],
            f"No returns CSV found for {strategy}. Add data/{_returns_filename(strategy)} "
            f"(same product columns) to enable VaR scaling.",
        )

    notice = ""
    if not bool(cfg.get("active")):
        notice = "Preview at equal weight — VaR scaling is off. Click “Configure VaR…” to set and apply it."

    total_var = cfg.get("total_var") if cfg.get("total_var") is not None else DEFAULT_TOTAL_VAR
    allocations = cfg.get("allocations") or {}
    _, diag = compute_var_scaled_frame(df, returns_df, products, total_var, allocations)

    rows: List[dict] = []
    for p in products:
        latest = diag["latest"].get(p)
        if latest is None:
            rows.append({"product": format_product_label(p), "sigma": "—", "budget": "—",
                         "notional": "—", "weight": "—"})
            continue
        rows.append({
            "product": format_product_label(p),
            "sigma": _fmt_pct(latest["sigma"]),
            "budget": _fmt_cash(latest["var_alloc"]),
            "notional": _fmt_cash(latest["notional"]),
            "weight": _fmt_pct(latest["weight"]),
        })

    if diag["skipped"]:
        skipped = ", ".join(format_product_label(s) for s in diag["skipped"])
        notice = (notice + " " if notice else "") + f"No returns column for: {skipped}."
    return _STRATEGY_COLUMNS, rows, notice


def _build_portfolio_summary(strategy_names, strategies, returns, var_config):
    """Return (columns, rows, notice) for the Portfolio VaR tab."""
    rows: List[dict] = []
    for s in strategy_names:
        df = strategies.get(s)
        if df is None:
            continue
        products = valid_product_columns(df)
        cfg = (var_config or {}).get(s) or {}
        returns_df = returns.get(s)
        active = bool(cfg.get("active")) and returns_df is not None
        valid = active and validate_var_config(cfg.get("total_var"), cfg.get("allocations", {}), products)["ok"]
        if valid:
            status = "On"
        elif active:
            status = "On (check config)"
        elif returns_df is None:
            status = "No returns CSV"
        else:
            status = "Off"
        rows.append({
            "strategy": s,
            "varon": status,
            "total": _fmt_cash(cfg.get("total_var")) if valid else "—",
            "nprod": str(len(products)),
        })
    notice = (
        "VaR scaling is configured inside each strategy tab. The portfolio sums "
        "each strategy’s effective PnL — VaR-scaled where it’s on, raw otherwise."
    )
    return _PORTFOLIO_COLUMNS, rows, notice


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_var_callbacks(
    app: dash.Dash,
    strategies: Dict[str, pd.DataFrame],
    returns: Dict[str, pd.DataFrame],
) -> None:
    """Register all VaR-scaling callbacks on *app*."""
    strategy_names = list(strategies.keys())
    returns = returns or {}

    # ---- Config modal open/close ----
    @app.callback(
        Output("store-var-modal-open", "data"),
        Input("btn-var-open-from-tab", "n_clicks"),
        Input("close-var-modal", "n_clicks"),
        Input("var-modal-backdrop", "n_clicks"),
        State("store-var-modal-open", "data"),
        prevent_initial_call=True,
    )
    def toggle_var_modal(open_clicks, close_clicks, backdrop_clicks, is_open):
        trig = callback_context.triggered_id
        if trig == "btn-var-open-from-tab":
            return True
        if trig in ("close-var-modal", "var-modal-backdrop"):
            return False
        return is_open or False

    @app.callback(
        Output("var-modal-overlay", "className"),
        Input("store-var-modal-open", "data"),
    )
    def var_modal_class(is_open):
        return "var-modal-overlay open" if is_open else "var-modal-overlay"

    # ---- Expand modal open/close ----
    @app.callback(
        Output("store-var-expand-open", "data"),
        Input("open-var-expand-modal", "n_clicks"),
        Input("close-var-expand-modal", "n_clicks"),
        Input("var-expand-backdrop", "n_clicks"),
        State("store-var-expand-open", "data"),
        prevent_initial_call=True,
    )
    def toggle_var_expand(open_clicks, close_clicks, backdrop_clicks, is_open):
        trig = callback_context.triggered_id
        if trig == "open-var-expand-modal":
            return True
        if trig in ("close-var-expand-modal", "var-expand-backdrop"):
            return False
        return is_open or False

    @app.callback(
        Output("var-expand-overlay", "className"),
        Input("store-var-expand-open", "data"),
    )
    def var_expand_class(is_open):
        return "var-expand-overlay open" if is_open else "var-expand-overlay"

    # ---- Render the whole config form (total input, switch, allocation rows),
    # seeded per strategy. Seeding via children (not via `.value` outputs) keeps
    # the config<->inputs relationship acyclic: there is no `config -> value`
    # edge, only `config -> children`, so the renderer's cycle check passes.
    @app.callback(
        Output("var-total-container", "children"),
        Output("var-switch-container", "children"),
        Output("var-alloc-rows", "children"),
        Input("store-selected-strategy", "data"),
        Input("store-var-modal-open", "data"),
        Input("btn-var-reset", "n_clicks"),
        State("store-var-config", "data"),
    )
    def render_var_form(strategy, _modal_open, _reset, var_config):
        if not strategy or strategy == "Portfolio":
            notice = html.Div(
                "VaR scaling applies to an individual strategy. Select a strategy "
                "tab to configure it; the Portfolio view then sums each strategy’s "
                "effective PnL.",
                className="var-notice",
            )
            return _total_input(None, disabled=True), _active_switch(False, disabled=True), notice

        products = valid_product_columns(strategies.get(strategy))
        if not products:
            notice = html.Div("This strategy has no products to allocate.", className="var-notice")
            return _total_input(None, disabled=True), _active_switch(False, disabled=True), notice

        # Reset blanks the form directly (no config read -> no race with persist).
        cfg = {} if callback_context.triggered_id == "btn-var-reset" else ((var_config or {}).get(strategy) or {})
        allocations = cfg.get("allocations") or {}
        rows = [var_alloc_row(p, allocations.get(p)) for p in products]
        return _total_input(cfg.get("total_var")), _active_switch(bool(cfg.get("active"))), rows

    # ---- Live validation + inline error indication ----
    @app.callback(
        Output("var-alloc-total", "children"),
        Output("var-alloc-total", "className"),
        Output("var-validation-msg", "children"),
        Output("btn-var-apply", "disabled"),
        Output("var-total-error", "children"),
        Output("var-total-input", "className"),
        Output({"type": "var-alloc-error", "product": ALL}, "children"),
        Output({"type": "var-alloc-input", "product": ALL}, "className"),
        Input({"type": "var-alloc-input", "product": ALL}, "value"),
        Input("var-total-input", "value"),
        State({"type": "var-alloc-input", "product": ALL}, "id"),
    )
    def validate_inputs(values, total_var, ids):
        products = [i["product"] for i in ids]
        if not products:
            return "Total: —", "var-alloc-total", "Select a strategy to configure VaR.", True, "", "var-total-input", [], []
        allocations = {i["product"]: v for i, v in zip(ids, values)}
        verdict = validate_var_config(total_var, allocations, products)

        sum_ok = (not verdict["row_errors"]) and abs(verdict["alloc_sum"] - 100.0) <= VAR_ALLOC_TOLERANCE
        total_text = f"Total: {verdict['alloc_sum']:.2f}% / 100%"
        total_cls = "var-alloc-total ok" if sum_ok else "var-alloc-total bad"

        row_err = [verdict["row_errors"].get(i["product"], "") for i in ids]
        input_cls = [
            "var-alloc-input invalid" if i["product"] in verdict["row_errors"] else "var-alloc-input"
            for i in ids
        ]
        total_err = verdict["total_error"] or ""
        total_input_cls = "var-total-input invalid" if verdict["total_error"] else "var-total-input"
        return (
            total_text, total_cls, verdict["message"], (not verdict["ok"]),
            total_err, total_input_cls, row_err, input_cls,
        )

    # ---- Equal-weight / Normalize helpers ----
    @app.callback(
        Output({"type": "var-alloc-input", "product": ALL}, "value"),
        Input("btn-var-equal", "n_clicks"),
        Input("btn-var-normalize", "n_clicks"),
        State({"type": "var-alloc-input", "product": ALL}, "value"),
        State({"type": "var-alloc-input", "product": ALL}, "id"),
        prevent_initial_call=True,
    )
    def fill_allocations(_equal_clicks, _norm_clicks, values, ids):
        products = [i["product"] for i in ids]
        if not products:
            return []
        trig = callback_context.triggered_id
        if trig == "btn-var-equal":
            weights = equal_weight_allocations(products)
        else:
            current = {i["product"]: v for i, v in zip(ids, values)}
            weights = normalize_allocations(current, products)
        return [round(weights.get(p, 0.0), 4) for p in products]

    # ---- Persist config into the selected strategy's slot (per-strategy memory) ----
    @app.callback(
        Output("store-var-config", "data"),
        Input("btn-var-apply", "n_clicks"),
        Input("var-active-switch", "value"),
        Input("btn-var-reset", "n_clicks"),
        State("var-total-input", "value"),
        State({"type": "var-alloc-input", "product": ALL}, "value"),
        State({"type": "var-alloc-input", "product": ALL}, "id"),
        State("store-selected-strategy", "data"),
        State("store-var-config", "data"),
        prevent_initial_call=True,
    )
    def persist_var_config(_apply, switch_value, _reset, total_var, values, ids, strategy, config):
        if not strategy or strategy == "Portfolio":
            return dash.no_update
        config = dict(config or {})
        products = [i["product"] for i in ids]
        allocations = {i["product"]: v for i, v in zip(ids, values)}
        trig = callback_context.triggered_id

        if trig == "btn-var-reset":
            if strategy in config:
                config.pop(strategy, None)
                return config
            return dash.no_update

        if trig == "btn-var-apply":
            verdict = validate_var_config(total_var, allocations, products)
            if not verdict["ok"]:
                return dash.no_update
            config[strategy] = {"total_var": total_var, "allocations": allocations, "active": True}
            return config

        if trig == "var-active-switch":
            existing = dict(config.get(strategy) or {})
            desired = bool(switch_value)
            # Break the seeding feedback loop: only write on a genuine change.
            if existing and existing.get("active", False) == desired:
                return dash.no_update
            if not existing and not desired:
                return dash.no_update
            if not existing:
                existing = {"total_var": total_var, "allocations": allocations}
            existing["active"] = desired
            config[strategy] = existing
            return config

        return dash.no_update

    # ---- VaR tab: summary tables + pane/expand-button visibility ----
    @app.callback(
        Output("var-summary-wrapper", "style"),
        Output("custom-graph-wrapper", "style"),
        Output("open-custom-analytics-modal", "style"),
        Output("open-var-expand-modal", "style"),
        Output("var-summary-table", "data"),
        Output("var-summary-table", "columns"),
        Output("var-summary-table", "style_cell"),
        Output("var-summary-table", "style_header"),
        Output("var-notice", "children"),
        Output("var-expand-table", "data"),
        Output("var-expand-table", "columns"),
        Output("var-expand-table", "style_cell"),
        Output("var-expand-table", "style_header"),
        Output("var-expand-notice", "children"),
        Input("custom-tabs", "active_tab"),
        Input("store-selected-strategy", "data"),
        Input("store-var-config", "data"),
        Input("store-theme", "data"),
    )
    def var_summary_and_visibility(active_tab, strategy, var_config, theme):
        cell = metrics_cell_style(theme)
        header = metrics_header_style(theme)
        cell_lg = metrics_cell_style(theme, padding="12px 14px", min_width="140px")
        header_lg = metrics_header_style(theme, header_bg_dark="rgba(255,255,255,0.05)")

        if active_tab != "tab-var":
            # Show the chart + its expand button; hide the VaR pane + VaR expand.
            return (
                {"display": "none"}, {}, {}, {"display": "none"},
                [], [], cell, header, "",
                [], [], cell_lg, header_lg, "",
            )

        if strategy == "Portfolio":
            columns, rows, notice = _build_portfolio_summary(strategy_names, strategies, returns, var_config)
        else:
            columns, rows, notice = _build_strategy_summary(strategy, strategies, returns, var_config)

        return (
            {}, {"display": "none"}, {"display": "none"}, {},
            rows, columns, cell, header, notice,
            rows, columns, cell_lg, header_lg, notice,
        )
