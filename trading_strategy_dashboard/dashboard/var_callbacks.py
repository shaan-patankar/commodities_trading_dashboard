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
import pandas as pd

from dashboard.config import DEFAULT_TOTAL_VAR, RETURNS_FILES, VAR_ALLOC_TOLERANCE
from dashboard.layout import var_alloc_row, var_volume_row
from dashboard.table_styles import metrics_cell_style, metrics_header_style, table_data_conditional
from dashboard.utils import format_product_label, valid_product_columns
from dashboard.var_scaling import (
    compute_var_scaled_frame,
    config_is_valid,
    config_mode,
    equal_weight_allocations,
    validate_fixed_config,
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
    {"name": "VaR Allocation", "id": "budget"},
    {"name": "Volume", "id": "notional"},
    {"name": "Weight", "id": "weight"},
]

_FIXED_COLUMNS = [
    {"name": "Product", "id": "product"},
    {"name": "σ (20d)", "id": "sigma"},
    {"name": "Volume", "id": "volume"},
]

_PORTFOLIO_COLUMNS = [
    {"name": "Strategy", "id": "strategy"},
    {"name": "Status", "id": "varon"},
    {"name": "Mode", "id": "mode"},
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


# The VaR summary tables mirror the Key Metrics table: real product rows sit at
# the top and the remainder is padded with blank "—" rows so the table fills its
# box. The pad COUNT is computed clientside from the measured container height
# (so only as many blanks as the viewport needs), hence the builders below return
# the real rows only.


def _build_strategy_summary(strategy, strategies, returns, var_config):
    """Return (columns, rows, notice) for an individual strategy's VaR tab.

    Dispatches on the strategy's persisted mode: vol-targeting shows σ/budget/
    notional/weight; fixed-notional shows the per-product volume and the latest
    raw/effective PnL.
    """
    df = strategies.get(strategy)
    products = valid_product_columns(df) if df is not None else []
    cfg = (var_config or {}).get(strategy) or {}
    returns_df = returns.get(strategy)
    active = bool(cfg.get("active"))

    if df is None:
        return _STRATEGY_COLUMNS, [], ""

    if config_mode(cfg) == "fixed":
        return _build_fixed_summary(strategy, df, products, cfg, active, returns_df)

    # ---- Volatility (VaR budget) mode ----
    if returns_df is None:
        return _STRATEGY_COLUMNS, [], ""

    # σ is independent of the VaR config, so it is always shown. The budget /
    # notional / weight columns only carry values while scaling is actually on;
    # when off they blank out to "—" (the diagnostics are still computed only to
    # obtain σ).
    total_var = cfg.get("total_var") if cfg.get("total_var") is not None else DEFAULT_TOTAL_VAR
    allocations = cfg.get("allocations") or {}
    _, diag = compute_var_scaled_frame(df, returns_df, products, total_var, allocations)

    rows: List[dict] = []
    for p in products:
        latest = diag["latest"].get(p)
        sigma = _fmt_pct(latest["sigma"]) if latest else "—"
        if active and latest is not None:
            rows.append({
                "product": format_product_label(p),
                "sigma": sigma,
                "budget": _fmt_cash(latest["var_alloc"]),
                "notional": _fmt_cash(latest["notional"]),
                "weight": _fmt_pct(latest["weight"]),
            })
        else:
            rows.append({
                "product": format_product_label(p),
                "sigma": sigma, "budget": "—", "notional": "—", "weight": "—",
            })
    return _STRATEGY_COLUMNS, rows, ""


def _build_fixed_summary(strategy, df, products, cfg, active, returns_df):
    """Return (columns, rows, notice) for a strategy in fixed-volume mode.

    Shows the product's σ (20d) — the same returns-based volatility used in the
    VaR-scaled view, config-independent so always shown when a returns CSV is
    present — alongside the configured Volume (only while scaling is on).
    """
    volumes = cfg.get("volumes") or {}

    # σ is independent of the volumes, so compute it once off the returns CSV
    # (allocations are irrelevant to σ). Absent returns -> "—".
    sigma_latest: dict = {}
    if returns_df is not None:
        _, sdiag = compute_var_scaled_frame(df, returns_df, products, DEFAULT_TOTAL_VAR, {})
        sigma_latest = sdiag.get("latest", {})

    rows: List[dict] = []
    for p in products:
        slat = sigma_latest.get(p)
        sigma = _fmt_pct(slat["sigma"]) if slat else "—"
        vol = volumes.get(p)
        has_vol = vol is not None and vol != ""
        rows.append({
            "product": format_product_label(p),
            "sigma": sigma,
            "volume": _fmt_cash(vol) if (active and has_vol) else "—",
        })
    return _FIXED_COLUMNS, rows, ""


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
        active = bool(cfg.get("active"))
        mode = config_mode(cfg)
        valid = active and config_is_valid(cfg, returns_df, products)

        if not active:
            status = "Off"
        elif valid:
            status = "On"
        elif mode == "vol" and returns_df is None:
            status = "No returns CSV"
        else:
            status = "On (check config)"

        mode_label = ("Fixed Volume" if mode == "fixed" else "VaR Scaled") if valid else "—"
        rows.append({
            "strategy": s,
            "varon": status,
            "mode": mode_label,
            "nprod": str(len(products)),
        })
    return _PORTFOLIO_COLUMNS, rows, ""


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

    # ---- Scaling-mode selection (Volatility VaR budget vs Fixed notional).
    # A single callback both seeds the mode from the strategy's saved config (on
    # strategy change / modal open) and flips it on a toggle click. store-var-mode
    # is never an input to the form renderer, so this stays acyclic.
    @app.callback(
        Output("store-var-mode", "data"),
        Input("store-selected-strategy", "data"),
        Input("store-var-modal-open", "data"),
        Input("var-mode-vol-btn", "n_clicks"),
        Input("var-mode-fixed-btn", "n_clicks"),
        State("store-var-config", "data"),
    )
    def set_var_mode(strategy, _modal_open, _vol_clicks, _fixed_clicks, var_config):
        trig = callback_context.triggered_id
        if trig == "var-mode-vol-btn":
            return "vol"
        if trig == "var-mode-fixed-btn":
            return "fixed"
        cfg = (var_config or {}).get(strategy) or {}
        return config_mode(cfg)

    @app.callback(
        Output("var-vol-section", "style"),
        Output("var-fixed-section", "style"),
        Output("var-mode-vol-btn", "className"),
        Output("var-mode-fixed-btn", "className"),
        Output("btn-var-equal", "style"),
        Input("store-var-mode", "data"),
    )
    def toggle_var_mode_sections(mode):
        is_fixed = (mode == "fixed")
        vol_style = {"display": "none"} if is_fixed else {}
        fixed_style = {} if is_fixed else {"display": "none"}
        vol_cls = "var-mode-btn" if is_fixed else "var-mode-btn active"
        fixed_cls = "var-mode-btn active" if is_fixed else "var-mode-btn"
        # Equal Weight only applies to the allocation (vol) mode.
        equal_style = {"display": "none"} if is_fixed else {}
        return vol_style, fixed_style, vol_cls, fixed_cls, equal_style

    # ---- Render the whole config form (total input, switch, allocation rows,
    # volume rows), seeded per strategy. Seeding via children (not via `.value`
    # outputs) keeps the config<->inputs relationship acyclic: there is no
    # `config -> value` edge, only `config -> children`, so the renderer's cycle
    # check passes. Both mode row sets are always rendered; the inactive one is
    # simply hidden by toggle_var_mode_sections.
    @app.callback(
        Output("var-total-container", "children"),
        Output("var-alloc-rows", "children"),
        Output("var-volume-rows", "children"),
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
            return _total_input(None, disabled=True), notice, []

        products = valid_product_columns(strategies.get(strategy))
        if not products:
            notice = html.Div("This strategy has no products to allocate.", className="var-notice")
            return _total_input(None, disabled=True), notice, []

        # Reset blanks the form directly (no config read -> no race with persist).
        cfg = {} if callback_context.triggered_id == "btn-var-reset" else ((var_config or {}).get(strategy) or {})
        allocations = cfg.get("allocations") or {}
        volumes = cfg.get("volumes") or {}
        alloc_rows = [var_alloc_row(p, allocations.get(p)) for p in products]
        volume_rows = [var_volume_row(p, volumes.get(p)) for p in products]
        return _total_input(cfg.get("total_var")), alloc_rows, volume_rows

    # ---- Live validation + the on/off Apply button (mode-aware, calm) ----
    # The Apply button doubles as the on/off control: it reads "VaR Off" (and is
    # always enabled) when scaling is already on, and "VaR On" (enabled only when
    # the config is valid) when scaling is off. "Calm" validation: an empty field
    # is *incomplete*, not invalid — no red on the % boxes; only the Total VaR
    # field flags a hard error. The running "X% / 100%" total (green at 100%)
    # signals balance.
    @app.callback(
        Output("var-alloc-total", "children"),
        Output("var-alloc-total", "className"),
        Output("btn-var-apply", "disabled"),
        Output("btn-var-apply", "children"),
        Output("btn-var-apply", "className"),
        Output("var-total-error", "children"),
        Output("var-total-input", "className"),
        Output({"type": "var-alloc-error", "product": ALL}, "children"),
        Output({"type": "var-alloc-input", "product": ALL}, "className"),
        Output({"type": "var-volume-error", "product": ALL}, "children"),
        Output({"type": "var-volume-input", "product": ALL}, "className"),
        Input({"type": "var-alloc-input", "product": ALL}, "value"),
        Input("var-total-input", "value"),
        Input({"type": "var-volume-input", "product": ALL}, "value"),
        Input("store-var-mode", "data"),
        Input("store-var-config", "data"),
        Input("store-selected-strategy", "data"),
        State({"type": "var-alloc-input", "product": ALL}, "id"),
        State({"type": "var-volume-input", "product": ALL}, "id"),
    )
    def validate_inputs(alloc_values, total_var, volume_values, mode, var_config, strategy, alloc_ids, volume_ids):
        mode = mode or "vol"
        alloc_products = [i["product"] for i in alloc_ids]
        volume_products = [i["product"] for i in volume_ids]

        blank_alloc_err = ["" for _ in alloc_ids]
        ok_alloc_cls = ["var-alloc-input" for _ in alloc_ids]
        blank_vol_err = ["" for _ in volume_ids]
        ok_vol_cls = ["var-alloc-input" for _ in volume_ids]

        active = bool(
            (var_config or {}).get(strategy, {}).get("active")
        ) if (strategy and strategy != "Portfolio") else False
        # The button reflects the *current* state (so reopening a configured
        # strategy correctly shows "VaR On"), and a single click toggles it.
        # When on it carries the accent-filled "active" styling.
        label = "VaR On" if active else "VaR Off"
        apply_cls = "var-action-btn var-apply-btn" + (" active" if active else "")

        def _apply_state(valid: bool):
            # Always allow turning off; only allow turning on when valid.
            return (False if active else (not valid)), label, apply_cls

        if not alloc_products and not volume_products:
            disabled, lbl, cls = _apply_state(False)
            return (
                "0% / 100%", "var-alloc-total", disabled, lbl, cls,
                "", "var-total-input", blank_alloc_err, ok_alloc_cls,
                blank_vol_err, ok_vol_cls,
            )

        # A "real" error is anything other than the soft "Required" (empty) hint.
        def _real(err: str) -> bool:
            return bool(err) and err != "Required"

        if mode == "fixed":
            volumes = {i["product"]: v for i, v in zip(volume_ids, volume_values)}
            verdict = validate_fixed_config(volumes, volume_products)
            vol_err, vol_cls = [], []
            for i in volume_ids:
                err = verdict["row_errors"].get(i["product"], "")
                bad = _real(err)
                vol_err.append(err if bad else "")
                vol_cls.append("var-alloc-input invalid" if bad else "var-alloc-input")
            disabled, lbl, cls = _apply_state(verdict["ok"])
            return (
                "0% / 100%", "var-alloc-total", disabled, lbl, cls,
                "", "var-total-input", blank_alloc_err, ok_alloc_cls,
                vol_err, vol_cls,
            )

        # ---- Volatility (VaR budget) mode ----
        allocations = {i["product"]: v for i, v in zip(alloc_ids, alloc_values)}
        verdict = validate_var_config(total_var, allocations, alloc_products)
        sum_ok = (not verdict["row_errors"]) and abs(verdict["alloc_sum"] - 100.0) <= VAR_ALLOC_TOLERANCE

        total_text = f"{verdict['alloc_sum']:.0f}% / 100%"
        total_cls = "var-alloc-total ok" if sum_ok else "var-alloc-total"

        # Allocation % boxes never turn red; the running total signals balance.
        total_empty = total_var is None or total_var == ""
        total_bad = bool(verdict["total_error"]) and not total_empty
        total_err = verdict["total_error"] if total_bad else ""
        total_input_cls = "var-total-input invalid" if total_bad else "var-total-input"

        disabled, lbl, cls = _apply_state(verdict["ok"])
        return (
            total_text, total_cls, disabled, lbl, cls,
            total_err, total_input_cls, blank_alloc_err, ok_alloc_cls,
            blank_vol_err, ok_vol_cls,
        )

    # ---- Equal-weight helper ----
    @app.callback(
        Output({"type": "var-alloc-input", "product": ALL}, "value"),
        Input("btn-var-equal", "n_clicks"),
        State({"type": "var-alloc-input", "product": ALL}, "id"),
        prevent_initial_call=True,
    )
    def fill_allocations(_equal_clicks, ids):
        products = [i["product"] for i in ids]
        if not products:
            return []
        weights = equal_weight_allocations(products)
        return [round(weights.get(p, 0.0), 4) for p in products]

    # ---- Persist config into the selected strategy's slot (per-strategy memory).
    # Applying (or toggling the switch on/off) also closes the popup, since the
    # change has been applied — there is nothing left to confirm. Reset leaves the
    # popup open so the user can re-enter values.
    @app.callback(
        Output("store-var-config", "data"),
        Output("store-var-modal-open", "data", allow_duplicate=True),
        Input("btn-var-apply", "n_clicks"),
        Input("btn-var-reset", "n_clicks"),
        State("store-var-mode", "data"),
        State("var-total-input", "value"),
        State({"type": "var-alloc-input", "product": ALL}, "value"),
        State({"type": "var-alloc-input", "product": ALL}, "id"),
        State({"type": "var-volume-input", "product": ALL}, "value"),
        State({"type": "var-volume-input", "product": ALL}, "id"),
        State("store-selected-strategy", "data"),
        State("store-var-config", "data"),
        prevent_initial_call=True,
    )
    def persist_var_config(
        _apply, _reset, mode, total_var,
        alloc_values, alloc_ids, volume_values, volume_ids, strategy, config,
    ):
        if not strategy or strategy == "Portfolio":
            return dash.no_update, dash.no_update
        config = dict(config or {})
        mode = mode or "vol"
        alloc_products = [i["product"] for i in alloc_ids]
        volume_products = [i["product"] for i in volume_ids]
        allocations = {i["product"]: v for i, v in zip(alloc_ids, alloc_values)}
        volumes = {i["product"]: v for i, v in zip(volume_ids, volume_values)}
        trig = callback_context.triggered_id

        if mode == "fixed":
            mode_fields = {"mode": "fixed", "volumes": volumes}
            mode_ok = validate_fixed_config(volumes, volume_products)["ok"]
        else:
            mode_fields = {"mode": "vol", "total_var": total_var, "allocations": allocations}
            mode_ok = validate_var_config(total_var, allocations, alloc_products)["ok"]

        if trig == "btn-var-reset":
            if strategy in config:
                config.pop(strategy, None)
                return config, dash.no_update  # keep the popup open after Reset
            return dash.no_update, dash.no_update

        # The Apply button is the single on/off control. If scaling is already on,
        # clicking it turns it off; if it is off, clicking applies the current
        # (valid) inputs and turns it on. Either way the popup closes.
        if trig == "btn-var-apply":
            existing = dict(config.get(strategy) or {})
            if bool(existing.get("active")):
                existing["active"] = False
                config[strategy] = existing
                return config, False  # turned off -> close
            if not mode_ok:
                return dash.no_update, dash.no_update
            existing.update(mode_fields)  # preserve the other mode's saved inputs
            existing["active"] = True
            config[strategy] = existing
            return config, False  # applied + turned on -> close

        return dash.no_update, dash.no_update

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
        Output("var-expand-table", "data"),
        Output("var-expand-table", "columns"),
        Output("var-expand-table", "style_cell"),
        Output("var-expand-table", "style_header"),
        Output("var-summary-table", "style_data_conditional"),
        Output("var-expand-table", "style_data_conditional"),
        Output("store-var-rows", "data"),
        Output("btn-var-open-from-tab", "style"),
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
        label_ids = {"product", "strategy"}

        # The value heat-map is deliberately NOT applied to the VaR tables — it is
        # reserved for the Key Metrics table.
        base = table_data_conditional(label_ids=label_ids)

        if active_tab != "tab-var":
            # Show the chart + its expand button; hide the VaR pane + VaR expand.
            return (
                {"display": "none"}, {}, {}, {"display": "none"},
                [], [], cell, header,
                [], [], cell_lg, header_lg,
                base, base, {}, {},
            )

        if strategy == "Portfolio":
            columns, rows, _ = _build_portfolio_summary(strategy_names, strategies, returns, var_config)
            # VaR is configured per strategy, not for the Portfolio aggregate.
            configure_style = {"display": "none"}
        else:
            columns, rows, _ = _build_strategy_summary(strategy, strategies, returns, var_config)
            configure_style = {}

        # Emit the REAL rows for the clientside dynamic-pad callback. The tables
        # also get the real rows now so they render (and can be measured); the
        # clientside callback then appends exactly enough blank rows to fill.
        return (
            {}, {"display": "none"}, {"display": "none"}, {},
            rows, columns, cell, header,
            rows, columns, cell_lg, header_lg,
            base, base,
            {"rows": rows, "columns": columns}, configure_style,
        )

    # ---- Dynamic blank-row padding (clientside) ----
    # Measure each table's container and append only as many blank "—" rows as
    # are needed to fill the visible height (so the count adapts to the monitor
    # size). Re-runs on data change, window resize, and expand-modal open.
    app.clientside_callback(
        """
        function(varRows, winTick, expandOpen) {
            const noup = window.dash_clientside.no_update;
            const rows = (varRows && varRows.rows) || null;
            const cols = (varRows && varRows.columns) || null;
            function pad(wrapperSel, tableId) {
                if (!rows || !cols) return;
                const wrap = document.querySelector(wrapperSel);
                let target = rows.length;
                if (wrap && wrap.clientHeight > 8) {
                    const headerEl = wrap.querySelector('.dt-table-container__row-0') ||
                                     wrap.querySelector('thead');
                    const bodyEl = wrap.querySelector('.dt-table-container__row-1') || wrap;
                    const bodyRow = bodyEl.querySelector('tbody tr') || wrap.querySelector('tbody tr');
                    const headerH = headerEl ? headerEl.getBoundingClientRect().height : 42;
                    const rowH = (bodyRow ? bodyRow.getBoundingClientRect().height : 0) || 56;
                    const avail = wrap.clientHeight - headerH;
                    // Fill the whole visible body. floor() already guarantees the
                    // rows can't overflow into a scrollbar; a tiny 0.5px nudge
                    // absorbs sub-pixel rounding so we don't fall one row short.
                    const fit = Math.floor((avail + 0.5) / rowH);
                    target = Math.max(rows.length, fit);
                }
                const blank = {};
                cols.forEach(function(c){ blank[c.id] = '—'; });
                const out = rows.slice();
                while (out.length < target) out.push(Object.assign({}, blank));
                window.dash_clientside.set_props(tableId, {data: out});
            }
            function runAll() {
                pad('#var-summary-wrapper .var-summary-table-wrapper', 'var-summary-table');
                if (expandOpen) {
                    pad('.var-expand-modal .var-summary-table-wrapper', 'var-expand-table');
                }
            }
            // Run now, then again after the browser has settled the layout. This
            // matters on the Portfolio tab: switching to it hides the "Configure
            // VaR…" button, which grows the table wrapper by ~a row AFTER this
            // callback first fires — so an immediate-only measure would leave a
            // gap. The rAF + timeout re-measures against the final height.
            runAll();
            requestAnimationFrame(function(){ requestAnimationFrame(runAll); });
            setTimeout(runAll, 120);
            return noup;
        }
        """,
        Output("store-var-pad", "data"),
        Input("store-var-rows", "data"),
        Input("store-win-tick", "data"),
        Input("store-var-expand-open", "data"),
    )
