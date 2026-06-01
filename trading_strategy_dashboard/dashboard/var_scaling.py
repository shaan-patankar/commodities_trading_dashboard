"""VaR-based position sizing (volatility targeting).

Implements per-strategy VaR scaling: each product is sized so that its daily
95% VaR matches its slice of a total VaR budget, using

    notional(t) = (total_var * pct) / (z * sigma(t))
    scaled_pnl(t) = notional(t-1) * return(t)

where ``sigma`` is the rolling standard deviation of the underlying's *daily
returns* (from a per-strategy returns CSV), ``z = 1.645`` (95% one-tailed) and
the window is 20 trading days. The ``shift(1)`` avoids look-ahead. Results are
structurally identical to a PnL frame, so the existing equity/drawdown/metrics
engine consumes them unchanged.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import pandas as pd

from dashboard.config import (
    DEFAULT_INITIAL_CAPITAL,
    VAR_ALLOC_TOLERANCE,
    VAR_WINDOW,
    VAR_Z,
)
from dashboard.utils import valid_product_columns

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Allocation helpers
# ---------------------------------------------------------------------------

def _coerce_float(value: object) -> Optional[float]:
    """Best-effort float coercion; returns ``None`` for blank/invalid input."""
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def equal_weight_allocations(products: List[str]) -> Dict[str, float]:
    """Return ``{product: 100/n}`` percentages; the last product absorbs rounding."""
    n = len(products)
    if n == 0:
        return {}
    base = round(100.0 / n, 6)
    weights = {p: base for p in products}
    weights[products[-1]] = round(100.0 - base * (n - 1), 6)
    return weights


def normalize_allocations(allocations: Dict[str, float], products: List[str]) -> Dict[str, float]:
    """Restrict *allocations* to *products* and rescale them to sum to 100.

    Non-numeric/blank entries are treated as 0. If the restricted total is
    non-positive, falls back to equal weights.
    """
    if not products:
        return {}
    vals = {p: (_coerce_float(allocations.get(p)) or 0.0) for p in products}
    total = sum(vals.values())
    if total <= 0:
        return equal_weight_allocations(products)
    return {p: v / total * 100.0 for p, v in vals.items()}


def validate_var_config(
    total_var: object,
    allocations: Dict[str, float],
    products: List[str],
) -> dict:
    """Validate VaR inputs and return a structured result for the UI.

    Returns a dict with:
        ``ok``          -- True only when everything is valid and sums to 100%.
        ``total_error`` -- message for the Total VaR field (or None).
        ``alloc_sum``   -- sum of the valid per-product allocations.
        ``row_errors``  -- ``{product: message}`` for invalid rows.
        ``message``     -- a single consolidated status message.
    """
    allocations = allocations or {}

    total_error: Optional[str] = None
    tv = _coerce_float(total_var)
    if tv is None:
        total_error = "Enter a number"
    elif tv <= 0:
        total_error = "Must be greater than 0"

    row_errors: Dict[str, str] = {}
    alloc_sum = 0.0
    for p in products:
        raw = allocations.get(p)
        if raw is None or raw == "":
            row_errors[p] = "Required"
            continue
        v = _coerce_float(raw)
        if v is None:
            row_errors[p] = "Not a number"
        elif v < 0:
            row_errors[p] = "Must be ≥ 0"
        elif v > 100:
            row_errors[p] = "Must be ≤ 100"
        else:
            alloc_sum += v

    sum_ok = (not row_errors) and abs(alloc_sum - 100.0) <= VAR_ALLOC_TOLERANCE
    ok = (total_error is None) and (not row_errors) and sum_ok and len(products) > 0

    if not products:
        message = "No products to allocate."
    elif total_error is not None:
        message = f"Total VaR: {total_error.lower()}."
    elif row_errors:
        message = "Fix the highlighted allocation(s)."
    elif not sum_ok:
        message = f"Allocations sum to {alloc_sum:.1f}% — must total 100%."
    else:
        message = "Ready to apply."

    return {
        "ok": ok,
        "total_error": total_error,
        "alloc_sum": alloc_sum,
        "row_errors": row_errors,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Scaling engine
# ---------------------------------------------------------------------------

def compute_var_scaled_frame(
    strategy_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    products: List[str],
    total_var: float,
    allocations: Dict[str, float],
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    z: float = VAR_Z,
    window: int = VAR_WINDOW,
) -> Tuple[pd.DataFrame, dict]:
    """Build a vol-targeted scaled-PnL frame for a single strategy.

    Args:
        strategy_df:     Strategy PnL frame (used only for its date anchor + product set).
        returns_df:      Daily-returns frame with matching product columns.
        products:        Products to size (restricted to real strategy columns).
        total_var:       Total VaR budget (currency).
        allocations:     ``{product: percent}`` budget split (rescaled to 100).
        initial_capital: Unused here but kept for signature symmetry.
        z:               VaR z-factor (default 1.645).
        window:          Rolling vol window (default 20).

    Returns:
        ``(scaled_df, diagnostics)``. ``scaled_df`` has ``"date"`` + one scaled-PnL
        column per usable product (structurally a PnL frame). ``diagnostics`` =
        ``{"skipped": [...], "latest": {product: {sigma, var_alloc, notional, weight}}}``.
    """
    diagnostics: dict = {"skipped": [], "latest": {}}

    if (
        strategy_df is None
        or returns_df is None
        or "date" not in getattr(strategy_df, "columns", [])
        or "date" not in getattr(returns_df, "columns", [])
    ):
        return pd.DataFrame(columns=["date"]), diagnostics

    products = valid_product_columns(strategy_df, products)
    tv = _coerce_float(total_var) or 0.0

    # Inner-join on date so sigma warms up on the full shared history.
    aligned = (
        pd.merge(strategy_df[["date"]], returns_df, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if aligned.empty:
        return pd.DataFrame(columns=["date"]), diagnostics

    norm = normalize_allocations(allocations, products)
    scaled_cols: Dict[str, object] = {}

    for p in products:
        if p not in returns_df.columns:
            diagnostics["skipped"].append(p)
            continue
        r = pd.to_numeric(aligned[p], errors="coerce")
        pct = norm.get(p, 0.0) / 100.0
        var_alloc = tv * pct

        sigma = r.rolling(window, min_periods=window).std(ddof=1)
        denom = z * sigma
        notional = var_alloc / denom.where(denom > _EPS)  # NaN where sigma invalid/too small
        prev_notional = notional.shift(1)
        scaled = (prev_notional * r).where(prev_notional.notna(), 0.0).fillna(0.0)
        scaled_cols[p] = scaled.to_numpy()

        sigma_valid = sigma.dropna()
        notional_valid = notional.dropna()
        diagnostics["latest"][p] = {
            "sigma": float(sigma_valid.iloc[-1]) if len(sigma_valid) else float("nan"),
            "var_alloc": float(var_alloc),
            "notional": float(notional_valid.iloc[-1]) if len(notional_valid) else float("nan"),
            "weight": float(pct),
        }

    if not scaled_cols:
        return pd.DataFrame(columns=["date"]), diagnostics

    scaled_df = pd.DataFrame({"date": aligned["date"].to_numpy(), **scaled_cols})
    return scaled_df, diagnostics


def var_scaled_aggregate_series(
    strategy_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    products: List[str],
    total_var: float,
    allocations: Dict[str, float],
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> pd.DataFrame:
    """Return ``["date", "pnl"]`` -- the summed scaled PnL across products.

    Used to stack a strategy's VaR-scaled contribution into the portfolio frame.
    Empty frame if scaling is unavailable.
    """
    scaled_df, _ = compute_var_scaled_frame(
        strategy_df, returns_df, products, total_var, allocations, initial_capital
    )
    if scaled_df.empty:
        return pd.DataFrame(columns=["date", "pnl"])
    value_cols = [c for c in scaled_df.columns if c != "date"]
    agg = scaled_df[value_cols].sum(axis=1)
    return pd.DataFrame({"date": scaled_df["date"].to_numpy(), "pnl": agg.to_numpy()})


def portfolio_effective_dataframe(
    strategies: Dict[str, pd.DataFrame],
    returns: Dict[str, pd.DataFrame],
    selected_strategies: List[str],
    var_configs: Optional[Dict[str, dict]],
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Build the portfolio frame using each strategy's *effective* aggregate PnL.

    A strategy contributes its VaR-scaled aggregate when it has an active, valid
    config and a returns file; otherwise its raw summed PnL (matching the plain
    ``portfolio_dataframe`` behaviour). Strategies are merged on date (outer join,
    missing filled with 0).

    Returns:
        ``(merged_df, breakdown)`` where ``breakdown[strategy] = {var_on, total_var,
        n_products}`` drives the Portfolio VaR summary table.
    """
    var_configs = var_configs or {}
    frames: List[pd.DataFrame] = []
    breakdown: Dict[str, dict] = {}

    for strategy in selected_strategies:
        df = strategies.get(strategy)
        if df is None:
            continue
        products = valid_product_columns(df)
        cfg = var_configs.get(strategy) or {}
        returns_df = returns.get(strategy)
        used_scaled = False

        if bool(cfg.get("active")) and returns_df is not None:
            verdict = validate_var_config(cfg.get("total_var"), cfg.get("allocations", {}), products)
            if verdict["ok"]:
                agg = var_scaled_aggregate_series(
                    df, returns_df, products,
                    cfg.get("total_var"), cfg.get("allocations", {}), initial_capital,
                )
                if not agg.empty:
                    frames.append(agg.rename(columns={"pnl": strategy}))
                    used_scaled = True

        if not used_scaled:
            frame = df[["date"]].copy()
            frame[strategy] = df[products].sum(axis=1)
            frames.append(frame)

        breakdown[strategy] = {
            "var_on": used_scaled,
            "total_var": (_coerce_float(cfg.get("total_var")) if used_scaled else None),
            "n_products": len(products),
        }

    if not frames:
        return pd.DataFrame(columns=["date"]), breakdown

    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge(merged, frame, on="date", how="outer")
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.sort_values("date").reset_index(drop=True).fillna(0.0)
    return merged, breakdown
