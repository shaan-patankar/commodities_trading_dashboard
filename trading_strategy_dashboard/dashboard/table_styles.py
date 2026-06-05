"""Theme-aware ``dash_table`` style builders.

Single source of truth for the metrics / CSV DataTable cell and header styles so
the initial (dark) layout render and the theme-switch callback stay in sync.
Dark ("deep") output is byte-identical to the original inline style dicts; the
"light" branch swaps only the text/header colours so light mode is legible.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

_FONT = "'Inter', 'Segoe UI', system-ui"

_CELL_COLOR = {"deep": "#e6e6e6", "light": "#1a2233"}
_HEADER_COLOR = {"deep": "#ffffff", "light": "#1a2233"}
_HEADER_BG_LIGHT = "rgba(0,0,0,0.04)"


def metrics_cell_style(
    theme: str | None = "deep",
    *,
    padding: str = "10px 12px",
    min_width: str = "120px",
) -> dict:
    """Build the ``style_cell`` dict for a metrics/CSV DataTable.

    Args:
        theme:     ``"deep"`` (dark, default) or ``"light"``.
        padding:   Cell padding (per-table).
        min_width: Minimum column width (per-table).
    """
    key = theme if theme in _CELL_COLOR else "deep"
    return {
        "backgroundColor": "rgba(0,0,0,0)",
        "color": _CELL_COLOR[key],
        "border": "0px",
        "fontFamily": _FONT,
        "fontSize": "13px",
        "padding": padding,
        "textAlign": "right",
        "whiteSpace": "normal",
        "height": "auto",
        "minWidth": min_width,
        "width": "auto",
        "maxWidth": "none",
    }


def metrics_header_style(
    theme: str | None = "deep",
    *,
    header_bg_dark: str = "rgba(255,255,255,0.03)",
) -> dict:
    """Build the ``style_header`` dict for a metrics/CSV DataTable.

    Args:
        theme:          ``"deep"`` (dark, default) or ``"light"``.
        header_bg_dark: Dark-mode header background (per-table).
    """
    light = theme == "light"
    return {
        "backgroundColor": _HEADER_BG_LIGHT if light else header_bg_dark,
        "color": _HEADER_COLOR["light"] if light else _HEADER_COLOR["deep"],
        "border": "0px",
        "fontWeight": "700",
        "fontFamily": _FONT,
        "fontSize": "13px",
        "textTransform": "none",
        "letterSpacing": "0.2px",
        "textAlign": "right",
        "whiteSpace": "normal",
        "lineHeight": "1.2",
    }


# ---------------------------------------------------------------------------
# Value heat-map overlay (toggleable in Settings)
# ---------------------------------------------------------------------------
# A spreadsheet-style "green -> white" value scale (à la Excel). Each tinted cell
# is an OPAQUE tile interpolated from a soft near-white (worst) to a rich green
# (best), with dark text on top. Opaque light tiles are what make the steps easy
# to tell apart: the scale spans a wide white->green LIGHTNESS range, instead of
# a translucent green over the dark panel that collapses every value into a
# narrow band of near-identical dark greens. The scale is intentionally
# theme-independent so the metric cells read like a familiar Excel heat-map on
# both the dark and light dashboards; only the surrounding chrome (header, label
# column) follows the theme.

_HEAT_LOW = (233, 246, 239)    # worst value -> soft near-white green
_HEAT_HIGH = (36, 156, 88)     # best value  -> rich, professional green
_HEAT_TEXT = "#0f1c15"         # dark text, readable across the whole scale
_HEAT_RING = (16, 74, 48)      # faint tile outline so cells separate on both themes

# Metrics where a *smaller* number is the better outcome, so the colour scale is
# inverted (small = green). Drawdowns are negative numbers, so a larger (closer
# to zero) value is already the better outcome and needs no inversion.
LOWER_IS_BETTER = frozenset({"Max DD Duration (Days)", "Std Daily PnL"})


def _mix(c0: Sequence[float], c1: Sequence[float], t: float) -> tuple:
    """Linear-interpolate two RGB triples at ``t`` in [0, 1]."""
    return tuple(round(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))


def _parse_number(value) -> float | None:
    """Best-effort parse of a formatted cell into a float, else ``None``.

    Strips thousands separators, ``%`` and currency glyphs; blank / ``"—"`` /
    non-numeric padding cells return ``None`` (left untinted).
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return None if math.isnan(v) else v
    text = str(value).strip()
    if not text or text in {"—", "-", "N/A", "nan", "NaN"}:
        return None
    cleaned = text.replace(",", "").replace("%", "").replace("$", "").replace("£", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _intensity(v: float, lo: float, hi: float, invert: bool) -> float:
    """Map *v* to a 0..1 goodness intensity within ``[lo, hi]``.

    Degenerate spans (a single value / all-equal row) fall back to the sign of
    the value so single-column tables still tint positives green.
    """
    if hi == lo:
        t = 0.55 if v > 0 else 0.0
    else:
        t = (v - lo) / (hi - lo)
        if invert:
            t = 1.0 - t
    return max(0.0, min(1.0, t))


def _cell_tint(row_index: int, column_id: str, intensity: float) -> dict:
    """An opaque green→white heat tile (Excel-style) with dark, readable text.

    The background interpolates soft near-white (worst) → rich green (best); a
    fixed dark text colour stays legible across the whole range, and a faint ring
    seats each value in its own tile. Opaque light tiles give the wide, easy-to-
    read lightness gradient that a translucent-over-dark tint cannot.
    """
    # Mild gamma so the mid-range greens separate a touch more than pure linear.
    t = intensity ** 0.85
    r, g, b = _mix(_HEAT_LOW, _HEAT_HIGH, t)
    rr, rg, rb = _HEAT_RING
    return {
        "if": {"row_index": row_index, "column_id": column_id},
        "backgroundColor": f"rgb({r},{g},{b})",
        "color": _HEAT_TEXT,
        "borderRadius": "8px",
        "boxShadow": f"inset 0 0 0 1px rgba({rr},{rg},{rb},0.28)",
    }


def _heatmap_cells(
    data: Sequence[dict],
    columns: Sequence[dict],
    label_ids: set,
    orient: str,
    lower_better: Iterable[str],
) -> List[dict]:
    """Build per-cell tint rules, normalised per-row or per-column."""
    lower_better = set(lower_better)
    value_cols = [c["id"] for c in columns if c["id"] not in label_ids]
    cells: List[dict] = []

    if orient == "row":
        for r, row in enumerate(data):
            nums = [(c, _parse_number(row.get(c))) for c in value_cols]
            nums = [(c, v) for c, v in nums if v is not None]
            if not nums:
                continue
            metric = next((row.get(l) for l in label_ids if l in row), None)
            invert = metric in lower_better
            vals = [v for _, v in nums]
            lo, hi = min(vals), max(vals)
            for c, v in nums:
                cells.append(_cell_tint(r, c, _intensity(v, lo, hi, invert)))
    else:  # per-column
        for c in value_cols:
            nums = [(r, _parse_number(row.get(c))) for r, row in enumerate(data)]
            nums = [(r, v) for r, v in nums if v is not None]
            if not nums:
                continue
            vals = [v for _, v in nums]
            lo, hi = min(vals), max(vals)
            for r, v in nums:
                cells.append(_cell_tint(r, c, _intensity(v, lo, hi, False)))
    return cells


_STATE_RESET = [
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
]


def table_data_conditional(
    data: Sequence[dict] | None = None,
    columns: Sequence[dict] | None = None,
    *,
    label_ids: Iterable[str] = (),
    orient: str = "row",
    lower_better: Iterable[str] = (),
    heatmap_on: bool = False,
) -> List[dict]:
    """Compose a DataTable ``style_data_conditional`` list.

    Always includes the left-aligned label column(s) and the transparent
    active/selected reset (so the selection box never flashes). When
    *heatmap_on*, the opaque value-tile rules are appended *after* the state
    reset so a clicked/selected tinted cell keeps its tile and dark text (rather
    than clearing to a transparent — and, with dark text, unreadable — cell).
    """
    label_ids = set(label_ids)
    align = [{"if": {"column_id": c}, "textAlign": "left"} for c in label_ids]
    heat: List[dict] = []
    if heatmap_on and data and columns:
        heat = _heatmap_cells(data, columns, label_ids, orient, lower_better)
    return align + _STATE_RESET + heat
