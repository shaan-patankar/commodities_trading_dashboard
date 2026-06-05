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
# Tints data cells with a translucent brand-green whose strength scales with how
# "good" the value is — strong green = best, fading to the bare panel for the
# worst. The tint is a single rgba(green, alpha) so it reads correctly over both
# the dark and light panel backgrounds without theme branching.

_HEAT_MAX_ALPHA = 0.62

# Worst -> best colour ramp. Spanning hue (deep teal-green through mid green to a
# bright lime-green) as well as lightness gives adjacent values an extra axis of
# contrast, so similar magnitudes no longer collapse into near-identical greens.
_HEAT_RAMP = (
    (28, 102, 84),     # t = 0.0  deep teal-green (worst-but-present)
    (44, 168, 104),    # t = 0.5  mid brand green
    (164, 244, 120),   # t = 1.0  bright lime-green (best)
)


def _ramp_rgb(t: float) -> tuple:
    """Piecewise-linear interpolate the worst->best heat ramp at ``t`` in [0, 1]."""
    t = max(0.0, min(1.0, t))
    if t <= 0.5:
        f, c0, c1 = t / 0.5, _HEAT_RAMP[0], _HEAT_RAMP[1]
    else:
        f, c0, c1 = (t - 0.5) / 0.5, _HEAT_RAMP[1], _HEAT_RAMP[2]
    return tuple(round(c0[i] + (c1[i] - c0[i]) * f) for i in range(3))

# Metrics where a *smaller* number is the better outcome, so the colour scale is
# inverted (small = green). Drawdowns are negative numbers, so a larger (closer
# to zero) value is already the better outcome and needs no inversion.
LOWER_IS_BETTER = frozenset({"Max DD Duration (Days)", "Std Daily PnL"})


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
    """A rounded, softly-graded heat tile with a crisp ring.

    Each value sits in its own rounded tile. Both the *colour* (along the worst->
    best ramp) and the *alpha* track the value's goodness, so adjacent cells get a
    distinct hue + lightness + opacity — making it easy to read which is better at
    a glance instead of squinting at near-identical greens. A small alpha floor
    keeps even the weakest value visibly tinted (so "tinted-low" reads apart from
    "untinted").
    """
    t = intensity
    r, g, b = _ramp_rgb(t)
    a = (0.22 + 0.78 * t) * _HEAT_MAX_ALPHA
    top = min(a * 1.16, 0.74)
    ring = min(a * 1.5, 0.9)
    return {
        "if": {"row_index": row_index, "column_id": column_id},
        "background": f"linear-gradient(180deg, rgba({r},{g},{b},{top:.3f}), rgba({r},{g},{b},{a:.3f}))",
        "borderRadius": "9px",
        "boxShadow": f"inset 0 0 0 1px rgba({r},{g},{b},{ring:.3f})",
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
    *heatmap_on*, value-tint rules are inserted *before* the state reset so a
    clicked cell still clears cleanly.
    """
    label_ids = set(label_ids)
    align = [{"if": {"column_id": c}, "textAlign": "left"} for c in label_ids]
    heat: List[dict] = []
    if heatmap_on and data and columns:
        heat = _heatmap_cells(data, columns, label_ids, orient, lower_better)
    return align + heat + _STATE_RESET
