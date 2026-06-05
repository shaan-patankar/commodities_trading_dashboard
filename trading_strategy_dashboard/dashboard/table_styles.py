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
# Tiles are very slightly translucent so they settle into the dashboard rather
# than sitting on top as flat blocks — it softens the bright "white" end against
# the dark panel and lets the panel's tone read faintly through, while staying
# opaque enough that the hues still pop and the dark text stays readable (the
# greenest cell still clears WCAG AA over the dark panel at this alpha).
_HEAT_ALPHA = 0.94

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
    """A green→white heat-tile rule for one cell (Excel-style), with dark text.

    The background interpolates soft near-white (worst) → rich green (best); a
    fixed dark text colour stays legible across the whole range, and a faint ring
    seats each value in its own tile. The rule carries no ``state`` condition, so
    it applies in every cell state — paired with the background-preserving state
    reset (see :data:`_STATE_RESET_KEEP_BG`), a click can never blank the tile
    back to the dark panel.
    """
    r, g, b = _mix(_HEAT_LOW, _HEAT_HIGH, intensity)
    rr, rg, rb = _HEAT_RING
    return {
        "if": {"row_index": row_index, "column_id": column_id},
        "backgroundColor": f"rgba({r},{g},{b},{_HEAT_ALPHA})",
        "color": _HEAT_TEXT,
        "borderRadius": "8px",
        "boxShadow": f"inset 0 0 0 1px rgba({rr},{rg},{rb},0.28)",
    }


# How strongly to spread shades by RANK rather than raw value. Pure value scaling
# (0.0) leaves clustered values looking near-identical; pure rank (1.0) gives every
# cell a distinct shade but ignores magnitude. Blending leans toward rank so the
# ordering "pops" — within EACH row the best gets the greenest tile and the worst
# the whitest, with the others clearly stepped between — while a value that is far
# ahead still reads as noticeably deeper.
_RANK_WEIGHT = 0.7


def _blend_with_rank(items: List[tuple]) -> List[tuple]:
    """Blend each ``(key, value_intensity)`` with a rank-spread intensity.

    Ranking is by goodness (higher value-intensity = better); the best cell maps
    to 1.0 and the worst to 0.0 with even spacing, so a row's cells always span
    the full green→white range and are easy to rank even when the raw numbers
    cluster. Ties share an averaged rank; a single cell keeps its value intensity.
    """
    n = len(items)
    if n <= 1:
        return items
    order = sorted(range(n), key=lambda i: items[i][1])  # ascending by goodness
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and items[order[j + 1]][1] == items[order[i]][1]:
            j += 1  # group equal-intensity ties
        avg_rank = ((i + j) / 2.0) / (n - 1)
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return [
        (key, (1.0 - _RANK_WEIGHT) * vi + _RANK_WEIGHT * ranks[idx])
        for idx, (key, vi) in enumerate(items)
    ]


def _heatmap_cells(
    data: Sequence[dict],
    columns: Sequence[dict],
    label_ids: set,
    orient: str,
    lower_better: Iterable[str],
) -> List[dict]:
    """Build per-cell tint rules, normalised (and rank-spread) per-row or per-column."""
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
            scored = [(c, _intensity(v, lo, hi, invert)) for c, v in nums]
            for c, t in _blend_with_rank(scored):
                cells.append(_cell_tint(r, c, t))
    else:  # per-column
        for c in value_cols:
            nums = [(r, _parse_number(row.get(c))) for r, row in enumerate(data)]
            nums = [(r, v) for r, v in nums if v is not None]
            if not nums:
                continue
            vals = [v for _, v in nums]
            lo, hi = min(vals), max(vals)
            scored = [(r, _intensity(v, lo, hi, False)) for r, v in nums]
            for r, t in _blend_with_rank(scored):
                cells.append(_cell_tint(r, c, t))
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

# Background-preserving variant for the heat-map: clicking a cell makes it the
# "active" cell, and the plain reset above would blank its background to
# transparent — turning the tile (with its dark text) black. This variant strips
# only the default selection border/outline and leaves the background alone, so
# the per-cell tile (a stateless rule) shows through in every state.
_STATE_RESET_KEEP_BG = [
    {"if": {"state": "active"}, "border": "0px", "borderBottom": "0px", "boxShadow": "none"},
    {"if": {"state": "selected"}, "border": "0px", "borderBottom": "0px", "boxShadow": "none"},
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

    Always includes the left-aligned label column(s) and an active/selected reset
    so the selection box never flashes. Without the heat-map that reset blanks the
    cell background (the original transparent behaviour). With *heatmap_on* it uses
    the background-preserving reset instead, so clicking a tinted cell keeps its
    tile and dark text rather than clearing to a transparent (black, unreadable)
    cell.
    """
    label_ids = set(label_ids)
    align = [{"if": {"column_id": c}, "textAlign": "left"} for c in label_ids]
    if heatmap_on and data and columns:
        heat = _heatmap_cells(data, columns, label_ids, orient, lower_better)
        return align + heat + _STATE_RESET_KEEP_BG
    return align + _STATE_RESET
