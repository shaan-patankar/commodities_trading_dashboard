"""Theme-aware ``dash_table`` style builders.

Single source of truth for the metrics / CSV DataTable cell and header styles so
the initial (dark) layout render and the theme-switch callback stay in sync.
Dark ("deep") output is byte-identical to the original inline style dicts; the
"light" branch swaps only the text/header colours so light mode is legible.
"""

from __future__ import annotations

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
