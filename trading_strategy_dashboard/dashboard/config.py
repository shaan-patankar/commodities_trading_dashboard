"""Dashboard configuration constants and defaults.

Centralizes all static configuration for the trading strategy dashboard,
including file paths, graph settings, layout options, theming defaults,
the settings gear SVG icon, and analytics defaults.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"

STRATEGY_FILES: dict[str, Path] = {
    "Momentum": DATA_DIR / "momentum_pnls.csv",
    "Mean Reversion": DATA_DIR / "mean_reversion_pnls.csv",
    "Carry": DATA_DIR / "carry_pnls.csv",
    "Machine Learning": DATA_DIR / "machine_learning_pnls.csv",
    "Short Strangle": DATA_DIR / "short_strangle_pnls.csv",
    "Intraweek Seasonality": DATA_DIR / "intraweek_seasonality_pnls.csv",
}

# Per-strategy daily-returns CSVs for VaR scaling. These mirror STRATEGY_FILES
# with the SAME product column headers, but hold daily *returns* of the
# underlyings (fractional, e.g. 0.012 = 1.2%). Files are optional: VaR scaling
# is simply unavailable for any strategy whose returns file is missing.
RETURNS_FILES: dict[str, Path] = {
    "Momentum": DATA_DIR / "momentum_returns.csv",
    "Mean Reversion": DATA_DIR / "mean_reversion_returns.csv",
    "Carry": DATA_DIR / "carry_returns.csv",
    "Machine Learning": DATA_DIR / "machine_learning_returns.csv",
    "Short Strangle": DATA_DIR / "short_strangle_returns.csv",
    "Intraweek Seasonality": DATA_DIR / "intraweek_seasonality_returns.csv",
}

# ---------------------------------------------------------------------------
# Plotly graph / interaction config
# ---------------------------------------------------------------------------

GRAPH_CONFIG: dict[str, object] = {
    "scrollZoom": True,
    "responsive": True,
    "displaylogo": False,
    "displayModeBar": False,
    "doubleClick": "reset",
    "showTips": True,
}

# ---------------------------------------------------------------------------
# Range selector and layout presets
# ---------------------------------------------------------------------------

RANGE_OPTIONS: list[str] = ["1M", "3M", "YTD", "1Y", "All"]

LAYOUT_OPTIONS: list[dict[str, str]] = [
    {"label": "Default", "value": "default"},
    {"label": "Focused", "value": "focused"},
    {"label": "Analytics", "value": "analytics"},
]

PANEL_KEYS: list[str] = ["equity", "custom", "drawdown", "metrics"]

# ---------------------------------------------------------------------------
# Settings gear SVG icon (URL-encoded inline SVG)
# ---------------------------------------------------------------------------

SETTINGS_GEAR_SVG: str = "data:image/svg+xml;utf8," + quote(
    """
<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2"
    stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"
    xmlns="http://www.w3.org/2000/svg">
    <circle cx="12" cy="12" r="3"/>
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
</svg>
""".strip()
)

# ---------------------------------------------------------------------------
# Base theme / font defaults
# ---------------------------------------------------------------------------

BASE_FONT: dict[str, str] = {
    "family": "Inter, 'Segoe UI', system-ui",
    "color": "#e6e9f0",
}

BASE_BACKGROUND: str = "rgba(0,0,0,0)"

BASE_HOVERLABEL: dict[str, object] = {
    "bgcolor": "#151c2c",
    "bordercolor": "#6f7c95",
    "font": {"color": "#e6e9f0", "size": 12, "family": "Inter, 'Segoe UI', system-ui"},
}

# ---------------------------------------------------------------------------
# Analytics defaults
# ---------------------------------------------------------------------------

DEFAULT_INITIAL_CAPITAL: float = 1_000_000.0
DEFAULT_RF: float = 0.0
DEFAULT_ROLL_WINDOW: int = 252

# ---------------------------------------------------------------------------
# VaR scaling defaults (fixed per spec)
# ---------------------------------------------------------------------------

VAR_Z: float = 1.645              # 95% one-tailed normal quantile (VaR95 = z * sigma)
VAR_WINDOW: int = 20              # rolling volatility window (trading days)
VAR_ALLOC_TOLERANCE: float = 0.01  # allowed deviation of the allocation sum from 100 (%)
DEFAULT_TOTAL_VAR: float = 10_000.0  # default Total VaR Allocation (currency)
