"""Application entry point for the trading strategy dashboard.

Creates the Dash application, loads strategy data from CSV files,
builds the UI layout, registers all callbacks, and exposes the WSGI
``server`` object for deployment.

Run directly with ``python app.py`` to launch the development server
on ``http://127.0.0.1:8050``.
"""

from __future__ import annotations

import logging

import dash
import dash_bootstrap_components as dbc

from dashboard.callbacks import register_callbacks
from dashboard.config import RETURNS_FILES, STRATEGY_FILES
from dashboard.data import load_returns, load_strategies, products_for_strategy
from dashboard.layout import build_layout

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

STRATEGIES = load_strategies(STRATEGY_FILES, logger)
RETURNS = load_returns(RETURNS_FILES, logger)
STRATEGY_NAMES = list(STRATEGIES.keys())
DEFAULT_STRATEGY = "Portfolio"

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

external_stylesheets = [dbc.themes.DARKLY]
app: dash.Dash = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "Trading Dashboard"
server = app.server

products = products_for_strategy(STRATEGIES, DEFAULT_STRATEGY)
app.layout = build_layout(STRATEGY_NAMES, DEFAULT_STRATEGY, products)

register_callbacks(app, STRATEGIES, RETURNS)

# ---------------------------------------------------------------------------
# Development server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # threaded=True lets the panel callbacks triggered by one interaction be
    # handled concurrently instead of serialised on a single worker thread.
    app.run(debug=False, host="127.0.0.1", port=8050, threaded=True)
