"""Tests for module-level helpers in dashboard.callbacks.

These exercise the pure metrics-table builders directly (no Dash app needed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dashboard.callbacks import build_metrics_table, build_portfolio_metrics_table


@pytest.fixture
def products_df():
    # Mixed-sign daily PnL so the equity curve has real drawdowns; constant
    # positive PnL would give max_dd == 0 and trip compute_metrics' Calmar divide.
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=n, freq="B"),
        "Brent_M3": rng.normal(100, 500, n),
        "Gasoil_M3": rng.normal(50, 300, n),
    })


class TestBuildMetricsTable:
    def test_all_selection_single_column(self, products_df):
        columns, rows = build_metrics_table(products_df, ["ALL"], ["Brent_M3", "Gasoil_M3"])
        assert columns[0]["id"] == "Metric"
        assert any(c["name"] == "All" for c in columns)
        assert len(rows) > 0

    def test_individual_products_columns(self, products_df):
        columns, rows = build_metrics_table(
            products_df, ["Brent_M3", "Gasoil_M3"], ["Brent_M3", "Gasoil_M3"]
        )
        col_names = {c["name"] for c in columns}
        assert "Brent M3" in col_names and "Gasoil M3" in col_names

    def test_empty_selection_informational_row(self, products_df):
        """No products selected -> a single informational row, not a blank table."""
        columns, rows = build_metrics_table(products_df, [], ["Brent_M3", "Gasoil_M3"])
        assert len(rows) == 1
        assert rows[0]["Metric"] == "No products selected"

    def test_unknown_products_filtered(self, products_df):
        columns, rows = build_metrics_table(products_df, ["ghost"], ["Brent_M3"])
        assert rows[0]["Metric"] == "No products selected"


class TestBuildPortfolioMetricsTable:
    @pytest.fixture
    def portfolio_df(self):
        # Mixed-sign PnL so the equity curve drawsdown (avoids the Calmar divide).
        rng = np.random.default_rng(1)
        n = 40
        return pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=n, freq="B"),
            "Momentum": rng.normal(100, 500, n),
            "Carry": rng.normal(40, 300, n),
        })

    def test_all_strategies_column(self, portfolio_df):
        columns, rows = build_portfolio_metrics_table(["ALL"], ["Momentum", "Carry"], portfolio_df)
        assert any(c["name"] == "All Strategies" for c in columns)
        assert len(rows) > 0

    def test_empty_selection_informational_row(self, portfolio_df):
        columns, rows = build_portfolio_metrics_table([], ["Momentum", "Carry"], portfolio_df)
        assert len(rows) == 1
        assert rows[0]["Metric"] == "No strategies selected"
