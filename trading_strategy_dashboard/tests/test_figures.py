"""Tests for dashboard.figures module — verify figure functions return go.Figure objects."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from dashboard.analytics import SeriesPack, compute_series
from dashboard.config import DEFAULT_INITIAL_CAPITAL
import numpy as np
import pandas as pd

from dashboard.figures import (
    decimate,
    drawdown_figure,
    equity_figure,
    placeholder_figure,
    rolling_correlation_figure,
    rolling_sharpe_figure,
    seasonality_figure,
)


class TestDecimate:
    def test_small_series_unchanged(self):
        x = pd.Series(pd.date_range("2024-01-01", periods=10))
        y = pd.Series(range(10), dtype="float64")
        dx, dy = decimate(x, y, max_points=1500)
        assert dx is x and dy is y  # returned as-is

    def test_large_series_reduced(self):
        n = 5000
        x = pd.Series(pd.date_range("2010-01-01", periods=n))
        y = pd.Series(np.random.default_rng(0).normal(size=n))
        dx, dy = decimate(x, y, max_points=1000)
        assert len(dx) == len(dy)
        assert len(dy) <= 1002  # ~max_points (2 per bucket) + endpoints
        assert len(dy) < n // 2  # meaningfully reduced

    def test_preserves_global_extremes(self):
        n = 4000
        x = pd.Series(pd.date_range("2010-01-01", periods=n))
        y = np.zeros(n)
        y[1234] = 99.0   # spike up
        y[2500] = -55.0  # spike down
        dx, dy = decimate(x, pd.Series(y), max_points=800)
        assert dy.max() == 99.0
        assert dy.min() == -55.0


class TestEquityFigure:
    def test_returns_figure(self, sample_series_pack):
        fig = equity_figure({"Test": sample_series_pack}, "Equity")
        assert isinstance(fig, go.Figure)

    def test_with_drawdown_overlay(self, sample_series_pack):
        fig = equity_figure(
            {"Test": sample_series_pack},
            "Equity",
            drawdown_series={"Test": sample_series_pack},
        )
        assert isinstance(fig, go.Figure)

    def test_multiple_series(self, sample_df):
        sp_a = compute_series(sample_df, ["product_a"], DEFAULT_INITIAL_CAPITAL)
        sp_b = compute_series(sample_df, ["product_b"], DEFAULT_INITIAL_CAPITAL)
        fig = equity_figure({"A": sp_a, "B": sp_b}, "Multi")
        assert isinstance(fig, go.Figure)
        # Should have traces for each series (equity + hwm per label)
        assert len(fig.data) >= 4


class TestDrawdownFigure:
    def test_returns_figure(self, sample_series_pack):
        fig = drawdown_figure({"Test": sample_series_pack}, "DD")
        assert isinstance(fig, go.Figure)


class TestRollingCorrelationFigure:
    def test_returns_figure(self, sample_df):
        fig = rolling_correlation_figure(sample_df, ["product_a", "product_b"], DEFAULT_INITIAL_CAPITAL, 20, "Corr")
        assert isinstance(fig, go.Figure)

    def test_single_product_no_error(self, sample_df):
        fig = rolling_correlation_figure(sample_df, ["product_a"], DEFAULT_INITIAL_CAPITAL, 20, "Corr")
        assert isinstance(fig, go.Figure)


class TestRollingSharpe:
    def test_returns_figure(self, sample_df):
        fig = rolling_sharpe_figure(sample_df, ["product_a", "product_b"], DEFAULT_INITIAL_CAPITAL, 60, "Sharpe")
        assert isinstance(fig, go.Figure)

    def test_individual_only(self, sample_df):
        fig = rolling_sharpe_figure(
            sample_df, ["product_a"], DEFAULT_INITIAL_CAPITAL, 60, "Sharpe",
            include_individuals=True, include_aggregate=False,
        )
        assert isinstance(fig, go.Figure)

    def test_aggregate_trace_has_color(self, sample_df):
        """The 'ALL (agg)' trace must have a non-None line color (A5 guard)."""
        fig = rolling_sharpe_figure(
            sample_df, ["product_a", "product_b"], DEFAULT_INITIAL_CAPITAL, 60, "Sharpe",
            include_individuals=False, include_aggregate=True,
        )
        agg = [t for t in fig.data if t.name == "ALL (agg)"]
        assert len(agg) == 1
        assert agg[0].line.color is not None


class TestSeasonalityFigure:
    def test_returns_figure(self, sample_df):
        fig = seasonality_figure(sample_df, ["product_a", "product_b"], "Seasonality")
        assert isinstance(fig, go.Figure)

    def test_empty_products_fallback(self, sample_df):
        fig = seasonality_figure(sample_df, [], "Seasonality")
        assert isinstance(fig, go.Figure)


class TestPlaceholderFigure:
    def test_returns_figure(self):
        fig = placeholder_figure("Loading...")
        assert isinstance(fig, go.Figure)

    def test_with_subtitle(self):
        fig = placeholder_figure("Loading...", subtitle="Please wait")
        assert isinstance(fig, go.Figure)
