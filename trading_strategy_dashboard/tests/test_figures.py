"""Tests for dashboard.figures module — verify figure functions return go.Figure objects."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from dashboard.analytics import SeriesPack, compute_series
from dashboard.config import DEFAULT_INITIAL_CAPITAL
from dashboard.figures import (
    drawdown_figure,
    equity_figure,
    placeholder_figure,
    rolling_correlation_figure,
    rolling_sharpe_figure,
    seasonality_figure,
)


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
