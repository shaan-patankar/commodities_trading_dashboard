"""Tests for dashboard.figures module — verify figure functions return go.Figure objects."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from dashboard.analytics import SeriesPack, compute_series
import numpy as np
import pandas as pd

from dashboard.figures import (
    correlation_matrix_figure,
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
        sp_a = compute_series(sample_df, ["product_a"])
        sp_b = compute_series(sample_df, ["product_b"])
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
        fig = rolling_correlation_figure(sample_df, ["product_a", "product_b"], 20, "Corr")
        assert isinstance(fig, go.Figure)

    def test_single_product_no_error(self, sample_df):
        fig = rolling_correlation_figure(sample_df, ["product_a"], 20, "Corr")
        assert isinstance(fig, go.Figure)


class TestRollingSharpe:
    def test_returns_figure(self, sample_df):
        fig = rolling_sharpe_figure(sample_df, ["product_a", "product_b"], 60, "Sharpe")
        assert isinstance(fig, go.Figure)

    def test_individual_only(self, sample_df):
        fig = rolling_sharpe_figure(
            sample_df, ["product_a"], 60, "Sharpe",
            include_individuals=True, include_aggregate=False,
        )
        assert isinstance(fig, go.Figure)

    def test_aggregate_trace_has_color(self, sample_df):
        """The 'ALL (agg)' trace must have a non-None line color (A5 guard)."""
        fig = rolling_sharpe_figure(
            sample_df, ["product_a", "product_b"], 60, "Sharpe",
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


class TestCorrelationMatrix:
    def test_returns_figure(self, sample_df):
        fig = correlation_matrix_figure(sample_df, ["product_a", "product_b"], "Corr Matrix")
        assert isinstance(fig, go.Figure)
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmaps) == 1
        z = np.asarray(heatmaps[0].z)
        assert z.shape == (2, 2)

    def test_single_product_shows_note(self, sample_df):
        fig = correlation_matrix_figure(sample_df, ["product_a"], "Corr Matrix")
        assert isinstance(fig, go.Figure)
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmaps) == 0

    def test_unknown_products_filtered(self, sample_df):
        fig = correlation_matrix_figure(sample_df, ["product_a", "ghost"], "Corr Matrix")
        assert isinstance(fig, go.Figure)
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmaps) == 0

    def test_three_products_matrix_3x3(self):
        dates = pd.bdate_range("2024-01-02", periods=120, freq="B")
        rng = np.random.default_rng(7)
        n = len(dates)
        df = pd.DataFrame({
            "date": dates,
            "product_a": rng.normal(100, 500, n),
            "product_b": rng.normal(50, 300, n),
            "product_c": rng.normal(75, 400, n),
        })
        fig = correlation_matrix_figure(df, ["product_a", "product_b", "product_c"], "Corr Matrix")
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmaps) == 1
        z = np.asarray(heatmaps[0].z)
        assert z.shape == (3, 3)


class TestPlaceholderFigure:
    def test_returns_figure(self):
        fig = placeholder_figure("Loading...")
        assert isinstance(fig, go.Figure)

    def test_with_subtitle(self):
        fig = placeholder_figure("Loading...", subtitle="Please wait")
        assert isinstance(fig, go.Figure)
