"""Tests for dashboard.data module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from dashboard.data import products_for_strategy, read_strategy_csv


class TestReadStrategyCsv:
    def test_reads_valid_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("date,product_a,product_b\n2024-01-01,100,200\n2024-01-02,150,250\n")
        df = read_strategy_csv(csv_path)
        assert "date" in df.columns
        assert "product_a" in df.columns
        assert len(df) == 2
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_auto_detects_Date_column(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("Date,product_a\n2024-01-01,100\n")
        df = read_strategy_csv(csv_path)
        assert "date" in df.columns

    def test_auto_detects_datetime_column(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("datetime,product_a\n2024-01-01,100\n")
        df = read_strategy_csv(csv_path)
        assert "date" in df.columns

    def test_raises_on_missing_date_column(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("foo,bar\n1,2\n")
        with pytest.raises(ValueError, match="Could not find a date column"):
            read_strategy_csv(csv_path)

    def test_non_numeric_coerced_to_zero(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("date,product_a\n2024-01-01,abc\n2024-01-02,100\n")
        df = read_strategy_csv(csv_path)
        assert df["product_a"].iloc[0] == 0.0
        assert df["product_a"].iloc[1] == 100.0

    def test_sorts_by_date(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("date,product_a\n2024-01-03,300\n2024-01-01,100\n2024-01-02,200\n")
        df = read_strategy_csv(csv_path)
        assert df["date"].is_monotonic_increasing

    def test_drops_invalid_dates(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("date,product_a\n2024-01-01,100\nnot-a-date,200\n2024-01-02,300\n")
        df = read_strategy_csv(csv_path)
        assert len(df) == 2


class TestProductsForStrategy:
    def test_returns_product_columns(self):
        df = pd.DataFrame({"date": [1], "crude_oil": [100], "gold": [200]})
        strategies = {"Momentum": df}
        result = products_for_strategy(strategies, "Momentum")
        assert set(result) == {"crude_oil", "gold"}

    def test_missing_strategy_returns_empty(self):
        result = products_for_strategy({}, "NonExistent")
        assert result == []

    def test_excludes_date_column(self):
        df = pd.DataFrame({"date": [1], "product_a": [100]})
        result = products_for_strategy({"S": df}, "S")
        assert "date" not in result
