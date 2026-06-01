"""Tests for dashboard.utils module."""

from __future__ import annotations

import pandas as pd

from dashboard.utils import format_product_label, valid_product_columns


class TestValidProductColumns:
    def _df(self):
        return pd.DataFrame({"date": [1, 2], "a": [1, 2], "b": [3, 4]})

    def test_none_returns_all_non_date(self):
        assert valid_product_columns(self._df()) == ["a", "b"]

    def test_subset_preserves_order(self):
        assert valid_product_columns(self._df(), ["b", "a"]) == ["b", "a"]

    def test_unknown_products_dropped(self):
        assert valid_product_columns(self._df(), ["a", "ghost"]) == ["a"]

    def test_date_never_returned(self):
        assert "date" not in valid_product_columns(self._df(), ["date", "a"])

    def test_empty_when_no_match(self):
        assert valid_product_columns(self._df(), ["ghost"]) == []

    def test_none_df_returns_empty(self):
        assert valid_product_columns(None) == []


class TestFormatProductLabel:
    def test_snake_case_to_title(self):
        assert format_product_label("crude_oil") == "Crude Oil"

    def test_single_word(self):
        assert format_product_label("gold") == "Gold"

    def test_multiple_underscores(self):
        assert format_product_label("natural_gas_futures") == "Natural Gas Futures"

    def test_already_titled(self):
        assert format_product_label("Gold") == "Gold"

    def test_empty_string(self):
        assert format_product_label("") == ""
