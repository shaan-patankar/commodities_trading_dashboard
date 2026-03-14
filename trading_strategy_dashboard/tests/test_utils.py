"""Tests for dashboard.utils module."""

from __future__ import annotations

from dashboard.utils import format_product_label


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
