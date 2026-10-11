"""Tests for CSV-grid loading in scripts/brain_harness.py.

Regression coverage for the bug where unquoted expressions containing
commas (e.g. `ts_rank(net_income/cap, 252)`) caused csv.DictReader to
split the expression across multiple columns and explode the batch
job with ValueError on int() conversion.

The fix: expressions in alpha_grid_example.csv must be wrapped in
double-quotes so Python's csv module treats the inner commas as
literal characters.

These tests guard against:
  - the example CSV accidentally dropping the quotes
  - _load_grid() failing on a row with an internal comma
  - the batch job crashing with int() errors
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

from brain_harness import _load_grid


EXAMPLE_CSV = Path(__file__).resolve().parent.parent / "examples" / "alpha_grid_example.csv"


class TestExampleCsvIsLoadable:
    """The shipped example CSV must parse without errors."""

    def test_csv_file_exists(self):
        assert EXAMPLE_CSV.exists(), "examples/alpha_grid_example.csv missing"

    def test_csv_loads_n_rows(self):
        rows = _load_grid(str(EXAMPLE_CSV))
        assert len(rows) == 16

    def test_csv_expressions_contain_internal_commas(self):
        # Sanity: several expressions in the file DO contain commas; if
        # the file were unquoted, csv.DictReader would split them.
        rows = _load_csv_rows()
        comma_rows = [r for r in rows if "," in r["expression"]]
        assert len(comma_rows) >= 4, (
            "Expected at least 4 expressions with internal commas — "
            "if this fails, the file may have been silently corrupted."
        )

    def test_csv_expressions_parse_intact(self):
        # Every expression should be a syntactically complete string with
        # balanced parens — the bug we hit was the parser eating part of
        # the expression as a "column" and leaving unbalanced fragments.
        rows = _load_csv_rows()
        for r in rows:
            expr = r["expression"]
            assert expr.count("(") == expr.count(")"), (
                f"Expression has unbalanced parens: {expr!r}"
            )


class TestLoadGridQuotingBehavior:
    """_load_grid uses csv.DictReader which respects quoting."""

    def test_unquoted_csv_with_internal_commas_misparses(self, tmp_path):
        # Demonstrates the original bug: a CSV with an unquoted
        # comma-in-expression row is split into multiple columns.
        bad = tmp_path / "bad.csv"
        bad.write_text(
            "expression,region,delay\n"
            "ts_rank(net_income/cap,252),USA,1\n"  # unquoted comma in expression
        )
        rows = _load_grid(str(bad))
        # Misparsed: "ts_rank(net_income/cap" became the expression,
        # "252)" became the region, "USA" became the delay.
        assert "252)" not in rows[0]["expression"]
        assert rows[0]["region"] == "252)"

    def test_quoted_csv_with_internal_commas_parses(self, tmp_path):
        # The fix: quoted expressions parse correctly.
        good = tmp_path / "good.csv"
        good.write_text(
            "expression,region,delay\n"
            '"ts_rank(net_income/cap,252)",USA,1\n'
        )
        rows = _load_grid(str(good))
        assert rows[0]["expression"] == "ts_rank(net_income/cap,252)"
        assert rows[0]["region"] == "USA"
        assert rows[0]["delay"] == "1"


def _load_csv_rows() -> list[dict[str, str]]:
    """Load the example CSV using csv.DictReader directly (no harness)."""
    with open(EXAMPLE_CSV) as f:
        return list(csv.DictReader(f))
