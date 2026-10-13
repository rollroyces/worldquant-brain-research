"""Tests for scripts/cross_validate.py and scripts/parameter_sweep.py.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cross_validate import (
    _annualized_return,
    _concentration,
    _max_drawdown,
    _sharpe,
    _turnover,
    cross_validate,
)
from parameter_sweep import fragility_summary, sweep

# ---------------------------------------------------------------------------
# Cross-validate helpers
# ---------------------------------------------------------------------------


class TestSharpeHelper:
    def test_short_array(self):
        assert _sharpe(np.array([0.01])) == 0.0

    def test_constant_returns_zero(self):
        # Zero std → 0 Sharpe (no signal).
        assert _sharpe(np.full(100, 0.001)) == 0.0

    def test_positive_sharpe(self):
        np.random.seed(0)
        rets = np.random.normal(0.001, 0.01, 252)
        assert _sharpe(rets) > 0


class TestAnnualizedReturn:
    def test_empty_array(self):
        assert _annualized_return(np.array([])) == 0.0


class TestMaxDrawdown:
    def test_monotonic_increase_zero_dd(self):
        rets = np.full(100, 0.001)
        assert _max_drawdown(rets) == 0.0

    def test_handles_numpy_array(self):
        # Regression: pandas .cummax() would crash on this.
        rets = np.array([0.05, -0.10, 0.02, 0.03, -0.01])
        result = _max_drawdown(rets)
        assert isinstance(result, float)
        assert result <= 0


class TestTurnover:
    def test_static_weights_zero(self):
        w = np.ones((10, 5)) * 0.1
        assert _turnover(w) == 0.0


class TestConcentration:
    def test_max_abs_weight(self):
        assert _concentration(np.array([0.1, 0.3, 0.05])) == 0.3


# ---------------------------------------------------------------------------
# Cross-validate end-to-end
# ---------------------------------------------------------------------------


def _build_panel(
    n_days: int = 200, n_stocks: int = 20, is_strength: float = 0.001, os_strength: float = 0.0005
) -> pd.DataFrame:
    """Build a deterministic panel where IS has stronger returns than OS."""
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    stock_ids = [f"S{i:03d}" for i in range(n_stocks)]
    subindustries = ["Tech", "Finance", "Healthcare"] * 7
    rows = []
    for d_idx, d in enumerate(dates):
        strength = is_strength if d_idx < int(n_days * 0.7) else os_strength
        for s_idx, s in enumerate(stock_ids):
            sig = np.sin(d_idx / 10 + s_idx) * 0.3 + 0.5
            ret = strength * sig + np.random.RandomState(d_idx * 100 + s_idx).normal(0, 0.008)
            rows.append(
                {
                    "date": d,
                    "stock_id": s,
                    "subindustry": subindustries[s_idx],
                    "return": ret,
                    "alpha_test": sig,
                }
            )
    return pd.DataFrame(rows)


class TestCrossValidateEndToEnd:
    def test_split_fractions_produce_two_halves(self):
        panel = _build_panel()
        is_m, os_m, ratio = cross_validate(panel, "alpha_test", split_date=None, split_fraction=0.7)
        assert is_m.n_days > 0
        assert os_m.n_days > 0
        # IS should be roughly 70% of the dates, OS 30%.
        total = is_m.n_days + os_m.n_days
        assert 0.6 * total < is_m.n_days < 0.8 * total

    def test_split_date_produces_two_halves(self):
        panel = _build_panel()
        is_m, os_m, _ = cross_validate(
            panel, "alpha_test", split_date="2024-06-01", split_fraction=0.7
        )
        assert is_m.n_days > 0
        assert os_m.n_days > 0

    def test_empty_split_raises(self):
        # Split-date outside the panel range → one half empty.
        panel = _build_panel(n_days=200, n_stocks=20)
        with pytest.raises(ValueError, match="empty half"):
            cross_validate(panel, "alpha_test", split_date="2099-01-01", split_fraction=0.7)

    def test_is_stronger_than_os_when_constructed(self):
        # When IS has 5× the strength of OS, IS Sharpe should clearly
        # be higher than OS Sharpe (this is the overfit-alpha scenario).
        # We check the relative ordering, not the exact ratio — OS can
        # legitimately have a small or negative Sharpe due to random noise.
        panel = _build_panel(n_days=400, is_strength=0.002, os_strength=0.0002)
        is_m, os_m, _ = cross_validate(panel, "alpha_test", split_date=None, split_fraction=0.7)
        assert is_m.sharpe > os_m.sharpe, (
            f"IS ({is_m.sharpe}) should be stronger than OS ({os_m.sharpe}) " f"by construction"
        )

    def test_metrics_are_finite(self):
        panel = _build_panel()
        is_m, os_m, _ = cross_validate(panel, "alpha_test", split_date=None, split_fraction=0.7)
        for m in (is_m, os_m):
            assert np.isfinite(m.sharpe), f"Sharpe not finite: {m.sharpe}"
            assert np.isfinite(m.fitness)
            assert np.isfinite(m.turnover)


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------


class TestParameterSweep:
    def test_sweep_returns_results_in_mock_mode(self):
        results = sweep(
            expression="rank(close - ts_mean(close, 30))",
            region="USA",
            universe="TOP3000",
            delay=1,
            decays=[2, 4],
            truncations=[0.05, 0.08],
            neutralizations=["Market", "Subindustry"],
            mock=True,
        )
        # 2 × 2 × 2 = 8 combinations
        assert len(results) == 8
        for r in results:
            assert isinstance(r.sharpe, float)
            assert isinstance(r.passes, bool)

    def test_sweep_grid_size_is_product(self):
        results = sweep(
            expression="rank(close)",
            region="USA",
            universe="TOP3000",
            delay=1,
            decays=[2, 4, 6],
            truncations=[0.05, 0.08, 0.10],
            neutralizations=["Market"],
            mock=True,
        )
        assert len(results) == 3 * 3 * 1  # 9 combos

    def test_sweep_results_have_all_attributes(self):
        results = sweep(
            expression="rank(close)",
            region="USA",
            universe="TOP3000",
            delay=1,
            decays=[2],
            truncations=[0.05],
            neutralizations=["Market"],
            mock=True,
        )
        r = results[0]
        assert hasattr(r, "decay")
        assert hasattr(r, "truncation")
        assert hasattr(r, "neutralization")
        assert hasattr(r, "sharpe")
        assert hasattr(r, "fitness")
        assert hasattr(r, "turnover")
        assert hasattr(r, "concentration")
        assert hasattr(r, "passes")
        assert hasattr(r, "reasons")


class TestFragilitySummary:
    def test_empty_results(self):
        summary = fragility_summary([])
        assert summary["n_combos"] == 0
        assert summary["fragile"] is True

    def test_pass_rate_threshold(self):
        # All-passing result is not fragile.
        results = [type("R", (), {"sharpe": 1.5, "passes": True})() for _ in range(10)]
        summary = fragility_summary(results)
        assert summary["pass_rate"] == 1.0
        assert summary["fragile"] is False

    def test_low_pass_rate_is_fragile(self):
        results = [type("R", (), {"sharpe": 0.8, "passes": False})() for _ in range(10)]
        summary = fragility_summary(results)
        assert summary["pass_rate"] == 0.0
        assert summary["fragile"] is True

    def test_median_with_odd_count(self):
        results = [type("R", (), {"sharpe": s, "passes": True})() for s in [1.0, 2.0, 3.0]]
        summary = fragility_summary(results)
        assert summary["median_sharpe"] == 2.0

    def test_median_with_even_count(self):
        results = [type("R", (), {"sharpe": s, "passes": True})() for s in [1.0, 2.0, 3.0, 4.0]]
        summary = fragility_summary(results)
        assert summary["median_sharpe"] == 2.5  # avg of 2.0 and 3.0
