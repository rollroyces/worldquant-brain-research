"""Tests for brainlib.gates — the canonical BRAIN submission gate evaluator.

Run with:  python3 -m pytest tests/ -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Make brainlib importable from tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from brainlib import gates
from brainlib.gates import (
    GateConfig,
    evaluate,
    fitness,
    sub_universe_cutoff,
)

# ---------------------------------------------------------------------------
# Fitness formula
# ---------------------------------------------------------------------------


class TestFitnessFormula:
    """The fitness formula is the single most important rule in this repo."""

    def test_basic_floor_case(self):
        # The worked example from the Quantt.co.uk 2026 review:
        # Sharpe=1.4, returns=8%, turnover=30% → fitness ≈ 0.72
        result = fitness(sharpe=1.4, returns=0.08, turnover=0.30)
        assert math.isclose(result, 1.4 * (0.08 / 0.30) ** 0.5, rel_tol=1e-9)
        assert math.isclose(result, 0.7227, abs_tol=0.001)

    def test_floor_engages_at_12_5pct(self):
        # Below 12.5% turnover, the formula uses the floor as the denominator.
        # Sharpe=1.4, returns=8%, turnover=10% → fitness uses max(0.10, 0.125)=0.125
        result = fitness(sharpe=1.4, returns=0.08, turnover=0.10)
        assert math.isclose(result, 1.4 * (0.08 / 0.125) ** 0.5, rel_tol=1e-9)
        assert math.isclose(result, 1.1196, abs_tol=0.001)

    def test_floor_makes_turnover_below_12_5_pct_free(self):
        # Going from 10% to 5% turnover does NOT improve fitness — formula floors.
        at_10 = fitness(sharpe=1.4, returns=0.08, turnover=0.10)
        at_5 = fitness(sharpe=1.4, returns=0.08, turnover=0.05)
        assert math.isclose(at_10, at_5, rel_tol=1e-9)

    def test_halving_turnover_above_floor_multiplies_by_sqrt2(self):
        # Above the floor, every halving multiplies fitness by sqrt(2).
        at_30 = fitness(sharpe=1.4, returns=0.08, turnover=0.30)
        at_15 = fitness(sharpe=1.4, returns=0.08, turnover=0.15)
        assert math.isclose(at_15 / at_30, math.sqrt(2), rel_tol=1e-9)

    def test_zero_turnover_returns_zero(self):
        assert fitness(sharpe=2.0, returns=0.10, turnover=0.0) == 0.0

    def test_negative_returns_take_absolute_value(self):
        # The formula uses |returns| — a negative-returning alpha with
        # positive Sharpe (rare but possible — could be a volatility signal)
        # should still get a non-zero fitness.
        positive = fitness(sharpe=1.5, returns=0.05, turnover=0.15)
        negative = fitness(sharpe=1.5, returns=-0.05, turnover=0.15)
        assert math.isclose(positive, negative, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Sub-universe cutoff
# ---------------------------------------------------------------------------


class TestSubUniverseCutoff:
    """The sub-universe Sharpe cutoff is what most alphas miss silently."""

    def test_5pct_sub_universe_needs_small_sharpe(self):
        # 5% sub-universe → cutoff is 0.75 * sqrt(0.05) ≈ 0.168 of alpha_sharpe
        alpha = 1.4
        cutoff = sub_universe_cutoff(alpha_sharpe=alpha, sub_size=50, full_size=1000)
        assert math.isclose(cutoff, 0.75 * math.sqrt(0.05) * 1.4, rel_tol=1e-9)
        assert cutoff < 0.25  # tiny — small sub-universes are easy

    def test_50pct_sub_universe_needs_half_alpha_sharpe(self):
        cutoff = sub_universe_cutoff(alpha_sharpe=2.0, sub_size=500, full_size=1000)
        assert math.isclose(cutoff, 0.75 * math.sqrt(0.5) * 2.0, rel_tol=1e-9)
        assert math.isclose(cutoff, 1.0607, abs_tol=0.001)

    def test_100pct_sub_universe_needs_75pct_alpha_sharpe(self):
        # Full-universe "sub" needs 75% of alpha Sharpe — that's the scale factor.
        cutoff = sub_universe_cutoff(alpha_sharpe=2.0, sub_size=1000, full_size=1000)
        assert math.isclose(cutoff, 0.75 * 2.0, rel_tol=1e-9)

    def test_zero_size_returns_infinity(self):
        # Degenerate cases shouldn't crash.
        assert sub_universe_cutoff(alpha_sharpe=2.0, sub_size=0, full_size=1000) == math.inf
        assert sub_universe_cutoff(alpha_sharpe=2.0, sub_size=100, full_size=0) == math.inf


# ---------------------------------------------------------------------------
# Gate evaluator
# ---------------------------------------------------------------------------


def passing_metrics(**overrides):
    """A metrics dict that should pass the default US/Delay-1 gates."""
    base = {
        "sharpe": 1.50,
        "fitness": 1.20,
        "turnover": 0.13,
        "self_correlation": 0.45,
        "concentration": 0.07,
        "returns": 0.08,
        "drawdown": 0.05,
    }
    base.update(overrides)
    return base


class TestEvaluatePasses:
    """Strong alphas should pass cleanly."""

    def test_strong_alpha_passes(self):
        passes, reasons = evaluate(passing_metrics())
        assert passes is True
        assert reasons == []

    def test_exact_threshold_sharpe_passes(self):
        # At exactly 1.25 the comparison is `sharpe < min_sharpe`, so 1.25 passes.
        passes, _ = evaluate(passing_metrics(sharpe=1.25))
        assert passes is True

    def test_exact_threshold_turnover_lower_bound_passes(self):
        passes, _ = evaluate(passing_metrics(turnover=0.01))
        assert passes is True

    def test_exact_threshold_turnover_upper_bound_passes(self):
        passes, _ = evaluate(passing_metrics(turnover=0.70))
        assert passes is True


class TestEvaluateFailures:
    """Each gate should fire on the right failure mode."""

    def test_sharpe_too_low(self):
        passes, reasons = evaluate(passing_metrics(sharpe=1.20))
        assert passes is False
        assert any("Sharpe" in r and "1.200 < 1.25" in r for r in reasons)

    def test_fitness_too_low(self):
        passes, reasons = evaluate(passing_metrics(fitness=0.95))
        assert passes is False
        assert any("Fitness" in r for r in reasons)

    def test_turnover_below_floor(self):
        passes, reasons = evaluate(passing_metrics(turnover=0.005))
        assert passes is False
        assert any("Turnover" in r for r in reasons)

    def test_turnover_above_ceiling(self):
        passes, reasons = evaluate(passing_metrics(turnover=0.85))
        assert passes is False
        assert any("Turnover" in r for r in reasons)

    def test_self_correlation_too_high_without_override(self):
        passes, reasons = evaluate(passing_metrics(self_correlation=0.85, sharpe=1.30))
        assert passes is False
        assert any("Self-Corr" in r for r in reasons)

    def test_self_correlation_too_high_with_sharpe_override(self):
        # Sharpe 1.40 > override 1.375 → self-corr override engages, passes.
        passes, reasons = evaluate(passing_metrics(self_correlation=0.85, sharpe=1.40))
        assert passes is True

    def test_self_correlation_at_exact_threshold_passes(self):
        # 0.70 == max_self_corr → comparison is strict >, so passes.
        passes, _ = evaluate(passing_metrics(self_correlation=0.70))
        assert passes is True

    def test_concentration_too_high(self):
        passes, reasons = evaluate(passing_metrics(concentration=0.15))
        assert passes is False
        assert any("Concentration" in r for r in reasons)

    def test_multiple_failures_all_reported(self):
        # A genuinely bad alpha should fail multiple gates at once.
        passes, reasons = evaluate(
            {
                "sharpe": 0.80,
                "fitness": 0.50,
                "turnover": 0.005,
                "self_correlation": 0.90,
                "concentration": 0.20,
            }
        )
        assert passes is False
        assert len(reasons) >= 4


# ---------------------------------------------------------------------------
# Custom gate configs (region/delay variants)
# ---------------------------------------------------------------------------


class TestCustomGateConfigs:
    """Per-region / per-delay configs should override the defaults."""

    def test_stricter_min_sharpe(self):
        strict = GateConfig(min_sharpe=1.50)
        passes, _ = evaluate(passing_metrics(sharpe=1.30), gates=strict)
        assert passes is False

    def test_looser_min_sharpe(self):
        loose = GateConfig(min_sharpe=1.10)
        passes, _ = evaluate(passing_metrics(sharpe=1.20), gates=loose)
        assert passes is True

    def test_china_style_stricter_turnover_ceiling(self):
        # CHN is reportedly stricter.
        chn = GateConfig(turnover_max=0.60)
        passes, reasons = evaluate(passing_metrics(turnover=0.65), gates=chn)
        assert passes is False
        assert any("Turnover" in r for r in reasons)


# ---------------------------------------------------------------------------
# Round-trip helpers
# ---------------------------------------------------------------------------


class TestAsDict:
    def test_default_gates_serialize(self):
        d = gates.as_dict()
        assert d["min_sharpe"] == 1.25
        assert d["min_fitness"] == 1.0
        assert d["max_self_corr"] == 0.70
        assert "turnover_floor" in d
        assert "self_corr_sharpe_override" in d
