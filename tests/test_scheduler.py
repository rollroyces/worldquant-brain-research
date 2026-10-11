"""Tests for scripts/iqc_scheduler.py — rotation + filtering logic.

Run with:  python3 -m pytest tests/ -v
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

from iqc_scheduler import (
    DEFAULT_POOL,
    _pick_template_by_region,
    _pick_template_for_family,
    build_schedule,
)


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

class TestPickTemplateForFamily:
    def test_avoids_recently_used_template(self):
        from iqc_scheduler import Template
        candidates = [
            Template("PV", "USA", "TOP3000", 1, "a", "A", 1.4, 0.1),
            Template("PV", "ASI", "TOP1000", 1, "b", "B", 1.4, 0.1),
            Template("PV", "CHN", "TOP3000", 0, "c", "C", 1.4, 0.1),
        ]
        picked = _pick_template_for_family(candidates, "A")
        assert picked.settings_name != "A"

    def test_falls_back_when_pool_too_small(self):
        from iqc_scheduler import Template
        candidates = [Template("PV", "USA", "TOP3000", 1, "a", "A", 1.4, 0.1)]
        picked = _pick_template_for_family(candidates, "A")
        # Only one option — must pick it even though it equals last used.
        assert picked.settings_name == "A"


class TestPickTemplateByRegion:
    def test_avoids_recently_used_region(self):
        from iqc_scheduler import Template
        candidates = [
            Template("PV", "USA", "TOP3000", 1, "a", "A", 1.4, 0.1),
            Template("PV", "ASI", "TOP1000", 1, "b", "B", 1.4, 0.1),
        ]
        picked = _pick_template_by_region(candidates, "USA")
        assert picked.region != "USA"

    def test_falls_back_when_pool_too_small(self):
        from iqc_scheduler import Template
        candidates = [Template("PV", "USA", "TOP3000", 1, "a", "A", 1.4, 0.1)]
        picked = _pick_template_by_region(candidates, "USA")
        assert picked.region == "USA"


# ---------------------------------------------------------------------------
# build_schedule end-to-end
# ---------------------------------------------------------------------------

class TestBuildSchedule:
    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="No templates match"):
            build_schedule(
                start=dt.date(2026, 1, 1), days=7,
                families=["PV"], regions=["ZZ"], per_day=1,
                skip_weekends=False, pool=DEFAULT_POOL,
            )

    def test_per_day_must_be_positive(self):
        with pytest.raises(ValueError, match="--per-day"):
            build_schedule(
                start=dt.date(2026, 1, 1), days=7,
                families=["PV"], regions=["USA"], per_day=0,
                skip_weekends=False, pool=DEFAULT_POOL,
            )

    def test_per_day_one_rotates_families(self):
        schedule = build_schedule(
            start=dt.date(2026, 1, 1), days=15,
            families=["PV", "Fundamental", "Analyst"],
            regions=["USA", "ASI"], per_day=1, skip_weekends=False,
            pool=DEFAULT_POOL,
        )
        # 15 days, 1 per day → 15 entries.
        assert len(schedule) == 15
        # Same family should never appear on consecutive days.
        families = [row["family"] for row in schedule]
        for i in range(1, len(families)):
            assert families[i] != families[i - 1], \
                f"family {families[i]} repeated on day {i}"

    def test_per_day_two_different_families_same_day(self):
        schedule = build_schedule(
            start=dt.date(2026, 1, 1), days=5,
            families=["PV", "Fundamental", "Analyst", "Sentiment"],
            regions=["USA"], per_day=2, skip_weekends=False,
            pool=DEFAULT_POOL,
        )
        # Each day should have 2 entries with different families.
        dates = sorted({row["date"] for row in schedule})
        for d in dates:
            day_entries = [r for r in schedule if r["date"] == d]
            assert len(day_entries) == 2
            assert day_entries[0]["family"] != day_entries[1]["family"]

    def test_skip_weekends(self):
        # 2026-01-01 is a Thursday. 14 days includes 4 weekend days.
        schedule = build_schedule(
            start=dt.date(2026, 1, 1), days=14,
            families=["PV"], regions=["USA"], per_day=1,
            skip_weekends=True, pool=DEFAULT_POOL,
        )
        for row in schedule:
            d = dt.date.fromisoformat(row["date"])
            assert d.weekday() < 5, f"{d} is a weekend day"

    def test_template_not_repeated_consecutively(self):
        schedule = build_schedule(
            start=dt.date(2026, 1, 1), days=30,
            families=["PV", "Fundamental", "Analyst", "Sentiment", "Composite"],
            regions=["USA", "ASI", "CHN"], per_day=1, skip_weekends=False,
            pool=DEFAULT_POOL,
        )
        # Same template (settings_name) should not appear twice in a row.
        # (Family always differs; within the same family the settings rotate.)
        for i in range(1, len(schedule)):
            if schedule[i]["family"] == schedule[i - 1]["family"]:
                assert schedule[i]["settings"] != schedule[i - 1]["settings"], \
                    f"template {schedule[i]['settings']} repeated"

    def test_summary_columns(self):
        schedule = build_schedule(
            start=dt.date(2026, 1, 1), days=3,
            families=["PV"], regions=["USA"], per_day=1,
            skip_weekends=False, pool=DEFAULT_POOL,
        )
        required = {"day_offset", "date", "slot", "family", "region",
                    "universe", "delay", "expression", "settings",
                    "expected_sharpe", "expected_turnover", "notes"}
        for row in schedule:
            assert required.issubset(row.keys())
