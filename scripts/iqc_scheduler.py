#!/usr/bin/env python3
"""
iqc_scheduler.py — WorldQuant BRAIN / IQC submission scheduler.

Generates a submission schedule that:
  1. Rotates across data families to minimize Self-Correlation
     (each consecutive submission comes from a different family).
  2. Rotates across regions to diversify IQC score.
  3. Respects the daily submission cap (≤ 4 alphas/day total — 2 standard
     + 1 SuperAlpha + buffer; the platform may change this).
  4. Pulls from a pool of templates (one per family × region combination).
  5. Outputs a CSV with: date, family, region, expression, settings, etc.

This is a planner, not a runner. Use brain_harness.py to actually submit.

Examples:
  # 60-day schedule starting tomorrow
  python iqc_scheduler.py --start 2026-11-01 --days 60

  # Custom pool of families and regions
  python iqc_scheduler.py \\
      --families PV,Fundamental,Analyst,Sentiment \\
      --regions USA,ASI,CHN,EMEA,GLB \\
      --output my_iqc_plan.csv

  # Skip weekends (BRAIN still accepts weekend submissions but doesn't score)
  python iqc_scheduler.py --start 2026-11-01 --days 60 --skip-weekends

  # Up to 2 submissions per day (still under the daily cap)
  python iqc_scheduler.py --start 2026-11-01 --days 60 --per-day 2
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Template pool
# ---------------------------------------------------------------------------

@dataclass
class Template:
    family: str
    region: str
    universe: str
    delay: int
    expression: str
    settings_name: str
    expected_sharpe: float
    expected_turnover: float
    notes: str = ""


# Default pool: 4 families × 5 regions = 14 base templates.
DEFAULT_POOL = [
    # ----- PV family -----
    Template("PV", "USA", "TOP3000", 1,
             "rank(ts_mean(close, 30) - close)",
             "PV1_baseline", 1.45, 0.20,
             "Classic 30-day reversion; start here"),
    Template("PV", "ASI", "TOP1000", 1,
             "event=volume>adv20; alpha=-ts_delta(close,5); trade_when(event, alpha, -1)",
             "PV2_volume_gated", 1.50, 0.15,
             "Volume-gated reversion; works on liquid Asian markets"),
    Template("PV", "CHN", "TOP3000", 0,
             "(vwap-close)/vwap",
             "PV3_vwap_reversion", 1.40, 0.10,
             "Intraday VWAP reversion; CHN-specific"),

    # ----- Fundamental family -----
    Template("Fundamental", "USA", "TOP3000", 1,
             "ts_rank(cashflow_op/cap, 60); group_rank(alpha, subindustry)",
             "F1_op_yield", 1.40, 0.08,
             "Operating yield; quality + value"),
    Template("Fundamental", "EMEA", "TOP1000", 1,
             "ts_rank(net_income/cap, 252)",
             "F2_earnings_yield", 1.20, 0.06,
             "Earnings yield; slower signal"),
    Template("Fundamental", "GLB", "TOP2000", 1,
             "ts_rank(revenue/assets, 250)",
             "F3_asset_turnover", 1.25, 0.07,
             "Asset turnover; efficiency factor"),

    # ----- Analyst family -----
    Template("Analyst", "USA", "TOP1000", 1,
             "ts_rank(anl14_actvalue_eps_fp0, 60)",
             "A1_earnings_revision", 1.55, 0.15,
             "Earnings revision; high Sharpe"),
    Template("Analyst", "ASI", "TOP500", 1,
             "rank(anl13_eps_est_std/anl13_eps_est_mean)",
             "A2_dispersion", 1.30, 0.12,
             "Estimate dispersion; uncertainty filter"),
    Template("Analyst", "GLB", "TOP1000", 1,
             "rank(target_price/close - 1)",
             "A3_target_upside", 1.30, 0.10,
             "Target price upside"),

    # ----- Sentiment family -----
    Template("Sentiment", "USA", "TOP3000", 1,
             "ts_rank(news_polarity, 5)",
             "S1_news_polarity", 1.40, 0.20,
             "News polarity; high turnover"),
    Template("Sentiment", "CHN", "TOP3000", 0,
             "rank(-mdl175_volatility*log(volume))*(1+group_rank(mdl175_revenuettm, sector))",
             "S4_chn_turnover_vol", 1.35, 0.15,
             "CHN-specific turnover vol + revenue"),
    Template("Sentiment", "EMEA", "TOP1000", 1,
             "rank(implied_vol - realized_vol)",
             "S3_ivol_rvol", 1.30, 0.12,
             "IV minus RV; options data"),

    # ----- Composite blends -----
    Template("Composite", "USA", "TOP3000", 1,
             "0.5*ts_rank(cashflow_op/cap, 60) + 0.5*ts_rank(book_value/cap, 252)",
             "C1_quality_value", 1.50, 0.08,
             "Quality + value blend; pushes Sharpe"),
    Template("Composite", "GLB", "TOP2000", 1,
             "ts_rank(anl14_actvalue_eps_fp0, 60)*ts_rank(cashflow_op/cap, 60)",
             "C3_revisions_quality", 1.45, 0.12,
             "Earnings revision × quality"),
]


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def _pick_template_for_family(candidates: list[Template],
                              last_used_settings: str | None) -> Template:
    """Pick the template in `candidates` whose settings_name is not the
    most-recently-used one. Falls back to the first candidate if the pool
    is too small to avoid the previous template."""
    if not candidates:
        raise ValueError("No templates available for this family.")
    for t in candidates:
        if t.settings_name != last_used_settings:
            return t
    return candidates[0]


def _pick_template_by_region(candidates: list[Template],
                            last_used_region: str | None) -> Template:
    """Pick the template whose region differs from the most-recently-used
    region. Falls back to the first candidate if necessary."""
    if not candidates:
        raise ValueError("No templates available for this family.")
    for t in candidates:
        if t.region != last_used_region:
            return t
    return candidates[0]


def build_schedule(start: dt.date, days: int, families: list[str],
                   regions: list[str], per_day: int, skip_weekends: bool,
                   pool: list[Template]) -> list[dict]:
    """Build a daily submission plan that rotates families, regions, and
    individual templates.

    Strategy:
      - Each day emits 1..per_day submissions, each from a different family.
      - Across days, the same family only repeats when every other family
        has been used more recently.
      - Within a family, we avoid picking the same settings_name twice in
        a row (different skeleton beats same skeleton).

    Returns a list of dicts ready to be written to CSV.
    """
    if per_day < 1:
        raise ValueError("--per-day must be >= 1.")

    pool = [t for t in pool if t.family in families and t.region in regions]
    if not pool:
        raise ValueError("No templates match the family/region filter.")

    by_family: dict[str, list[Template]] = {}
    for t in pool:
        by_family.setdefault(t.family, []).append(t)
    family_order = list(by_family.keys())

    # Per-day tracking.
    last_template_by_family: dict[str, str | None] = {f: None for f in family_order}
    last_region_by_family: dict[str, str | None] = {f: None for f in family_order}
    # Across-day tracking.
    last_used_day: dict[str, int] = {f: -10**9 for f in family_order}

    schedule: list[dict] = []
    current_day = start
    end_day = start + dt.timedelta(days=days)
    day_idx = 0

    while current_day < end_day:
        if skip_weekends and current_day.weekday() >= 5:
            current_day += dt.timedelta(days=1)
            continue

        # Within a day, emit `per_day` submissions from different families.
        # Re-rank the family order for the second/third slot so we don't
        # re-pick the same family on the same day.
        family_order_today = list(family_order)
        for slot in range(per_day):
            # Pick the family with the oldest last-used day index.
            chosen_family = max(family_order_today, key=lambda f: -last_used_day[f])
            last_used_day[chosen_family] = day_idx

            # Pick a template within that family that avoids last template and last region.
            candidates = by_family[chosen_family]
            t = _pick_template_by_region(candidates, last_region_by_family[chosen_family])
            t = _pick_template_for_family(
                [c for c in candidates if c.region == t.region],
                last_template_by_family[chosen_family],
            )

            last_template_by_family[chosen_family] = t.settings_name
            last_region_by_family[chosen_family] = t.region

            schedule.append({
                "day_offset": day_idx,
                "date": current_day.isoformat(),
                "slot": slot + 1,
                "family": t.family,
                "region": t.region,
                "universe": t.universe,
                "delay": t.delay,
                "expression": t.expression,
                "settings": t.settings_name,
                "expected_sharpe": t.expected_sharpe,
                "expected_turnover": t.expected_turnover,
                "notes": t.notes,
            })

            # Rotate the chosen family to the back so we pick a different one next slot.
            family_order_today = [f for f in family_order_today if f != chosen_family]
            if not family_order_today:
                # Fewer families than per_day — stop early for this day.
                break

        day_idx += 1
        current_day += dt.timedelta(days=1)

    return schedule


def write_schedule(schedule: list[dict], output: Path) -> None:
    if not schedule:
        print("Empty schedule.", file=sys.stderr)
        return
    fieldnames = list(schedule[0].keys())
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in schedule:
            w.writerow(row)


def summarize(schedule: list[dict]) -> str:
    by_family: dict[str, int] = {}
    by_region: dict[str, int] = {}
    by_settings: dict[str, int] = {}
    for row in schedule:
        by_family[row["family"]] = by_family.get(row["family"], 0) + 1
        by_region[row["region"]] = by_region.get(row["region"], 0) + 1
        by_settings[row["settings"]] = by_settings.get(row["settings"], 0) + 1
    lines = [
        f"Schedule: {len(schedule)} submission(s) over {len({r['date'] for r in schedule})} day(s).",
        "By family: " + ", ".join(f"{k}={v}" for k, v in sorted(by_family.items())),
        "By region: " + ", ".join(f"{k}={v}" for k, v in sorted(by_region.items())),
        "By template: " + ", ".join(f"{k}={v}" for k, v in sorted(by_settings.items())),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="IQC submission scheduler")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--families", default="PV,Fundamental,Analyst,Sentiment,Composite")
    p.add_argument("--regions", default="USA,ASI,CHN,EMEA,GLB")
    p.add_argument("--per-day", type=int, default=1,
                   help="Submissions per day (default 1; max 4 to stay under daily cap)")
    p.add_argument("--skip-weekends", action="store_true")
    p.add_argument("--output", default="iqc_schedule.csv")
    args = p.parse_args(argv)

    try:
        start = dt.date.fromisoformat(args.start)
        families = [s.strip() for s in args.families.split(",") if s.strip()]
        regions = [s.strip() for s in args.regions.split(",") if s.strip()]
        schedule = build_schedule(
            start=start, days=args.days, families=families, regions=regions,
            per_day=args.per_day, skip_weekends=args.skip_weekends, pool=DEFAULT_POOL,
        )
    except ValueError as exc:
        print(f"Failed to build schedule: {exc}", file=sys.stderr)
        return 2

    out = Path(args.output)
    write_schedule(schedule, out)
    print(summarize(schedule))
    print(f"\nWrote: {out}")
    print(f"First 5 rows:")
    for row in schedule[:5]:
        print(f"  {row['date']}  slot={row['slot']}  {row['family']:<12s}  "
              f"{row['region']:<5s}  {row['settings']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
