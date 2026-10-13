#!/usr/bin/env python3
"""scripts/_perf_benchmark.py — internal validator performance baseline.

NOT a user-facing script. The leading underscore keeps it out of README
"sections" and out of `python scripts/` user calls. Run it directly:

    python3 scripts/_perf_benchmark.py

What it does:
  1. Builds a synthetic returns panel at /tmp/perf_panel.csv with the
     shape used by the task brief:
         500 stocks x 1000 business days x 5 sub-industries,
         4 alpha columns (alpha_a..alpha_d).
  2. Times `scripts.subuniverse_validator.evaluate_alpha` for each alpha
     individually (the same code path the CLI uses), then prints per-alpha
     and total elapsed time plus a coarse breakdown of which sub-call
     dominates.

The validator's source is locked, so this script imports the function
rather than editing the CLI for a --timing flag.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Repo layout: scripts/ sits next to brainlib/. The validator expects this.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402  (imports follow sys.path setup)
import pandas as pd  # noqa: E402
from subuniverse_validator import (  # noqa: E402  (imports follow sys.path setup)
    annualized_return,
    concentration,
    evaluate_alpha,
    max_drawdown,
    per_subuniverse_sharpe,
    sharpe,
    turnover,
)

from brainlib.gates import DEFAULT_GATES  # noqa: E402

PANEL_PATH = Path("/tmp/perf_panel.csv")
STOCKS = 500
DAYS = 1000
SUBINDUSTRIES = 5
ALPHAS = ["alpha_a", "alpha_b", "alpha_c", "alpha_d"]


def build_panel(stocks: int, days: int, subindustries: int,
                alphas: list[str], seed: int = 42) -> pd.DataFrame:
    """Generate a (date, stock_id, subindustry, return, alpha_*) panel.

    Synthetic and flat: roughly stationary alpha signals plus a small
    daily return stream, so the validator has real work to do per alpha
    (pivots, groupby, sub-universe loop) without us claiming a signal
    that 'looks like truth'. Used for perf baseline only.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=days)
    stock_ids = [f"S{i:04d}" for i in range(stocks)]
    sub_assignments = rng.integers(0, subindustries, size=stocks)
    stock_to_sub = {stock_ids[i]: f"sub_{sub_assignments[i]}" for i in range(stocks)}

    n = stocks * days
    df = pd.DataFrame({
        "date": np.repeat(dates.values, stocks),
        "stock_id": np.tile(stock_ids, days),
    })
    df["subindustry"] = df["stock_id"].map(stock_to_sub)
    df["return"] = rng.normal(0.0, 0.01, size=n).astype(np.float32)
    for name in alphas:
        # Each alpha is a different random weight so the per-sub-universe
        # loop has real work to do. Values clipped to a small range so we
        # don't get absurd Sharpe numbers that distort qualitative reading.
        w = rng.normal(0.0, 1.0, size=n).astype(np.float32)
        df[name] = np.clip(w, -3.0, 3.0)
    return df


def profile_one_alpha(panel: pd.DataFrame, alpha_col: str,
                      sub_col: str) -> dict[str, float]:
    """Break down evaluate_alpha into its sub-calls and time each one.

    Keeps the validator's internal logic intact — just calls the same
    helpers it would call, in the same order, around wall-clock timing.
    """
    timings: dict[str, float] = {}
    t_overall = time.perf_counter()

    panel_sorted = panel.sort_values(["date", "stock_id"])
    alpha_signal = panel_sorted[alpha_col].values
    returns = panel_sorted["return"].values

    t = time.perf_counter()
    daily_pnl_df = pd.DataFrame({
        "date": panel_sorted["date"].values,
        "pnl": alpha_signal * returns,
    })
    daily_pnl = daily_pnl_df.groupby("date")["pnl"].sum().values
    timings["daily_pnl"] = time.perf_counter() - t

    t = time.perf_counter()
    weight_panel = panel_sorted.pivot(index="date", columns="stock_id",
                                      values=alpha_col)
    weights = weight_panel.fillna(0.0).values
    timings["pivot_weights"] = time.perf_counter() - t

    t = time.perf_counter()
    concentration(weights[-1] if len(weights) else np.array([0.0]))
    turnover(weights)
    timings["conc_turnover"] = time.perf_counter() - t

    t = time.perf_counter()
    annualized_return(daily_pnl)
    sharpe(daily_pnl)
    max_drawdown(daily_pnl)
    timings["metrics"] = time.perf_counter() - t

    t = time.perf_counter()
    per_subuniverse_sharpe(panel_sorted, alpha_col, sub_col)
    timings["per_subuniverse_sharpe"] = time.perf_counter() - t

    timings["total_inside_profile"] = time.perf_counter() - t_overall
    return timings


def main() -> int:
    if not PANEL_PATH.exists():
        print(f"[perf] Building synthetic panel ({STOCKS} x {DAYS} x {SUBINDUSTRIES}, "
              f"alphas={ALPHAS})...")
        panel = build_panel(STOCKS, DAYS, SUBINDUSTRIES, ALPHAS)
        panel.to_csv(PANEL_PATH, index=False)
        print(f"[perf] Wrote {PANEL_PATH} ({panel.shape[0]:,} rows)")
    else:
        print(f"[perf] Reusing existing {PANEL_PATH}")
        panel = pd.read_csv(PANEL_PATH)

    print(f"[perf] Panel shape: {panel.shape}, "
          f"alphas: {[c for c in panel.columns if c.startswith('alpha_')]}")

    # Warmup: load the brainlib caches (e.g., GroupBy dispatch). The first
    # call into pandas for a fresh DataFrame layout is meaningfully slower
    # than subsequent ones, which we don't want to mistake for a slow alpha.
    sub_col = "subindustry"
    self_correlation = 0.0
    evaluate_alpha(panel, ALPHAS[0], sub_col, DEFAULT_GATES, self_correlation)

    overall_t0 = time.perf_counter()
    per_alpha: dict[str, float] = {}
    profile_totals: dict[str, dict[str, float]] = {}
    for a in ALPHAS:
        t0 = time.perf_counter()
        evaluate_alpha(panel, a, sub_col, DEFAULT_GATES, self_correlation)
        per_alpha[a] = time.perf_counter() - t0
        prof = profile_one_alpha(panel, a, sub_col)
        profile_totals[a] = prof
        sub_parts = (f"pivot={prof['pivot_weights']*1000:.0f}ms  "
                     f"sub_universe={prof['per_subuniverse_sharpe']*1000:.0f}ms  "
                     f"metrics={prof['metrics']*1000:.0f}ms  "
                     f"pnl_groupby={prof['daily_pnl']*1000:.0f}ms")
        print(f"[perf] {a}: {per_alpha[a]:.3f}s    |  {sub_parts}")
    overall = time.perf_counter() - overall_t0

    avg = sum(per_alpha.values()) / len(per_alpha)
    print()
    print(f"[perf] Per-alpha avg:  {avg:.3f}s")
    print(f"[perf] Total (4 alphas): {overall:.3f}s")

    # Heuristic: flag a "next step" note if any single alpha took > 5s.
    slow = [a for a, t in per_alpha.items() if t > 5.0]
    if slow:
        print(f"[perf] SLOW: alphas over 5s: {slow}")
        print("[perf] See docs/PERFORMANCE.md for bottleneck candidates.")
    else:
        print("[perf] All alphas < 5s on this panel size; no current bottleneck.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
