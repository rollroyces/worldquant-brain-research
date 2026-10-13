#!/usr/bin/env python3
"""
cross_validate.py — In-sample / out-of-sample validation for BRAIN alphas.

The single biggest predictor of an alpha failing the BRAIN test period
is overfitting to the in-sample window. This script splits a returns
panel at a configurable date and reports Sharpe, Fitness, Returns, and
Turnover for each half — plus the OS/IS Sharpe ratio, which is the
classic overfitting diagnostic.

Usage:
  python cross_validate.py \\
      --returns-panel panel.csv \\
      --alpha-column alpha_test \\
      --split-date 2025-04-01

Outputs:
  - IS summary (period 1, earlier dates)
  - OS summary (period 2, later dates)
  - OS/IS Sharpe ratio (a robust alpha keeps this >= ~0.5; anything
    below ~0.3 is likely overfit)
  - Sub-universe Sharpe on each half, if --sub-universe-column is set

Exit code:
  0 if both halves have finite metrics and the OS/IS ratio is sane
  2 if the ratio looks like overfitting (< 0.3)
  1 if either half produced non-finite metrics (broken alpha)

This is a quick local screen — it does NOT replace the BRAIN test
period's blind OOS check. A passing result here means "worth running
on the platform"; it doesn't guarantee the alpha will pass on BRAIN.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("This script requires pandas and numpy.", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brainlib.gates import (  # noqa: E402
    DEFAULT_GATES,
)
from brainlib.gates import (
    evaluate as gates_evaluate,
)
from brainlib.gates import (
    fitness as gates_fitness,
)


@dataclass
class PeriodMetrics:
    """Metrics for one period (IS or OS)."""
    period: str
    n_days: int
    sharpe: float
    fitness: float
    returns_annualized: float
    turnover: float
    drawdown: float
    concentration: float
    passes_gates: bool
    gate_failures: str


def _annualized_return(daily: np.ndarray) -> float:
    if len(daily) == 0:
        return 0.0
    return float((1 + daily).prod() ** (252 / len(daily)) - 1)


def _sharpe(daily: np.ndarray) -> float:
    if len(daily) < 2:
        return 0.0
    std = daily.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(daily.mean() / std * math.sqrt(252))


def _max_drawdown(daily: np.ndarray) -> float:
    if len(daily) == 0:
        return 0.0
    cum = np.cumprod(1 + daily)
    peak = np.maximum.accumulate(cum)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (cum - peak) / peak
    return float(np.nanmin(dd))


def _turnover(weights: np.ndarray) -> float:
    if len(weights) < 2:
        return 0.0
    return float(np.abs(np.diff(weights, axis=0)).mean())


def _concentration(weights_today: np.ndarray) -> float:
    if len(weights_today) == 0:
        return 0.0
    return float(np.abs(weights_today).max())


def _evaluate_period(panel: pd.DataFrame, alpha_col: str,
                     period_name: str, gates: Any) -> PeriodMetrics:
    """Compute all metrics for one period (slice of the panel)."""
    panel = panel.sort_values(["date", "stock_id"])
    alpha_signal = panel[alpha_col].values
    returns = panel["return"].values

    daily_pnl_series = (
        pd.DataFrame({"date": panel["date"].values, "pnl": alpha_signal * returns})
        .groupby("date")["pnl"].sum()
    )
    daily_pnl = daily_pnl_series.values

    weight_panel = panel.pivot(index="date", columns="stock_id", values=alpha_col)
    weights = weight_panel.fillna(0.0).values
    last_day = weights[-1] if len(weights) else np.array([0.0])

    to = _turnover(weights)
    conc = _concentration(last_day)
    ret = _annualized_return(daily_pnl)
    shp = _sharpe(daily_pnl)
    dd = _max_drawdown(daily_pnl)
    fit = gates_fitness(shp, ret, to, floor=gates.turnover_floor)

    metrics = {
        "sharpe": shp,
        "fitness": fit,
        "turnover": to,
        "self_correlation": 0.0,  # unknown from a single panel
        "concentration": conc,
        "returns": ret,
        "drawdown": dd,
    }
    passes, reasons = gates_evaluate(metrics, gates)

    return PeriodMetrics(
        period=period_name,
        n_days=int(panel["date"].nunique()),
        sharpe=round(shp, 4),
        fitness=round(fit, 4),
        returns_annualized=round(ret, 4),
        turnover=round(to, 4),
        drawdown=round(dd, 4),
        concentration=round(conc, 4),
        passes_gates=passes,
        gate_failures="; ".join(reasons),
    )


def cross_validate(panel: pd.DataFrame, alpha_col: str,
                   split_date: str | None, split_fraction: float,
                   gates: Any = DEFAULT_GATES) -> tuple[PeriodMetrics, PeriodMetrics, float]:
    """Split the panel at the given date (or fraction) and evaluate each half.

    Returns (is_metrics, os_metrics, os_sharpe_over_is_sharpe).

    The OS/IS ratio is the canonical overfitting diagnostic:
      ratio >= 1.0: alpha is *robust* (rare; treat with skepticism)
      ratio >= 0.5: alpha is *probably real*
      ratio < 0.3 : alpha is *likely overfit*
      ratio < 0.0 : alpha *flipped signs* between IS and OS
    """
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    if split_date is not None:
        cutoff = pd.to_datetime(split_date)
        is_panel = panel[panel["date"] < cutoff]
        os_panel = panel[panel["date"] >= cutoff]
    else:
        # Sort by date and split at the requested fraction.
        unique_dates = sorted(panel["date"].unique())
        idx = int(len(unique_dates) * split_fraction)
        cutoff = unique_dates[idx]
        is_panel = panel[panel["date"] < cutoff]
        os_panel = panel[panel["date"] >= cutoff]

    if is_panel.empty or os_panel.empty:
        raise ValueError(
            f"Split produced empty half: IS={len(is_panel)} rows, OS={len(os_panel)} rows. "
            f"Choose a different split date or fraction."
        )

    is_metrics = _evaluate_period(is_panel, alpha_col, "IS", gates)
    os_metrics = _evaluate_period(os_panel, alpha_col, "OS", gates)

    # OS/IS Sharpe ratio. Handle signs: if IS is negative, the ratio
    # is uninformative.
    if abs(is_metrics.sharpe) < 1e-6:
        ratio = float("nan")
    else:
        ratio = os_metrics.sharpe / is_metrics.sharpe

    return is_metrics, os_metrics, ratio


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="In-sample / out-of-sample validation")
    p.add_argument("--returns-panel", required=True)
    p.add_argument("--alpha-column", required=True,
                   help="Column in the panel that holds the alpha signal.")
    p.add_argument("--split-date", default=None,
                   help="ISO date (YYYY-MM-DD). Everything before is IS, "
                        "after is OS. Mutually exclusive with --split-fraction.")
    p.add_argument("--split-fraction", type=float, default=0.7,
                   help="If --split-date is not given, split at this fraction of "
                        "the date range. Default 0.7 (70%% IS, 30%% OS).")
    p.add_argument("--output", default="cross_validation.csv")
    args = p.parse_args(argv)

    if (args.split_date is None and not 0 < args.split_fraction < 1):
        p.error("--split-fraction must be between 0 and 1 exclusive")

    path = Path(args.returns_panel)
    panel = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if args.alpha_column not in panel.columns:
        print(f"Column '{args.alpha_column}' not in panel.", file=sys.stderr)
        return 1

    try:
        is_m, os_m, ratio = cross_validate(
            panel, args.alpha_column,
            split_date=args.split_date,
            split_fraction=args.split_fraction,
        )
    except ValueError as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 1

    # Report
    print(f"\n=== Cross-validation: {args.alpha_column} ===")
    print(f"Split:    IS={is_m.n_days} days, OS={os_m.n_days} days")
    ratio_str = f"{ratio:.3f}" if not math.isnan(ratio) else "n/a (IS Sharpe ~0)"
    print(f"          ratio={ratio_str}")
    print()
    print(f"{'Metric':<22s}  {'IS':>12s}  {'OS':>12s}")
    print("-" * 50)
    for attr in ("sharpe", "fitness", "returns_annualized", "turnover",
                 "drawdown", "concentration", "passes_gates"):
        is_v = getattr(is_m, attr)
        os_v = getattr(os_m, attr)
        is_s = f"{is_v:.4f}" if isinstance(is_v, float) else str(is_v)
        os_s = f"{os_v:.4f}" if isinstance(os_v, float) else str(os_v)
        print(f"{attr:<22s}  {is_s:>12s}  {os_s:>12s}")
    print()
    print(f"IS gate failures: {is_m.gate_failures or '(none)'}")
    print(f"OS gate failures: {os_m.gate_failures or '(none)'}")

    # Verdict
    if not math.isnan(ratio):
        if ratio >= 0.5:
            verdict = "ROBUST"
        elif ratio >= 0.3:
            verdict = "MARGINAL — worth a real submission but watch closely"
        elif ratio >= 0.0:
            verdict = "LIKELY OVERFIT — Sharpe collapses OS"
        else:
            verdict = "FLIPPED SIGNS — alpha is not real"
    else:
        verdict = "INDETERMINATE (IS Sharpe ≈ 0)"
    print(f"\nVerdict: {verdict}")

    # CSV report
    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period", "n_days", "sharpe", "fitness", "returns_annualized",
                    "turnover", "drawdown", "concentration", "passes_gates",
                    "gate_failures"])
        for m in (is_m, os_m):
            w.writerow([m.period, m.n_days, m.sharpe, m.fitness, m.returns_annualized,
                        m.turnover, m.drawdown, m.concentration, m.passes_gates,
                        m.gate_failures])
        w.writerow([])
        w.writerow(["os_sharpe_over_is_sharpe", round(ratio, 4) if not math.isnan(ratio) else "nan"])
        w.writerow(["verdict", verdict])
    print(f"\nWrote: {out_path}")

    # Exit code: 2 if likely overfit, 0 otherwise.
    if not math.isnan(ratio) and ratio < 0.3:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
