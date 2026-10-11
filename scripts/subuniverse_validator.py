#!/usr/bin/env python3
"""
subuniverse_validator.py — Local pre-screening for WorldQuant BRAIN alphas.

Computes the metrics that BRAIN reports and applies the gate thresholds,
LOCALLY, before burning simulation slots.

The sub-universe Sharpe cutoff is the gate most often missed without local
pre-screening — running all simulations on BRAIN is wasteful when ~80% of
your candidates would fail on a per-sub-universe basis.

What it does:
  1. Loads a returns panel: a DataFrame with columns
       [date, stock_id, subindustry (or sector/industry), return, alpha_signal]
     where alpha_signal is the daily alpha weight per stock (already
     neutralized / truncated / decayed as you intend to submit).
  2. For each alpha (one column of alpha_signal), computes:
       - Sharpe
       - Returns (annualized)
       - Turnover
       - Fitness
       - Drawdown
       - Per-sub-universe Sharpe (for each group, e.g. subindustry)
       - Pass/fail against all gates (including self-correlation) plus the
         scaled sub-universe Sharpe cutoff
  3. Outputs a CSV with one row per alpha + a column for each metric + a
     final PASS/FAIL column + reasons for failure.

Inputs:
  --alphas         Optional CSV with one row per alpha. The script actually
                   reads alpha columns from the returns panel (columns
                   starting with --alpha-prefix), so this argument is kept
                   for documentation only.
  --returns-panel  Parquet or CSV with the columns above. Use the actual
                   pnl from your local backtest engine.
  --sub-universe-column   Which column to slice on (default: subindustry).
  --output         Where to write the report CSV.
  --alpha-prefix   Column prefix in the returns panel that holds alpha signals
                   (default: alpha_).
  --expected-self-correlation   If you have an estimate of max self-corr
                   against your prior submitted alphas, pass it here; the
                   validator will then enforce the gate (default: assume 0).

Sub-universe Sharpe formula (from BRAIN docs):
  sub_sharpe >= 0.75 * sqrt(sub_size / full_size) * alpha_sharpe

This means an alpha with full-universe Sharpe 1.4 must achieve, in each
sub-universe, a Sharpe of at least 0.75 * sqrt(0.05) * 1.4 ≈ 0.23 for a
sub-universe of 5% of stocks, scaling up for larger sub-universes.

Caveats:
  This script assumes your local backtest already applies neutralization,
  truncation, decay, etc. as specified in the simulation settings. It does
  NOT simulate the BRAIN pipeline itself; it tells you whether the *output*
  of that pipeline would pass the gates.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("This script requires pandas and numpy. Run: pip install pandas numpy", file=sys.stderr)
    sys.exit(1)

# Use the canonical gate logic from brainlib.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brainlib.gates import (  # noqa: E402
    DEFAULT_GATES,
    GateConfig,
    evaluate as gates_evaluate,
    fitness as gates_fitness,
    sub_universe_cutoff as gates_sub_universe_cutoff,
)


# ---------------------------------------------------------------------------
# Metrics (per-alpha)
# ---------------------------------------------------------------------------

def annualized_return(daily_returns: np.ndarray) -> float:
    """Compound daily returns into an annualized figure."""
    if len(daily_returns) == 0:
        return 0.0
    return float((1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1)


def sharpe(daily_returns: np.ndarray) -> float:
    """Annualized Sharpe ratio (assuming rf=0)."""
    if len(daily_returns) < 2:
        return 0.0
    std = daily_returns.std(ddof=1)
    # Use a tolerance: floating-point arithmetic can leave a near-zero std
    # that would otherwise explode the ratio.
    if std < 1e-12:
        return 0.0
    return float(daily_returns.mean() / std * math.sqrt(252))


def max_drawdown(daily_returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of cumulative returns."""
    if len(daily_returns) == 0:
        return 0.0
    cum = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(cum)
    # Avoid divide-by-zero on the very first bar.
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (cum - peak) / peak
    return float(np.nanmin(dd))


def turnover(weights: np.ndarray) -> float:
    """Average daily L1 turnover of weight changes (per-stock)."""
    if len(weights) < 2:
        return 0.0
    return float(np.abs(np.diff(weights, axis=0)).mean())


def concentration(weights_today: np.ndarray) -> float:
    """Max absolute weight on any single stock (proxy for concentration)."""
    if len(weights_today) == 0:
        return 0.0
    return float(np.abs(weights_today).max())


def per_subuniverse_sharpe(panel: pd.DataFrame, alpha_col: str,
                           sub_col: str) -> dict[Any, float]:
    """Sharpe ratio of the alpha within each value of sub_col.

    The keys are the actual values of `sub_col` (strings for typical
    subindustry/sector columns, but Any covers other dtypes that pandas
    groupby can produce).

    Pivots to a (date, stock_id) matrix first to avoid shape-mismatch bugs
    when the panel has missing rows.
    """
    out: dict[Any, float] = {}
    for sub, sub_panel in panel.groupby(sub_col):
        sub_panel = sub_panel.sort_values("date")
        # Pivot to (date, stock_id) matrix; missing values become NaN → 0 weight.
        wide = sub_panel.pivot(index="date", columns="stock_id", values=alpha_col)
        wide_returns = sub_panel.pivot(index="date", columns="stock_id", values="return")
        # Align (in case a stock appears in one pivot but not the other).
        wide, wide_returns = wide.align(wide_returns, join="inner", axis=1)
        w = wide.fillna(0.0).values
        r = wide_returns.fillna(0.0).values
        # Daily pnl: per-day sum of weight × return.
        daily_pnl = (w * r).sum(axis=1)
        out[sub] = sharpe(daily_pnl)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_alpha(panel: pd.DataFrame, alpha_col: str, sub_col: str,
                   gates: GateConfig, self_correlation: float) -> dict:
    """Evaluate one alpha; return metrics + pass/fail + reasons."""
    panel = panel.sort_values(["date", "stock_id"])
    alpha_signal = panel[alpha_col].values
    returns = panel["return"].values

    # Daily pnl (sum of weight × return per day, collapsing stocks).
    df = pd.DataFrame({"date": panel["date"].values, "pnl": alpha_signal * returns})
    daily_pnl = df.groupby("date")["pnl"].sum().values

    # Weight per (date, stock) — pivot to handle missing (date, stock) pairs.
    weight_panel = panel.pivot(index="date", columns="stock_id", values=alpha_col)
    weights = weight_panel.fillna(0.0).values

    # Concentration: max abs weight on the latest day.
    last_day_weights = weights[-1] if len(weights) else np.array([0.0])
    conc = concentration(last_day_weights)

    # Turnover: average daily L1 change.
    to = turnover(weights)

    # Returns / Sharpe / Drawdown on daily pnl.
    ret = annualized_return(daily_pnl)
    shp = sharpe(daily_pnl)
    dd = max_drawdown(daily_pnl)
    fit = gates_fitness(shp, ret, to, floor=gates.turnover_floor)

    # Sub-universe Sharpe.
    sub_sharpes = per_subuniverse_sharpe(panel, alpha_col, sub_col)
    sub_min = min(sub_sharpes.values()) if sub_sharpes else 0.0
    full_size = panel["stock_id"].nunique()
    worst_sub = None
    sub_cutoff_violations = []
    for sub, ssh in sub_sharpes.items():
        sub_size = panel[panel[sub_col] == sub]["stock_id"].nunique()
        cutoff = gates_sub_universe_cutoff(shp, sub_size, full_size,
                                           scale=gates.sub_universe_scale)
        if ssh < cutoff:
            sub_cutoff_violations.append(f"{sub} ({ssh:.3f} < {cutoff:.3f})")
            worst_sub = sub

    # Apply the canonical gate rules via brainlib, plus our sub-universe check.
    metrics = {
        "sharpe": shp,
        "fitness": fit,
        "turnover": to,
        "self_correlation": self_correlation,
        "concentration": conc,
        "returns": ret,
        "drawdown": dd,
    }
    passes, reasons = gates_evaluate(metrics, gates)
    if sub_cutoff_violations:
        passes = False
        # Prepend the sub-universe failures so they're visible alongside
        # the other gate reasons.
        reasons = [f"Sub-universe Sharpe failures: {', '.join(sub_cutoff_violations[:3])}"] + reasons

    return {
        "alpha": alpha_col,
        "sharpe": round(shp, 4),
        "fitness": round(fit, 4),
        "returns_annualized": round(ret, 4),
        "turnover": round(to, 4),
        "drawdown": round(dd, 4),
        "concentration": round(conc, 4),
        "self_correlation": round(self_correlation, 4),
        "min_sub_sharpe": round(sub_min, 4),
        "worst_sub_universe": worst_sub,
        "passes": passes,
        "reasons": "; ".join(reasons),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Local BRAIN alpha validator")
    p.add_argument("--alphas",
                   help="Optional: CSV with one row per alpha (col 'name'). "
                        "The alpha signals themselves are read from the returns panel.")
    p.add_argument("--returns-panel", required=True, help="Parquet or CSV of returns panel")
    p.add_argument("--sub-universe-column", default="subindustry")
    p.add_argument("--output", default="local_validation.csv")
    p.add_argument("--alpha-prefix", default="alpha_",
                   help="Column prefix in the returns panel that holds alpha signals")
    p.add_argument("--expected-self-correlation", type=float, default=0.0,
                   help="Max self-correlation against your prior submitted alphas "
                        "(the validator cannot compute this from the panel alone; "
                        "set this to the value the BRAIN self-corr endpoint returned "
                        "for the most-correlated prior submission).")
    args = p.parse_args(argv)

    # Load returns panel.
    path = Path(args.returns_panel)
    panel = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {"date", "stock_id", args.sub_universe_column, "return"}
    missing = required - set(panel.columns)
    if missing:
        print(f"Returns panel missing columns: {missing}", file=sys.stderr)
        return 1

    # Find alpha columns in the panel.
    alpha_cols = [c for c in panel.columns if c.startswith(args.alpha_prefix)]
    if not alpha_cols:
        print(f"No alpha columns found with prefix '{args.alpha_prefix}'.", file=sys.stderr)
        return 1

    gates = DEFAULT_GATES
    results = []
    for col in alpha_cols:
        result = evaluate_alpha(panel, col, args.sub_universe_column, gates,
                                self_correlation=args.expected_self_correlation)
        results.append(result)

    # Write report.
    out_path = Path(args.output)
    fieldnames = ["alpha", "sharpe", "fitness", "returns_annualized", "turnover",
                  "drawdown", "concentration", "self_correlation", "min_sub_sharpe",
                  "worst_sub_universe", "passes", "reasons"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    n_pass = sum(1 for r in results if r["passes"])
    print(f"\nValidated {len(results)} alpha(s); {n_pass} pass all gates.")
    print(f"Report: {out_path}")
    return 0 if n_pass > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
