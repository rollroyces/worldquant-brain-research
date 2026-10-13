#!/usr/bin/env python3
"""
parameter_sweep.py — Robustness sweep across alpha simulation settings.

Given a base expression and parameter ranges, sweep each parameter and
report Sharpe + Fitness for every combination. The point is to detect
"fragile" alphas that only work at the exact settings they were
designed at — a real alpha should hold up across small parameter
perturbations.

Usage:
  python parameter_sweep.py \\
      --expression "rank(ts_mean(close, 30) - close)" \\
      --region USA --universe TOP3000 --delay 1 \\
      --sweep-decay 2,4,6,8,10 \\
      --sweep-truncation 0.05,0.08,0.10,0.15 \\
      --sweep-neutralization Market,Sector,Industry,Subindustry \\
      --mock

Outputs:
  - A CSV with one row per (decay, truncation, neutralization) combo,
    columns: decay, truncation, neutralization, sharpe, fitness, passes
  - A summary that flags "fragile" cells (Sharpe < 1.25 on >25% of
    combinations).

Note: in mock mode the Sharpe/Fitness values are deterministic noise,
not real backtests. The point of this script is to exercise the
sweep machinery; the user runs against real BRAIN data to get real
results.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Lazy-import brain_harness to avoid pulling in requests/pandas unnecessarily.
# We import only what we need for the mock-mode path.


@dataclass
class SweepResult:
    decay: int
    truncation: float
    neutralization: str
    sharpe: float
    fitness: float
    turnover: float
    concentration: float
    passes: bool
    reasons: str


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def sweep(expression: str, region: str, universe: str, delay: int,
          decays: list[int], truncations: list[float],
          neutralizations: list[str], mock: bool,
          pasteurization: bool = False, nan_handling: str = "Off",
          gates: Any = None) -> list[SweepResult]:
    """Run the (decay × truncation × neutralization) sweep, return results.

    Uses scripts/brain_harness in mock mode (no real BRAIN calls) so
    the script stays runnable without credentials. The numbers are
    deterministic noise; the script is the parameter-sweep plumbing.
    """
    import brain_harness
    if gates is None:
        from brainlib.gates import DEFAULT_GATES
        gates = DEFAULT_GATES

    client = brain_harness.BrainClient(mock=mock)
    client.authenticate()

    results = []
    for decay, trunc, neut in itertools.product(decays, truncations, neutralizations):
        payload = brain_harness.alpha_payload(
            expression=expression, region=region, universe=universe,
            delay=delay, neutralization=neut, decay=decay, truncation=trunc,
            pasteurization=pasteurization, nan_handling=nan_handling,
        )
        sim_id = client.submit_simulation(payload)
        # Mock mode returns immediately; real mode would poll. In mock
        # we can call fetch_simulation and get a result back.
        result = client.fetch_simulation(sim_id)
        metrics = result["metrics"]
        passes, reasons = brain_harness._gates_evaluate(metrics, gates)
        results.append(SweepResult(
            decay=decay,
            truncation=trunc,
            neutralization=neut,
            sharpe=round(metrics["sharpe"], 4),
            fitness=round(metrics["fitness"], 4),
            turnover=round(metrics["turnover"], 4),
            concentration=round(metrics["concentration"], 4),
            passes=passes,
            reasons="; ".join(reasons),
        ))
    return results


def fragility_summary(results: list[SweepResult]) -> dict[str, Any]:
    """Summarize how robust the alpha is across the parameter grid."""
    n = len(results)
    if n == 0:
        return {"n_combos": 0, "pass_rate": 0.0, "median_sharpe": 0.0,
                "min_sharpe": 0.0, "max_sharpe": 0.0, "fragile": True,
                "sharpe_std": 0.0}

    sharpes = [r.sharpe for r in results]
    passes = sum(1 for r in results if r.passes)
    mean = sum(sharpes) / n
    variance = sum((s - mean) ** 2 for s in sharpes) / n
    std = math.sqrt(variance)

    # Median requires sorted list.
    sorted_sharpes = sorted(sharpes)
    if n % 2 == 1:
        median = sorted_sharpes[n // 2]
    else:
        median = (sorted_sharpes[n // 2 - 1] + sorted_sharpes[n // 2]) / 2

    return {
        "n_combos": n,
        "pass_rate": passes / n,
        "median_sharpe": round(median, 4),
        "min_sharpe": round(min(sharpes), 4),
        "max_sharpe": round(max(sharpes), 4),
        "sharpe_std": round(std, 4),
        "fragile": (passes / n) < 0.75,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Parameter robustness sweep")
    p.add_argument("--expression", required=True)
    p.add_argument("--region", default="USA")
    p.add_argument("--universe", default="TOP3000")
    p.add_argument("--delay", type=int, default=1)
    p.add_argument("--sweep-decay", default="2,4,6,8")
    p.add_argument("--sweep-truncation", default="0.05,0.08,0.10")
    p.add_argument("--sweep-neutralization", default="Market,Subindustry")
    p.add_argument("--mock", action="store_true",
                   help="Run in mock mode (deterministic noise metrics).")
    p.add_argument("--output", default="sweep_results.csv")
    args = p.parse_args(argv)

    decays = _parse_csv_ints(args.sweep_decay)
    truncs = _parse_csv_floats(args.sweep_truncation)
    neuts = _parse_csv_strs(args.sweep_neutralization)
    n_combos = len(decays) * len(truncs) * len(neuts)

    print(f"Sweeping {n_combos} combinations across "
          f"decay × truncation × neutralization...")
    print(f"  decay:           {decays}")
    print(f"  truncation:      {truncs}")
    print(f"  neutralization:  {neuts}")
    print()

    results = sweep(
        expression=args.expression,
        region=args.region,
        universe=args.universe,
        delay=args.delay,
        decays=decays,
        truncations=truncs,
        neutralizations=neuts,
        mock=args.mock,
    )

    # CSV report
    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["decay", "truncation", "neutralization", "sharpe", "fitness",
                    "turnover", "concentration", "passes", "reasons"])
        for r in results:
            w.writerow([r.decay, r.truncation, r.neutralization, r.sharpe, r.fitness,
                        r.turnover, r.concentration, r.passes, r.reasons])
    print(f"Wrote: {out_path}")

    # Summary
    summary = fragility_summary(results)
    print(f"\n=== Sweep summary ({summary['n_combos']} combos) ===")
    print(f"  pass rate:    {summary['pass_rate']:.1%}")
    print(f"  Sharpe range: {summary['min_sharpe']:.3f} — {summary['max_sharpe']:.3f}")
    print(f"  Sharpe median:{summary['median_sharpe']:.3f}")
    print(f"  Sharpe std:   {summary['sharpe_std']:.3f}")
    if summary["fragile"]:
        print("  → FRAGILE: pass rate below 75%. "
              "Consider sweeping the alpha's expression or parameters more carefully.")
    else:
        print("  → ROBUST: pass rate at or above 75%.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
