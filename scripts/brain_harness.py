#!/usr/bin/env python3
"""
brain_harness.py — WorldQuant BRAIN batch simulator + gate filter + submitter.

A CLI tool that wraps the BRAIN HTTP API to:
  1. Authenticate (POST /api/v1/auth/login)
  2. Submit one or many alpha simulations
  3. Poll for results
  4. Filter results by the standard gates (Sharpe, Fitness, Turnover, Self-Correlation)
  5. Submit qualifying alphas

Subcommands:
  simulate   Run a single alpha and print the result.
  batch      Run a grid of alphas from a CSV file.
  submit     Re-submit qualifying alphas from previous simulation results.
  gates      Print the passing thresholds (for reference).

Auth:
  Reads BRAIN_EMAIL and BRAIN_PASSWORD from the environment (or .env).
  NEVER hardcode credentials.

Mock mode:
  Pass --mock to skip real network calls. Useful for offline development;
  produces deterministic synthetic results so you can exercise the filter logic.

Examples:
  # Single alpha
  python brain_harness.py simulate \\
      --expression "rank(ts_mean(close, 30) - close)" \\
      --region USA --universe TOP3000 --delay 1 \\
      --neutralization Subindustry --decay 4 --truncation 0.08

  # Batch from CSV
  python brain_harness.py batch --grid alpha_grid.csv --output results/

  # Filter & submit qualifying alphas
  python brain_harness.py submit --from-results results/sim_*.json

NOTE on endpoints: the BRAIN API has shifted over time. The endpoint paths
below are best-effort based on community references (alexisdpc/WorldQuant-alpha-trading,
gentlecactus.top guides). If a path returns 404, check your account's API docs
and update BRAIN_ENDPOINTS below.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    print("requests not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)

# Make brainlib importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brainlib.gates import DEFAULT_GATES, GateConfig  # noqa: E402
from brainlib.gates import evaluate as _gates_evaluate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BRAIN_API_BASE = os.environ.get("BRAIN_API_BASE", "https://api.worldquantbrain.com")

# Endpoint paths — verify against your account's API docs.
BRAIN_ENDPOINTS = {
    "login": "/api/v1/auth/login",
    "simulate": "/api/v1/simulate",
    "simulations": "/api/v1/simulations",
    "alpha": "/api/v1/alpha",
    "alpha_submit": "/api/v1/alpha/submit",
}


# ---------------------------------------------------------------------------
# Gates — re-export the canonical evaluator from brainlib.gates so callers
# using `from brain_harness import Gates` still get a compatible shim.
# ---------------------------------------------------------------------------


@dataclass
class Gates:
    """Backwards-compat shim — delegates to brainlib.gates.

    Prefer importing Gates / DEFAULT_GATES directly from brainlib.gates.
    This shim exists so callers using `from brain_harness import Gates` keep
    working, and so the harness exposes a single .evaluate() entry point.
    """

    config: GateConfig = DEFAULT_GATES

    def evaluate(self, metrics: dict) -> tuple[bool, list[str]]:
        return _gates_evaluate(metrics, self.config)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


@dataclass
class BrainClient:
    """Lightweight client for the BRAIN HTTP API.

    Security:
      - email and password are NEVER printed or included in __repr__.
      - password is read from environment variables or .env, NOT from CLI
        args (which would expose it in `ps`, shell history, and CI logs).
      - if repr() or str() is called on this object, credentials show as
        '***' so accidental logging doesn't leak them.
    """

    base_url: str = BRAIN_API_BASE
    email: str = ""
    password: str = ""
    mock: bool = False
    session: requests.Session = field(default_factory=requests.Session)
    cookies: dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"BrainClient(base_url={self.base_url!r}, email={self.email!r}, "
            f"password='***', mock={self.mock})"
        )

    def authenticate(self) -> None:
        if self.mock:
            self.cookies = {"session_id": "mock-" + str(uuid.uuid4())}
            return
        if not self.email or not self.password:
            raise RuntimeError("Set BRAIN_EMAIL and BRAIN_PASSWORD (or use --mock).")
        url = self.base_url + BRAIN_ENDPOINTS["login"]
        resp = self.session.post(
            url, json={"email": self.email, "password": self.password}, timeout=30
        )
        resp.raise_for_status()
        # RequestsCookieJar has a __getitem__ overload that returns
        # str | None; coerce to str for our stricter annotation.
        self.cookies = {str(k): str(v) for k, v in self.session.cookies.items()}

    def _headers(self) -> dict[str, str]:
        return {"Accept": "application/json", "Content-Type": "application/json"}

    def submit_simulation(self, alpha: dict) -> str:
        """POST a simulation; return the simulation id."""
        if self.mock:
            return "mock-sim-" + str(uuid.uuid4())[:8]
        url = self.base_url + BRAIN_ENDPOINTS["simulate"]
        resp = self.session.post(
            url, json=alpha, cookies=self.cookies, headers=self._headers(), timeout=60
        )
        resp.raise_for_status()
        body: dict[str, Any] = resp.json()
        sim_id = body.get("id") or body.get("simulation_id")
        if not sim_id:
            # Don't dump the response body — it could contain echoed
            # credentials or session metadata.
            raise RuntimeError(
                f"BRAIN /simulate returned no id; status={resp.status_code}, "
                f"keys={sorted(body.keys()) if isinstance(body, dict) else type(body).__name__}"
            )
        return str(sim_id)

    def fetch_simulation(self, sim_id: str) -> dict[str, Any]:
        """GET a simulation result."""
        if self.mock:
            return _mock_simulation_result(sim_id, alpha=None)
        url = self.base_url + BRAIN_ENDPOINTS["simulations"] + "/" + sim_id
        resp = self.session.get(url, cookies=self.cookies, headers=self._headers(), timeout=60)
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result

    def submit_alpha(self, sim_id: str) -> dict[str, Any]:
        """Submit a qualifying alpha from a completed simulation."""
        if self.mock:
            return {"status": "mock-submitted", "sim_id": sim_id}
        url = self.base_url + BRAIN_ENDPOINTS["alpha_submit"]
        resp = self.session.post(
            url,
            json={"simulation_id": sim_id},
            cookies=self.cookies,
            headers=self._headers(),
            timeout=60,
        )
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result


# ---------------------------------------------------------------------------
# Alpha definition + mock results
# ---------------------------------------------------------------------------


def alpha_payload(
    expression: str,
    region: str,
    universe: str,
    delay: int,
    neutralization: str,
    decay: int,
    truncation: float,
    pasteurization: bool = False,
    nan_handling: str = "Off",
) -> dict:
    """Build the JSON payload for a simulation."""
    return {
        "type": "REGULAR",
        "expression": expression,
        "region": region,
        "universe": universe,
        "delay": delay,
        "neutralization": neutralization,
        "decay": decay,
        "truncation": truncation,
        "pasteurization": pasteurization,
        "nan_handling": nan_handling,
        "unit_handling": "Verify",
    }


def _mock_simulation_result(sim_id: str, alpha: dict[str, Any] | None) -> dict[str, Any]:
    """Deterministic synthetic result so --mock exercises the gate logic."""
    # Use the sim_id hash to make results repeatable for the same id.
    seed = int(sim_id[-8:], 16) if len(sim_id) >= 8 else hash(sim_id) & 0xFFFFFFFF
    rng = random.Random(seed)
    sharpe = rng.uniform(0.8, 2.2)
    fitness = rng.uniform(0.6, 1.8)
    turnover = rng.uniform(0.05, 0.35)
    returns = rng.uniform(0.03, 0.12)
    drawdown = rng.uniform(0.02, 0.10)
    self_corr = rng.uniform(0.0, 0.85)
    concentration = rng.uniform(0.03, 0.15)
    return {
        "id": sim_id,
        "status": "COMPLETED",
        "metrics": {
            "sharpe": sharpe,
            "fitness": fitness,
            "turnover": turnover,
            "returns": returns,
            "drawdown": drawdown,
            "margin": rng.uniform(0.001, 0.01),
            "concentration": concentration,
            "self_correlation": self_corr,
        },
    }


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_simulate(args, client: BrainClient, gates: Gates) -> int:
    payload = alpha_payload(
        args.expression,
        args.region,
        args.universe,
        args.delay,
        args.neutralization,
        args.decay,
        args.truncation,
        pasteurization=args.pasteurization,
        nan_handling=args.nan_handling,
    )
    client.authenticate()
    sim_id = client.submit_simulation(payload)
    print(f"Submitted simulation: {sim_id}")

    # Poll until complete.
    for _ in range(args.max_polls):
        result = client.fetch_simulation(sim_id)
        if result.get("status") == "COMPLETED":
            break
        time.sleep(args.poll_interval)
    else:
        print(
            f"Simulation {sim_id} did not complete within {args.max_polls} polls.", file=sys.stderr
        )
        return 1

    metrics = result["metrics"]
    passes, reasons = gates.evaluate(metrics)
    print(
        json.dumps(
            {"sim_id": sim_id, "metrics": metrics, "passes": passes, "reasons": reasons}, indent=2
        )
    )
    # --no-exit-on-fail: print JSON but always return success (useful for piping).
    if passes or args.no_exit_on_fail:
        return 0
    return 1


def cmd_batch(args, client: BrainClient, gates: Gates) -> int:
    client.authenticate()
    grid = _load_grid(args.grid)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(_run_one, client, gates, row): row for row in grid}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                result = {"error": str(exc), "row": row}
            sim_id = result.get("sim_id", str(uuid.uuid4()))
            out_path = out_dir / f"sim_{sim_id}.json"
            out_path.write_text(json.dumps(result, indent=2))
            status = "PASS" if result.get("passes") else "FAIL"
            print(f"[{status}] {row.get('expression', '?')[:60]:60s} -> {out_path}")

    return 0


def _run_one(client: BrainClient, gates: Gates, row: dict[str, str]) -> dict[str, Any]:
    payload = alpha_payload(
        expression=row["expression"],
        region=row.get("region", "USA"),
        universe=row.get("universe", "TOP3000"),
        delay=int(row.get("delay", 1)),
        neutralization=row.get("neutralization", "Subindustry"),
        decay=int(row.get("decay", 4)),
        truncation=float(row.get("truncation", 0.08)),
        pasteurization=str(row.get("pasteurization", "Off")).lower() == "true",
        nan_handling=row.get("nan_handling", "Off"),
    )
    sim_id = client.submit_simulation(payload)
    for _ in range(60):
        result = client.fetch_simulation(sim_id)
        if result.get("status") == "COMPLETED":
            break
        time.sleep(2)
    metrics = result.get("metrics", {})
    passes, reasons = gates.evaluate(metrics)
    return {
        "sim_id": sim_id,
        "expression": row["expression"],
        "metrics": metrics,
        "passes": passes,
        "reasons": reasons,
    }


def cmd_submit(args, client: BrainClient, gates: Gates) -> int:
    client.authenticate()
    # Sort to get deterministic order — matters for the daily cap and reproducibility.
    files = sorted(Path(".").glob(args.from_results))
    if not files:
        print(f"No result files matched: {args.from_results}", file=sys.stderr)
        return 1
    submitted = 0
    for f in files:
        try:
            result = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Skipping {f.name}: {exc}", file=sys.stderr)
            continue
        if not result.get("passes"):
            continue
        sim_id = result.get("sim_id")
        if not sim_id:
            print(f"Skipping {f.name}: no sim_id field.", file=sys.stderr)
            continue
        resp = client.submit_alpha(sim_id)
        # Sanitize the response print: just show the status field, not
        # the full body (which could include session metadata).
        status = resp.get("status", "?") if isinstance(resp, dict) else "?"
        print(f"Submitted {sim_id}: status={status}")
        submitted += 1
        if submitted >= args.max_per_day:
            print(f"Hit daily cap of {args.max_per_day}; stopping.")
            break
    print(f"\nSubmitted {submitted} alpha(s).")
    return 0


def cmd_gates(args, client: BrainClient, gates: Gates) -> int:
    # Delegate to the brainlib serializer so the printed thresholds always
    # match the canonical rules.
    from brainlib.gates import as_dict as gates_as_dict

    print(json.dumps(gates_as_dict(gates.config), indent=2))
    return 0


def _load_grid(path: str) -> list[dict[str, str]]:
    """Load a CSV grid of alpha definitions (all values parsed as strings)."""
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WorldQuant BRAIN harness")
    p.add_argument("--mock", action="store_true", help="Offline mode (no network calls)")
    # NOTE: credentials come from environment variables (BRAIN_EMAIL /
    # BRAIN_PASSWORD) or .env, NEVER from CLI args. Exposing them as CLI
    # flags would put them into shell history, process listings, and CI logs.
    p.add_argument(
        "--email",
        default=os.environ.get("BRAIN_EMAIL", ""),
        help="BRAIN account email (or set BRAIN_EMAIL env var; "
        "do not pass on the CLI in shared environments).",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="Run a single alpha")
    s.add_argument("--expression", required=True)
    s.add_argument("--region", default="USA")
    s.add_argument("--universe", default="TOP3000")
    s.add_argument("--delay", type=int, default=1)
    s.add_argument("--neutralization", default="Subindustry")
    s.add_argument("--decay", type=int, default=4)
    s.add_argument("--truncation", type=float, default=0.08)
    s.add_argument("--pasteurization", action="store_true")
    s.add_argument("--nan-handling", default="Off")
    s.add_argument("--max-polls", type=int, default=60)
    s.add_argument("--poll-interval", type=float, default=2.0)
    s.add_argument("--no-exit-on-fail", action="store_true")
    s.set_defaults(func=cmd_simulate)

    b = sub.add_parser("batch", help="Run a grid of alphas from CSV")
    b.add_argument("--grid", required=True)
    b.add_argument("--output", default="results/")
    b.add_argument("--parallel", type=int, default=2)
    b.set_defaults(func=cmd_batch)

    su = sub.add_parser("submit", help="Submit qualifying alphas from result files")
    su.add_argument("--from-results", default="results/sim_*.json")
    su.add_argument("--max-per-day", type=int, default=4)
    su.set_defaults(func=cmd_submit)

    g = sub.add_parser("gates", help="Print the passing thresholds")
    g.set_defaults(func=cmd_gates)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Password is sourced from env (NOT CLI) so it never lands in shell
    # history, process listings, or CI logs.
    password = os.environ.get("BRAIN_PASSWORD", "")
    client = BrainClient(email=args.email, password=password, mock=args.mock)
    gates = Gates()
    func = args.func
    if func is None:
        # argparse with required=True on the subparsers means func is
        # always set when parse_args succeeds, but mypy can't see that.
        return 2
    result = func(args, client, gates)
    return int(result)


if __name__ == "__main__":
    sys.exit(main())
