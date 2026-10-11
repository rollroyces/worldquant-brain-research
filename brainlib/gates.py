"""
brainlib.gates — canonical WorldQuant BRAIN submission gates.

Extracted into its own module so the rules are testable in isolation and
changeable from one place. The harness and validator scripts import from
here.

The thresholds below are the US / Delay-1 baseline. They vary slightly by
region (CHN is stricter) and by delay (Delay-0 is stricter than Delay-1).
Adjust per call site as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


# Baseline thresholds for US, Delay-1.
@dataclass(frozen=True)
class GateConfig:
    min_sharpe: float = 1.25
    min_fitness: float = 1.0
    turnover_min: float = 0.01
    turnover_max: float = 0.70
    max_self_corr: float = 0.70
    max_concentration: float = 0.10
    sub_universe_scale: float = 0.75
    ideal_max_drawdown: float = 0.10
    turnover_floor: float = 0.125         # the floor in the Fitness formula
    self_corr_sharpe_override: float = 1.375   # 1.25 * 1.10 = 10% over Sharpe threshold


# Convenience aliases.
DEFAULT_GATES = GateConfig()
US_DELAY1 = DEFAULT_GATES


def fitness(sharpe: float, returns: float, turnover: float,
            floor: float = DEFAULT_GATES.turnover_floor) -> float:
    """The BRAIN Fitness formula:
        Fitness = Sharpe * sqrt(|returns| / max(turnover, floor))

    Args:
        sharpe:    annualised Sharpe ratio.
        returns:   annualised return (as a fraction, e.g. 0.08 for 8%).
        turnover:  daily turnover fraction.
        floor:     minimum turnover the formula treats as the denominator.
    """
    if turnover <= 0:
        return 0.0
    return sharpe * (abs(returns) / max(turnover, floor)) ** 0.5


def evaluate(metrics: dict[str, float], gates: GateConfig = DEFAULT_GATES) -> tuple[bool, list[str]]:
    """Apply the gate thresholds to a metrics dict.

    Args:
        metrics: dict mapping metric name (e.g. "sharpe", "fitness",
                 "turnover", "self_correlation", "concentration",
                 "returns", "drawdown") to a numeric value. Missing keys
                 default to 0.
        gates:   GateConfig with thresholds.

    Returns:
        (passes, reasons) — passes is True iff no failures; reasons lists
        human-readable strings for each failure.
    """
    sharpe = metrics.get("sharpe", 0.0)
    fit = metrics.get("fitness", 0.0)
    turnover = metrics.get("turnover", 0.0)
    self_corr = metrics.get("self_correlation", 0.0)
    concentration = metrics.get("concentration", 0.0)

    reasons: list[str] = []

    if sharpe < gates.min_sharpe:
        reasons.append(f"Sharpe {sharpe:.3f} < {gates.min_sharpe}")
    if fit < gates.min_fitness:
        reasons.append(f"Fitness {fit:.3f} < {gates.min_fitness}")
    if not (gates.turnover_min <= turnover <= gates.turnover_max):
        reasons.append(
            f"Turnover {turnover:.3f} outside [{gates.turnover_min}, {gates.turnover_max}]"
        )
    if self_corr > gates.max_self_corr and sharpe < gates.self_corr_sharpe_override:
        reasons.append(
            f"Self-Corr {self_corr:.3f} > {gates.max_self_corr} and Sharpe "
            f"{sharpe:.3f} < override {gates.self_corr_sharpe_override:.3f}"
        )
    if concentration > gates.max_concentration:
        reasons.append(
            f"Concentration {concentration:.3f} > {gates.max_concentration}"
        )

    return (len(reasons) == 0), reasons


def sub_universe_cutoff(alpha_sharpe: float, sub_size: int, full_size: int,
                        scale: float = DEFAULT_GATES.sub_universe_scale) -> float:
    """Minimum Sharpe the alpha must achieve within a sub-universe.

    Formula (from BRAIN docs):
        sub_sharpe >= scale * sqrt(sub_size / full_size) * alpha_sharpe
    """
    if full_size <= 0 or sub_size <= 0:
        return float("inf")
    return scale * (sub_size / full_size) ** 0.5 * alpha_sharpe


def as_dict(gates: GateConfig = DEFAULT_GATES) -> dict[str, float]:
    """Return gates as a plain dict (for printing/JSON). All GateConfig
    fields are floats so the return type is precisely `dict[str, float]`."""
    return asdict(gates)
