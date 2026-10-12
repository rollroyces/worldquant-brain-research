# WorldQuant BRAIN Research

A toolkit for building, validating, and submitting alpha expressions that pass
the [WorldQuant BRAIN](https://platform.worldquantbrain.com) test period gates.

## What's in here

```
worldquant-brain-research/
├── README.md                       # this file
├── .gitignore                      # ignores secrets, output artifacts, etc.
├── requirements.txt                # pip install -r requirements.txt
├── brainlib/                       # shared library code
│   ├── __init__.py
│   └── gates.py                    # canonical gate thresholds + evaluator
├── docs/
│   └── ALPHA_TEMPLATES.md          # 20 alpha templates + parameter sweeps
├── scripts/
│   ├── brain_harness.py            # batch simulator via BRAIN API
│   ├── subuniverse_validator.py    # local pre-screening of sub-universe Sharpe
│   ├── iqc_scheduler.py            # region-rotation submission planner
│   ├── cross_validate.py           # IS / OS split for overfitting detection
│   └── parameter_sweep.py          # robustness sweep across decay/truncation/neutralization
├── tests/
│   ├── test_gates.py               # canonical gate logic
│   ├── test_validator_metrics.py   # validator metric helpers + self-corr gate
│   ├── test_scheduler.py           # rotation invariants
│   ├── test_security.py            # credential hygiene
│   ├── test_csv_grid.py            # CSV-loading regression
│   └── test_cross_validation.py    # cross-validate + parameter-sweep helpers
└── examples/
    └── alpha_grid_example.csv      # example skeleton × data × parameter grid
```

## The plan (from our conversation)

| Step | Artifact | Status |
|---|---|---|
| 1 | Alpha templates playbook (`docs/ALPHA_TEMPLATES.md`) | ✅ done |
| 2 | BRAIN API harness (`scripts/brain_harness.py`) | ✅ done |
| 3 | Local sub-universe validator (`scripts/subuniverse_validator.py`) | ✅ done |
| 4 | IQC region-rotation scheduler (`scripts/iqc_scheduler.py`) | ✅ done |

## Quick start

### Read the templates

Open `docs/ALPHA_TEMPLATES.md` for the 20 alpha expressions across PV /
Fundamental / Analyst / Sentiment families, with full simulation settings,
hypotheses, expected metrics, and variation hooks.

### Run the harness against BRAIN

```bash
# Set your BRAIN credentials via environment variables. The harness does
# NOT accept a --password CLI flag on purpose: CLI args land in shell
# history, process listings, and CI logs.
export BRAIN_EMAIL="you@example.com"
export BRAIN_PASSWORD="your-password"   # never commit this

# Simulate a single alpha
python scripts/brain_harness.py simulate \
    --expression "rank(ts_mean(close, 30) - close)" \
    --region USA --universe TOP3000 --delay 1 \
    --neutralization Subindustry --decay 4 --truncation 0.08

# Run a batch from a grid file (template × data × parameter)
python scripts/brain_harness.py batch --grid examples/alpha_grid_example.csv

# Filter the results and submit the qualifying ones
python scripts/brain_harness.py submit --from-results results/sim_*.json
```

### Pre-screen locally before burning simulation slots

```bash
# Generate a returns panel (you provide this; see scripts/subuniverse_validator.py docs)
python scripts/subuniverse_validator.py \
    --alphas my_alpha_grid.csv \
    --returns-panel data/returns_panel.parquet \
    --sub-universe-column subindustry \
    --output results/local_validation.csv
```

### Cross-validate before submitting (IS / OS split)

The single biggest reason alphas fail the BRAIN test period is
overfitting to the in-sample window. `cross_validate.py` splits your
returns panel at a date (or fraction) and reports Sharpe on each half
plus the OS/IS ratio — the canonical overfitting diagnostic.

```bash
python scripts/cross_validate.py \
    --returns-panel data/returns_panel.parquet \
    --alpha-column alpha_test \
    --split-date 2025-04-01 \
    --output results/cross_validation.csv
```

Interpretation:
- **OS/IS ≥ 0.5** → probably real, worth a real submission
- **OS/IS 0.3–0.5** → marginal, watch closely
- **OS/IS < 0.3** → likely overfit, signal collapsed OOS
- **OS/IS < 0** → flipped signs, alpha is not real

### Robustness sweep across parameters

`parameter_sweep.py` runs an alpha across a grid of (decay, truncation,
neutralization) and reports a "fragility" verdict — pass rate, Sharpe
range, median, std. An alpha that only works at one exact parameter
combination is fragile; one that holds across most of the grid is
robust.

```bash
python scripts/parameter_sweep.py \
    --expression "rank(ts_mean(close, 30) - close)" \
    --region USA --universe TOP3000 --delay 1 \
    --sweep-decay 2,4,6,8 \
    --sweep-truncation 0.05,0.08,0.10 \
    --sweep-neutralization Market,Subindustry \
    --mock \
    --output results/sweep.csv
```
```

### Plan an IQC submission schedule

```bash
python scripts/iqc_scheduler.py \
    --start 2026-11-01 \
    --days 60 \
    --families PV,Fundamental,Analyst,Sentiment \
    --regions USA,ASI,CHN,EMEA,GLB \
    --output iqc_schedule.csv
```

## The gates (US, Delay-1)

| Gate | Threshold |
|---|---|
| Sharpe | > 1.25 |
| Fitness | > 1.0 (`Sharpe × √(\|returns\| / max(turnover, 0.125))`) |
| Turnover | 1% – 70% (sweet spot 12–15%) |
| Self-Correlation | < 0.7 |
| Weight concentration | ≤ 10% |
| Sub-Universe Sharpe | scaled (`sub_sharpe ≥ 0.75 × √(sub_size/full_size) × alpha_sharpe`) |
| Drawdown | < 10% ideally |

## Caveats

- Field names vary by region. Verify exact names in your **Data → Data Explorer**.
- Crowdedness changes. A template that worked in 2024 may be arbitraged away by 2026.
- Don't peek at the OS line. The platform's blind out-of-sample test is the whole point.
- 90%+ of attempts fail. Iterate hundreds of skeleton × data × parameter combinations.
- The harness scaffolds the BRAIN API but exact endpoints may have shifted; verify
  against your account's API documentation before relying on it at scale.

## Install dependencies and run the test suite

```bash
pip install -r requirements.txt
# Optional: install the pre-commit hooks so lint/format/type/smoke
# run automatically on every commit. See .pre-commit-config.yaml.
pip install pre-commit && pre-commit install
python3 -m pytest tests/ -v
python3 -m mypy brainlib/ scripts/ tests/
```

The test suite pins down the canonical gate thresholds and Fitness formula
in `brainlib/gates.py` so a typo in the rules can't silently change what
counts as a passing alpha. **72 tests** cover the Fitness formula, the
sub-universe Sharpe cutoff, every individual gate failure, multi-failure
reporting, per-region / per-delay `GateConfig` overrides, validator
metric helpers, IQC scheduler rotation invariants, credential-leak
regressions in the harness, and CSV-grid loading (the bug GitHub
Actions caught). `mypy` runs as a separate CI job to catch any-return
widening and other type drift.

## Continuous Integration

`.github/workflows/test.yml` runs three jobs on every push and pull
request against `main`:

1. **`pytest`** — matrix across Python 3.10 / 3.11 / 3.12.
2. **`mypy`** — static type check on `brainlib/`, `scripts/`, `tests/`.
3. **`demo`** — full harness in mock mode, uploads the result JSONs
   and IQC plan as build artifacts (`harness-demo-results`).

Push a branch and check the Actions tab for green/red.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and feature
requests use the GitHub issue templates under `.github/ISSUE_TEMPLATE/`.
