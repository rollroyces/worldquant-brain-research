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
│   └── iqc_scheduler.py            # region-rotation submission planner
├── tests/
│   └── test_gates.py               # pytest suite for the gate logic
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
python3 -m pytest tests/ -v
```

The test suite pins down the canonical gate thresholds and Fitness formula
in `brainlib/gates.py` so a typo in the rules can't silently change what
counts as a passing alpha. 27 tests cover the Fitness formula, the sub-universe
Sharpe cutoff, every individual gate failure, multi-failure reporting, and
per-region / per-delay `GateConfig` overrides.
