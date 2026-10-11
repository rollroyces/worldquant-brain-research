# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **LICENSE** (MIT) — previously missing from the public repo.
- **mypy.ini** + mypy in CI — static type checking for `brainlib/` and
  `scripts/`. Discovered 4 real type bugs while wiring up (any-return
  widening through `abs()`/`max()`, requests `Cookies` `__getitem__`
  returning `str | None`, `argparse` subparser `func` typed as `Any`).
- **`docs/CLAIMS.md`** — placeholder for template claim verification.
- **Issue / PR templates** under `.github/`.
- **`CONTRIBUTING.md`** — onboarding doc.
- **`scripts/cross_validate.py`** — IS / OS split with the OS/IS Sharpe
  ratio as the canonical overfitting diagnostic.
- **`scripts/parameter_sweep.py`** — robustness sweep across
  (decay × truncation × neutralization) with a fragility verdict.
- **`tests/test_cross_validation.py`** — 21 cases pinning both
  scripts.

### Changed
- `requirements.txt` — added `mypy>=1.7`.
- CI workflow now runs **three jobs** in parallel: `pytest` (matrix
  3.10/3.11/3.12), `mypy` (static type check), and `demo` (full
  harness in mock mode with artifact upload).

## [0.1.0] — 2026-10-11

Initial release. 72 pytest cases passing across brainlib gates,
validator metrics, scheduler rotation, security hygiene, and CSV-grid
loading.

### Components
- `brainlib/gates.py` — canonical Sharpe/Fitness/Turnover/SelfCorr/
  Concentration/Sub-Universe gate thresholds with `fitness()`,
  `evaluate()`, `sub_universe_cutoff()`, `as_dict()`.
- `scripts/brain_harness.py` — BRAIN HTTP API client with `--mock`
  mode. Security-hardened: masked credential repr, no `--password`
  CLI flag, sanitized error and submission prints.
- `scripts/subuniverse_validator.py` — local pre-screener that reads
  a returns panel, computes per-alpha and per-sub-universe Sharpe,
  applies all gates including self-correlation.
- `scripts/iqc_scheduler.py` — daily submission planner rotating
  data families, regions, and individual templates to defeat
  Self-Correlation.
- `docs/ALPHA_TEMPLATES.md` — 20 alpha expressions across PV,
  Fundamental, Analyst, Sentiment, Composite families with full
  simulation settings, hypotheses, expected metrics, and variation
  hooks.
- `examples/alpha_grid_example.csv` — 16-row sample grid (expressions
  properly quoted to survive csv.DictReader).

### Audits completed in 0.1.0
- **Correctness audit** (commit `be83b19`): 12 bugs fixed, including
  the validator silently skipping the self-correlation gate and a
  numpy/Series crash in `max_drawdown`.
- **Security audit** (commit `2426ce5`): 4 credential-leakage risks
  fixed (repr, `--password` flag, error/print sanitization).
- **CI setup** (commit `af0f8cb`): pytest across Python 3.10/3.11/
  3.12 on every push/PR.
- **Type tightening** (commit `7996193`): tightened `dict` / `Any`
  annotations.
- **GitHub integration** (commit `33ec69d`): harness demo job runs
  the full pipeline in mock mode on every push.
- **CSV bug fix** (commit `05d5099`): caught by GitHub Actions; the
  example CSV had unquoted commas in expressions.
- **LICENSE** (commit `66fa7f6`): MIT.
