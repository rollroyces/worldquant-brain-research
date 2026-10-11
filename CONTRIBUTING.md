# Contributing

Thanks for considering a contribution. This is a small, focused
toolkit; the bar for a useful PR is intentionally low.

## Development setup

```bash
git clone https://github.com/rollroyces/worldquant-brain-research.git
cd worldquant-brain-research
pip install -r requirements.txt
```

The `requirements.txt` covers runtime + dev deps (pytest, mypy).

## Test workflow

```bash
# Run the full test suite (72 cases across brainlib, validator,
# scheduler, security, CSV-grid)
python -m pytest tests/ -v

# Run static type checks
python -m mypy brainlib/ scripts/
python -m mypy tests/

# Smoke-test the harness in mock mode (no BRAIN credentials needed)
python scripts/brain_harness.py --mock simulate \
    --expression "rank(ts_mean(close, 30) - close)" \
    --region USA --universe TOP3000 --delay 1 \
    --neutralization Subindustry --decay 4 --truncation 0.08
```

All three should pass before opening a PR. The CI workflow runs the
same checks plus a harness demo job on every push.

## Code conventions

- **Public surface lives in `brainlib/`**: gate rules, Fitness formula,
  sub-universe cutoff. This is the canonical source — if you change
  a threshold, change it there.
- **Scripts import from `brainlib`**: never duplicate the gate logic
  inline in the scripts. The validator already does this; the harness
  keeps a backwards-compat shim that delegates.
- **Tests pin down behavior**: any change to a gate threshold,
  metric formula, or rotation invariant must come with a test that
  would have failed before the change.
- **No secrets in code**: never commit credentials. The harness
  reads `BRAIN_EMAIL` / `BRAIN_PASSWORD` from the environment only.
- **Mock mode for offline testing**: when adding a code path that
  hits the BRAIN API, also wire a mock branch (`if self.mock: ...`)
  so the test suite stays runnable without credentials.

## Pull request process

1. Open a PR against `main`. The PR template will ask for a one-line
   summary, the related issue (if any), and a checklist of test
   results.
2. CI must be green: `pytest` matrix, `mypy`, and `demo` job all
   pass.
3. Reviewer will look for:
   - Tests that lock the change in
   - No duplicated logic (extend `brainlib` instead)
   - Mock-mode support if the change touches the harness
   - README / CHANGELOG updates if user-facing

## Filing issues

Use the bug-report template if it's a defect, feature-request
template if it's an enhancement. Include:
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- The Python version and OS you ran on
- Whether you ran in mock mode or against the real BRAIN API

## Areas where help is welcome

- **Backtest against real returns data** — the validator currently
  only exercises synthetic panels. A PR that adds a real-data
  fixture (and confirms the gate thresholds make sense against it)
  would be valuable.
- **Performance** — the validator does pivots per alpha per
  sub-universe. With hundreds of alphas this gets slow.
- **mypy coverage** — `scripts/` are looser than `brainlib/`.
  Tightening them up is welcome.
- **Additional templates** — the playbook has 20 alphas. More
  families (options, news sentiment, intraday microstructure)
  would broaden coverage.
