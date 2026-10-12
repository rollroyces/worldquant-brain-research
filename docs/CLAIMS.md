# CLAIMS — Numeric Claims in the Alpha Templates Playbook

This file is the audit trail for every numeric claim in
`docs/ALPHA_TEMPLATES.md`. It exists because none of the claims have
been verified against a real BRAIN backtest on this machine — they
all originate from the community write-ups cited in the playbook
header and from the
[alexisdpc/WorldQuant-alpha-trading](https://github.com/alexisdpc/WorldQuant-alpha-trading)
repo, which is itself an aggregation of forum posts and BRAIN
consultant reports.

If you actually run a template against the real BRAIN API and want
to mark a claim as verified, edit the row in the table below and
commit the result alongside the alpha's sim_id or settings.

## Status legend

- **[community]** — value is from a community write-up; not verified
  on this machine.
- **[verified]** — value was reproduced by running the template
  against the real BRAIN API on this account. Add the sim_id.
- **[disputed]** — value contradicts another source; flag for review.
- **[unverified]** — placeholder; no source currently.

## Templates — claim verification

| # | Template | Sharpe | Fitness | Turnover | Drawdown | Status |
|---|---|---|---|---|---|---|
| 1 | PV1 — Classic 30-day reversion | 1.2–1.7 | 0.8–1.2 | 15–25% | 5–8% | [community] |
| 2 | PV2 — Volume-gated 5-day reversion | 1.3–1.8 | 1.0–1.4 | 10–20% | 4–7% | [community] |
| 3 | PV3 — VWAP intraday reversion | 1.2–1.6 | 0.9–1.3 | 8–15% | 4–6% | [community] |
| 4 | PV4 — Risk-adjusted momentum | 1.1–1.5 | 0.8–1.1 | 10–18% | 5–9% | [community] |
| 5 | PV5 — Skew-gated reversal | 1.2–1.6 | 0.9–1.2 | 15–25% | 5–8% | [community] |
| 6 | F1 — Operating yield | 1.2–1.6 | 0.9–1.2 | 5–12% | 4–7% | [community] |
| 7 | F2 — Earnings yield | 1.0–1.4 | 0.7–1.0 | 4–10% | 5–9% | [community] |
| 8 | F3 — Asset turnover | 1.0–1.4 | 0.7–1.0 | 4–10% | 5–8% | [community] |
| 9 | F4 — Leverage (low = safer) | 1.0–1.4 | 0.7–1.0 | 5–12% | 4–7% | [community] |
| 10 | F5 — Book-to-price | 0.9–1.3 | 0.6–0.9 | 4–10% | 6–10% | [community] |
| 11 | A1 — Earnings revision | 1.3–1.8 | 1.0–1.4 | 8–16% | 4–6% | [community] |
| 12 | A2 — Estimate dispersion | 1.1–1.5 | 0.8–1.1 | 8–15% | 5–8% | [community] |
| 13 | A3 — Target-price upside | 1.1–1.5 | 0.8–1.1 | 8–15% | 5–8% | [community] |
| 14 | S1 — News polarity | 1.2–1.6 | 0.9–1.2 | 15–30% | 5–8% | [community] |
| 15 | S2 — Short interest | 1.0–1.4 | 0.7–1.0 | 6–14% | 5–9% | [community] |
| 16 | S3 — Implied-minus-realized vol | 1.1–1.5 | 0.8–1.1 | 8–16% | 4–7% | [community] |
| 17 | S4 — CHN turnover-vol | 1.2–1.6 | 0.9–1.2 | 12–20% | 5–8% | [community] |
| 18 | C1 — Quality + Value blend | 1.3–1.7 | 1.0–1.3 | 5–12% | 4–7% | [community] |
| 19 | C2 — Momentum + Reversion blend | 1.3–1.7 | 1.0–1.4 | 12–22% | 4–6% | [community] |
| 20 | C3 — Revisions × Quality | 1.3–1.7 | 1.0–1.4 | 8–16% | 4–7% | [community] |

**Summary:** 20 templates × 4 metrics = 80 numeric claims. All 80 are
currently **[community]**. Zero have been **[verified]** on this
machine.

## Threshold claims (not from templates, but also unverified)

| Claim | Source | Status |
|---|---|---|
| Sharpe > 1.25 is the US/Delay-1 passing threshold | BRAIN docs (cited) + Quantt.co.uk + gentlecactus.top + alexisdpc | [community] |
| Fitness > 1.0 is the passing threshold | BRAIN docs + alexisdpc + Quantt.co.uk | [community] |
| Turnover must be in [1%, 70%]; sweet spot 12–15% | Quantt.co.uk + alexisdpc | [community] |
| Self-Correlation must be < 0.7 | gentlecactus.top + alexisdpc | [community] |
| Self-Corr > 0.7 is overridden by Sharpe > 1.375 (10% above threshold) | gentlecactus.top | [community] |
| Sub-Universe Sharpe cutoff: `0.75 × √(sub/full) × alpha_sharpe` | dafu-zhu.github.io + BRAIN docs (cited) | [community] |
| Weight concentration ≤ 10% of book in one stock | BRAIN docs (cited) | [community] |
| Daily Challenge Score cap = 2,000 | gentlecactus.top + jikegang.com | [community] |
| 10,000 Challenge Score = Gold level = consultant threshold | BRAIN docs (cited) + jikegang.com | [community] |

## How to add a verified claim

1. Run the template against the real BRAIN API via the harness:
   ```bash
   export BRAIN_EMAIL=you@example.com
   export BRAIN_PASSWORD=your-password
   python3 scripts/brain_harness.py simulate \
       --expression "rank(ts_mean(close, 30) - close)" \
       --region USA --universe TOP3000 --delay 1 \
       --neutralization Subindustry --decay 4 --truncation 0.08
   ```
2. Note the sim_id from the output.
3. Edit the relevant row in this file:
   ```diff
   - | 1 | PV1 — Classic 30-day reversion | 1.2–1.7 | 0.8–1.2 | 15–25% | 5–8% | [community] |
   + | 1 | PV1 — Classic 30-day reversion | 1.2–1.7 | 0.8–1.2 | 15–25% | 5–8% | [verified] sim_id=abc123 Sharpe=1.45 |
   ```
4. Commit the change with the sim_id in the commit message.

## Why this file exists

The templates playbook is meant to be a starting point, not a verified
backtest result. The 80 numeric ranges above are typical outcomes from
BRAIN community reports — they reflect what people *have seen*, not
what you *will see*. The BRAIN test period is famously noisy: an alpha
that hit Sharpe 1.6 once might hit 0.8 the next month. Treat these
ranges as guidance, not as predictions, and always cross-validate
locally with `scripts/cross_validate.py` before submitting.
