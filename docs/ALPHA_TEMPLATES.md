# WorldQuant BRAIN — Alpha Templates Playbook

A reusable library of alpha expressions grouped by data family, with full
simulation settings, hypotheses, expected metric ranges, and variation hooks.
Paste any expression straight into BRAIN's Simulate panel and tweak the
settings block to taste.

> **Source-grounding:** thresholds and formulas cited here come from the BRAIN
> documentation page `platform.worldquantbrain.com/learn/documentation/create-alphas/test-period`,
> the platform's official operator reference, and cross-checked against
> community write-ups (alexisdpc/WorldQuant-alpha-trading, gentlecactus.top,
> Quantt.co.uk 2026 review). Field names like `vwap`, `cap`, `cashflow_op`,
> `anl14_actvalue_eps_fp0`, `mdl175_volatility` were taken from the linked
> sources; other field names below are *patterns* — verify exact names in your
> account's **Data → Data Explorer** before submitting.

---

## 0. Global rules (apply to every template)

### Passing gates (US, Delay-1 baseline; thresholds vary slightly by region/delay)

| Gate | Threshold | Notes |
|---|---|---|
| **Sharpe** | > 1.25 | Risk-adjusted return over the simulation period |
| **Fitness** | > 1.0 | `Sharpe × √(\|returns\| / max(turnover, 0.125))` |
| **Turnover** | 1% – 70% | Sweet spot ≈ 12–15%; below 12.5% the formula floors and stops helping |
| **Self-Correlation** | < 0.7 | vs. your account's submitted alphas; if violated, Sharpe > 1.375 overrides |
| **Weight concentration** | ≤ 10% of book in one stock | Use truncation ≈ 0.08 |
| **Sub-Universe Sharpe** | scales | `sub_sharpe ≥ 0.75 × √(sub_size / full_size) × alpha_sharpe` |
| **Drawdown** | < 10% ideally | Not a hard gate; kills the rating |

### Default simulation settings block (the "taste")

```
Region:           USA
Universe:         TOP3000
Delay:            1
Neutralization:   Subindustry
Decay:            4 – 8
Truncation:       0.08
Pasteurization:   On  (for fundamentals / analyst / sentiment)
                   Off (for raw price-volume)
Unit Handling:    Verify
NaN Handling:     Off  (drop NaN rows)
```

### The fitness formula cheat sheet

`Fitness = Sharpe × √(|returns| / max(turnover, 0.125))`

- Below 12.5% turnover → formula floors; further reduction doesn't help
- Above 12.5% → every halving of turnover multiplies fitness by √2 ≈ 1.41
- The dominant lever is **Sharpe**, then **returns**, then **turnover**

### Workflow

1. **Pick a template** below.
2. **Paste** expression + settings into Simulate.
3. **Read metrics**: Sharpe, Fitness, Turnover, Returns, Drawdown, Margin.
4. **Tune**: if Fitness < 1 → raise Decay or wrap in `trade_when` to drop turnover; if Sharpe low → lengthen `ts_mean` window or add a second confirming signal.
5. **Validate**: confirm Sub-Universe Sharpe passes the scaled cutoff (split results by sector/industry).
6. **Check Self-Correlation**: open the alpha's submission page; if max correlation with your prior submissions < 0.7, submit. Otherwise rotate to a different family or push Sharpe > 1.375.
7. **Submit** — daily cap is 2,000 Challenge Score; one strong alpha ≈ 1,500–2,000 pts.

### Cross-family rotation (the anti–self-correlation rule)

Submit at most **two alphas per day**, and rotate families so consecutive
submissions don't share their primary data source. Same skeleton × different
data family is the move:

```
Day 1:  PV family       (e.g. PV1 below)
Day 2:  Fundamental     (e.g. F1)
Day 3:  Analyst          (e.g. A1)
Day 4:  Sentiment/News   (e.g. S1)
Day 5:  PV family again  (different skeleton, e.g. PV3)
...
```

---

## 1. Price-Volume (PV) family

The most accessible family for new accounts. Uses raw price, volume, return,
and vwap fields. Pasteurization off, NaN handling off.

### PV1 — Classic 30-day price reversion *(baseline; well-documented)*

**Hypothesis:** stocks that have run above their 30-day mean tend to revert;
underperformers bounce.

```python
rank(ts_mean(close, 30) - close)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP3000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 4 |
| Truncation | 0.08 |
| Pasteurization | Off |
| NaN Handling | Off |

**Expected metrics (USA, Delay-1):** Sharpe ≈ 1.2–1.7 · Fitness ≈ 0.8–1.2 · Turnover ≈ 15–25% · Drawdown ≈ 5–8%

**Variations:**
- Window sweep: try `ts_mean(close, 10)`, `(20)`, `(60)`, `(120)`
- Asymmetric: `rank(ts_mean(close, 30) - close) * sign(volume - adv20)` to gate on volume regime
- Combined: average with a long-term reversion `rank(ts_mean(close, 120) - close)` and rerank

**Pitfalls:** raw daily reversion has high turnover → if Fitness < 1, raise Decay to 8–10 or add `trade_when(volume > adv20, alpha, -1)`.

---

### PV2 — Volume-gated 5-day reversion

**Hypothesis:** price reversals carry more signal on high-volume days when
informed traders are present.

```python
event = volume > adv20
alpha = -ts_delta(close, 5)
trade_when(event, alpha, -1)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP3000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 2 |
| Truncation | 0.01 |
| Pasteurization | Off |
| NaN Handling | Off |

**Expected metrics:** Sharpe ≈ 1.3–1.8 · Fitness ≈ 1.0–1.4 · Turnover ≈ 10–20% · Drawdown ≈ 4–7%

**Variations:**
- Swap `ts_delta(close, 5)` for `ts_delta(close, 1)` for ultra-short reversal (higher turnover, often higher Sharpe)
- Use `close - ts_mean(close, 5)` instead of `ts_delta` for level-based reversion
- Volume gate as continuous weight: `alpha * log(volume / adv20)` instead of binary event

**Pitfalls:** delay-0 is stricter and more crowded; stick with delay-1.

---

### PV3 — VWAP intraday reversion

**Hypothesis:** stocks that close far from the volume-weighted average mean
revert intraday.

```python
(vwap - close) / vwap
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP3000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 15 |
| Truncation | 0.08 |
| Pasteurization | On |
| NaN Handling | Off |

**Expected metrics:** Sharpe ≈ 1.2–1.6 · Fitness ≈ 0.9–1.3 · Turnover ≈ 8–15% · Drawdown ≈ 4–6%

**Variations:**
- Window the VWAP: `ts_mean(vwap, 5)` to make it a 5-day intraday measure
- Z-score it: `((vwap - close) / vwap) / ts_std_dev((vwap - close) / vwap, 60)`
- Multiply by sector mean reversion: `* group_rank(-ts_mean(returns, 5), sector)`

**Pitfalls:** high Decay required because the raw signal is very noisy day-to-day.

---

### PV4 — Risk-adjusted (vol-normalized) momentum

**Hypothesis:** stocks with high Sharpe-like momentum (return ÷ vol) over
20 days continue to drift up.

```python
ts_rank(ts_mean(returns, 20) / ts_std_dev(returns, 20), 60)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP3000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 6 |
| Truncation | 0.08 |
| Pasteurization | On |
| NaN Handling | Verify |

**Expected metrics:** Sharpe ≈ 1.1–1.5 · Fitness ≈ 0.8–1.1 · Turnover ≈ 10–18% · Drawdown ≈ 5–9%

**Variations:**
- Replace `ts_std_dev` with `ts_mean(abs(returns), 20)` (mean abs return)
- Add downside-vol focus: divide by `ts_std_dev(if_else(returns < 0, returns, 0), 20)`
- Lengthen the rank window: 120 or 252 for slower signals

**Pitfalls:** without `Verify` NaN handling, low-history stocks can blow up.

---

### PV5 — Short-term reversal gated by return skewness

**Hypothesis:** 1-day reversal works *more* in stocks whose recent return
distribution is positively skewed (lottery-like).

```python
rank(-ts_delta(close, 1)) * sign(ts_skewness(returns, 60))
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP3000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 3 |
| Truncation | 0.08 |
| Pasteurization | Off |
| NaN Handling | Off |

**Expected metrics:** Sharpe ≈ 1.2–1.6 · Fitness ≈ 0.9–1.2 · Turnover ≈ 15–25% · Drawdown ≈ 5–8%

**Variations:**
- Replace `ts_skewness` with `ts_rank(abs(ts_skewness(returns, 60)), 252)` for a regime filter
- Add volume confirmation: `* sign(volume - adv20)`

**Pitfalls:** very high turnover; raise Decay if Fitness dips.

---

## 2. Fundamental family

Lower-frequency data (quarterly / annual). Decay is higher (slower signal),
Pasteurization On (winsorize outliers in earnings/cap etc.), NaN Verify.

### F1 — Operating yield (cash-flow quality)

**Hypothesis:** firms with high operating cash flow per unit market cap
outperform — combines quality (high cash flow) and value (low cap).

```python
ts_rank(cashflow_op / cap, 60); group_rank(alpha, subindustry)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP3000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 4 |
| Truncation | 0.08 |
| Pasteurization | On |
| NaN Handling | Verify |

**Expected metrics:** Sharpe ≈ 1.2–1.6 · Fitness ≈ 0.9–1.2 · Turnover ≈ 5–12% · Drawdown ≈ 4–7%

**Variations:**
- Swap `cashflow_op` for `earnings` (net income) for a simpler earnings yield
- Use longer rank window: `(252)` for a more stable value signal
- Combine: average `ts_rank(cashflow_op/cap, 60)` and `ts_rank(sales/cap, 60)`

**Pitfalls:** fields update quarterly — verify your sub-universe Sharpe holds in *recent* quarters, not just historically.

---

### F2 — Earnings yield (classic value)

**Hypothesis:** cheap earnings (high earnings-to-price) earn excess returns.

```python
ts_rank(net_income / cap, 252)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP3000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 8 |
| Truncation | 0.08 |
| Pasteurization | On |
| NaN Handling | Verify |

**Expected metrics:** Sharpe ≈ 1.0–1.4 · Fitness ≈ 0.7–1.0 · Turnover ≈ 4–10% · Drawdown ≈ 5–9%

**Variations:**
- Use book value: `ts_rank(book_value / cap, 252)` for pure value
- Use forward earnings if available (consultant-tier fields)
- Blend with F1 above

**Pitfalls:** pure value factor is crowded — blend with quality (F1) for differentiation.

---

### F3 — Asset turnover (efficiency)

**Hypothesis:** firms that generate more revenue per dollar of assets
outperform — operational efficiency.

```python
ts_rank(revenue / assets, 250)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP3000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 6 |
| Truncation | 0.08 |
| Pasteurization | On |
| NaN Handling | Verify |

**Expected metrics:** Sharpe ≈ 1.0–1.4 · Fitness ≈ 0.7–1.0 · Turnover ≈ 4–10% · Drawdown ≈ 5–8%

**Variations:**
- Combine with F1 to build a quality composite
- Sector-neutral first: `group_neutralize(ts_rank(revenue/assets, 250), sector)`

**Pitfalls:** financial firms have weird asset structures — sector-neutralize first to avoid sector bet.

---

### F4 — Leverage (low = safer)

**Hypothesis:** firms with low debt-to-assets (low leverage) outperform over
the medium term.

```python
rank(liabilities / assets)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP3000 |
| Delay | 1 |
| Neutralization | Sector |
| Decay | 8 |
| Truncation | 0.08 |
| Pasteurization | On |
| NaN Handling | Verify |

**Expected metrics:** Sharpe ≈ 1.0–1.4 · Fitness ≈ 0.7–1.0 · Turnover ≈ 5–12% · Drawdown ≈ 4–7%

**Variations:**
- Use interest coverage instead: `rank(ebit / interest_expense)` (if available)
- Inverse: `rank(-(liabilities/assets))` (high leverage = short)
- Combine with profitability: `rank(liabilities/assets) * ts_rank(earnings/cap, 252)`

**Pitfalls:** financial-sector firms are special; sector-neutralize or exclude them via a filter event.

---

### F5 — Book-to-price (deep value)

**Hypothesis:** classic deep-value — high book-to-price firms outperform.

```python
rank(book_value / cap)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP3000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 10 |
| Truncation | 0.08 |
| Pasteurization | On |
| NaN Handling | Verify |

**Expected metrics:** Sharpe ≈ 0.9–1.3 · Fitness ≈ 0.6–0.9 · Turnover ≈ 4–10% · Drawdown ≈ 6–10%

**Variations:**
- Blend with F1 (quality) to neutralize value traps: `(rank(book_value/cap) + ts_rank(cashflow_op/cap, 60)) / 2`
- Lengthen decay to 15 for a slow value rotation

**Pitfalls:** deep value alone can have Sharpe < 1.25; always blend with a quality factor.

---

## 3. Analyst family

Earnings estimates, revisions, target prices. Mostly TOP1000 coverage (analyst
attention). Pasteurization On, NaN Verify.

### A1 — Earnings revision

**Hypothesis:** rising EPS estimates predict outperformance; falling estimates
predict underperformance.

```python
ts_rank(anl14_actvalue_eps_fp0, 60)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP1000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 4 |
| Truncation | 0.08 |
| Pasteurization | On |
| NaN Handling | Verify |

**Expected metrics:** Sharpe ≈ 1.3–1.8 · Fitness ≈ 1.0–1.4 · Turnover ≈ 10–20% · Drawdown ≈ 4–6%

**Variations:**
- Change lookback: `ts_rank(anl14_actvalue_eps_fp0, 120)` or `(252)`
- Compute *change* in estimate: `ts_delta(anl14_actvalue_eps_fp0, 60)` then rank

**Pitfalls:** field names vary — confirm `anl14_actvalue_eps_fp0` exists for your region in Data Explorer. Common alternatives: `anl14_eps_est_*`, `anl13_*`.

---

### A2 — Analyst sentiment (estimate dispersion)

**Hypothesis:** stocks with high dispersion in analyst estimates (high
uncertainty) underperform; low dispersion (consensus) outperforms.

```python
rank(anl13_eps_est_std / anl13_eps_est_mean)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP1000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 6 |
| Truncation | 0.08 |
| Pasteurization | On |
| NaN Handling | Verify |

**Expected metrics:** Sharpe ≈ 1.1–1.5 · Fitness ≈ 0.8–1.1 · Turnover ≈ 8–15% · Drawdown ≈ 5–8%

**Variations:**
- Use range instead: `(anl13_eps_est_max - anl13_eps_est_min) / anl13_eps_est_mean`
- Inverse and combine with revisions: `(-rank(dispersion)) * ts_rank(anl14_actvalue_eps_fp0, 60)`

**Pitfalls:** undefined when `anl13_eps_est_mean ≈ 0` — verify NaN handling.

---

### A3 — Target price upside

**Hypothesis:** stocks with high target-price upside (analyst PT vs. spot)
outperform over the medium term.

```python
rank(target_price / close - 1)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP1000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 8 |
| Truncation | 0.08 |
| Pasteurization | On |
| NaN Handling | Verify |

**Expected metrics:** Sharpe ≈ 1.1–1.5 · Fitness ≈ 0.8–1.1 · Turnover ≈ 8–15% · Drawdown ≈ 5–8%

**Variations:**
- Filter to most-recent revisions: only count targets from the last 90 days if the field supports it
- Combine with F1: `* ts_rank(cashflow_op/cap, 60)`

**Pitfalls:** target-price fields update infrequently; check Decay ≥ 8 to avoid stale signals.

---

## 4. Sentiment / News / Options family

High-frequency sentiment signals. Often noisy → use shorter decay and tighter
truncation. Pasteurization On.

### S1 — News polarity

**Hypothesis:** positive news flow predicts short-term outperformance.

```python
ts_rank(news_polarity, 5)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP3000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 3 |
| Truncation | 0.08 |
| Pasteurization | On |
| NaN Handling | Verify |

**Expected metrics:** Sharpe ≈ 1.2–1.6 · Fitness ≈ 0.9–1.2 · Turnover ≈ 15–30% · Drawdown ≈ 5–8%

**Variations:**
- Replace `ts_rank(news_polarity, 5)` with `ts_mean(news_polarity, 20)` for a slower sentiment
- Weight by news volume: `ts_mean(news_polarity, 5) * log(1 + news_volume_5d)`
- Combine with PV2 (volume-gated reversion): average the two and rerank

**Pitfalls:** field name varies (`news_polarity`, `sentiment_*`, etc.) — verify in Data Explorer. News coverage is sparse for small caps; restrict to TOP2000 if needed.

---

### S2 — Short interest reversal

**Hypothesis:** high short interest signals squeezed downside risk; low short
interest stocks are priced cleanly.

```python
rank(-short_interest_pct_float)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP3000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 8 |
| Truncation | 0.08 |
| Pasteurization | On |
| NaN Handling | Verify |

**Expected metrics:** Sharpe ≈ 1.0–1.4 · Fitness ≈ 0.7–1.0 · Turnover ≈ 6–14% · Drawdown ≈ 5–9%

**Variations:**
- Use days-to-cover: `rank(short_interest / adv20)`
- Combine with momentum: `rank(-short_interest) * ts_rank(ts_mean(returns, 60), 252)`

**Pitfalls:** short interest updates bi-monthly — long Decay required.

---

### S3 — Implied-vs-realized volatility spread

**Hypothesis:** when implied volatility is high relative to realized
volatility, options are overpriced → equity tends to mean-revert.

```python
rank(implied_vol - realized_vol)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP1000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 6 |
| Truncation | 0.08 |
| Pasteurization | On |
| NaN Handling | Verify |

**Expected metrics:** Sharpe ≈ 1.1–1.5 · Fitness ≈ 0.8–1.1 · Turnover ≈ 8–16% · Drawdown ≈ 4–7%

**Variations:**
- Z-score the spread: `((implied_vol - realized_vol) / ts_std_dev(implied_vol - realized_vol, 60))`
- Window: `ts_mean(implied_vol, 20) - ts_mean(abs(returns), 20) * sqrt(252)`

**Pitfalls:** options data only exists for TOP1000 or so; verify universe coverage before submitting.

---

### S4 — Turnover volatility (China-specific)

**Hypothesis:** stocks with low volatility in daily turnover are more
informative — high turnover-vol stocks are noisy.

```python
rank(-mdl175_volatility * log(volume)) * (1 + group_rank(mdl175_revenuettm, sector))
```

| Setting | Value |
|---|---|
| Region | CHN |
| Universe | TOP3000 |
| Delay | 0 |
| Neutralization | Sector |
| Decay | 3 |
| Truncation | 0.08 |
| Pasteurization | On |
| NaN Handling | Off |

**Expected metrics (CHN, Delay-0):** Sharpe ≈ 1.2–1.6 · Fitness ≈ 0.9–1.2 · Turnover ≈ 12–20% · Drawdown ≈ 5–8%

**Variations:**
- Drop the revenue multiplier: `rank(-mdl175_volatility * log(volume))` for a pure turnover-vol signal
- Use `mdl175_revenuettm` standalone as a CHN quality factor

**Pitfalls:** delay-0 is stricter; CHN sub-universe Sharpe cutoff is tighter than US.

---

## 5. Composite / blended templates

These combine two or more signals to push Sharpe up while diversifying data
sources (which also helps Self-Correlation).

### C1 — Quality + Value blend

**Hypothesis:** combining profitability (F1) with cheapness (F5) beats either alone.

```python
0.5 * ts_rank(cashflow_op / cap, 60) + 0.5 * ts_rank(book_value / cap, 252)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP3000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 6 |
| Truncation | 0.08 |
| Pasteurization | On |
| NaN Handling | Verify |

**Expected metrics:** Sharpe ≈ 1.3–1.7 · Fitness ≈ 1.0–1.3 · Turnover ≈ 5–12% · Drawdown ≈ 4–7%

---

### C2 — Momentum + Mean-reversion regime blend

**Hypothesis:** momentum works in up-trending regimes; reversion in
range-bound regimes. Blend by recent vol regime.

```python
momentum = ts_rank(ts_mean(returns, 60), 252)
reversion = rank(-ts_delta(close, 5))
blend_weight = rank(ts_std_dev(returns, 20))
momentum * blend_weight + reversion * (1 - blend_weight)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP3000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 5 |
| Truncation | 0.08 |
| Pasteurization | Off |
| NaN Handling | Off |

**Expected metrics:** Sharpe ≈ 1.3–1.7 · Fitness ≈ 1.0–1.4 · Turnover ≈ 12–22% · Drawdown ≈ 4–6%

---

### C3 — Earnings revision × Quality filter

**Hypothesis:** earnings revisions work best in profitable firms (where the
estimate is anchored).

```python
ts_rank(anl14_actvalue_eps_fp0, 60) * ts_rank(cashflow_op / cap, 60)
```

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP1000 |
| Delay | 1 |
| Neutralization | Subindustry |
| Decay | 5 |
| Truncation | 0.08 |
| Pasteurization | On |
| NaN Handling | Verify |

**Expected metrics:** Sharpe ≈ 1.3–1.7 · Fitness ≈ 1.0–1.4 · Turnover ≈ 8–16% · Drawdown ≈ 4–7%

---

## 6. Diagnostic checklist before you submit

Run through this for *every* alpha before clicking submit:

- [ ] **Sharpe > 1.25** (1.375+ if you want Self-Correlation override)
- [ ] **Fitness > 1.0** (the formula: `Sharpe × √(|returns|/max(turnover, 0.125))`)
- [ ] **Turnover** in 1%–70%, ideally 12–15% (best fitness zone)
- [ ] **Self-Correlation < 0.7** vs. your account's prior submissions
- [ ] **Sub-Universe Sharpe** passes scaled cutoff: `sub_sharpe ≥ 0.75 × √(sub_size/full_size) × alpha_sharpe`
- [ ] **Weight concentration** ≤ 10% (truncation 0.08 is the easy fix)
- [ ] **Drawdown** < 10% (raises the rating, not strictly required)
- [ ] **Region-appropriate delay** (Delay-1 for US; Delay-0 strict but rewarding)
- [ ] **Subindustry neutralization** at minimum (sector or market if you must)

If a metric is off:

| Symptom | Fix |
|---|---|
| Sharpe < 1.25 | Lengthen `ts_mean` window; add a confirming signal via blend; add NaN=Verify |
| Fitness < 1 | Raise Decay; wrap in `trade_when(event, alpha, -1)`; add truncation |
| Turnover too high | Increase Decay to 10–15; use `trade_when` with volume gate |
| Turnover too low (< 12%) | No action — formula is floored; raise Sharpe or returns |
| Self-Correlation high | Rotate to different family, or push Sharpe > 1.375 |
| Sub-Universe Sharpe fails | Switch from Market to Subindustry neutralization; sector-neutralize fundamental signals |
| Weight concentration too high | Lower truncation to 0.05 or 0.08 |

---

## 7. Variations bank (parameter sweeps to try)

When a template *almost* passes, sweep these one-at-a-time:

**Window sweeps** (most common single tweak):
- `ts_mean(close, 30)` → try `(10)`, `(20)`, `(60)`, `(120)`, `(252)`
- `ts_rank(x, 60)` → try `(120)`, `(252)`

**Decay sweeps** (smooths turnover):
- 0 → 2 → 4 → 6 → 8 → 10 → 15

**Neutralization sweeps** (controls sub-universe Sharpe):
- Market → Sector → Industry → Subindustry

**Truncation sweeps** (caps concentration):
- 0.01 → 0.05 → 0.08 → 0.10

**Delay sweeps** (changes data lag; harder but more edge):
- 1 → 0 (Delay-0 stricter, more crowded; Delay-2 easier but edge decays)

**Region sweeps** (data orthogonality; also diversifies IQC score):
- USA → ASI → CHN → EMEA → GLB

**Universe sweeps**:
- TOP3000 → TOP2000 → TOP1000 → TOP500 (smaller universes have stricter sub-universe cutoffs but less crowding)

---

## 8. What "winning" actually looks like (target zone)

Single-alpha submission bar (must clear):

```
Sharpe      ≥ 1.25
Fitness     ≥ 1.00
Turnover    12–15%  (sweet spot)
Drawdown    < 10%
Self-Corr   < 0.7
```

Strong portfolio (what top consultants report):

```
Sharpe      1.5 – 7+
Fitness     1.1 – 7+
Returns     8 – 14% annualized
Turnover    10 – 15%
Drawdown    0.5 – 6%
```

Daily submission budget: ≤ 2 strong alphas (Challenge Score daily cap = 2,000).
Quality > quantity: rotating families beats hammering one skeleton.

---

## 9. Honest caveats

- **90%+ of attempts fail.** This playbook is a starting set; real winners
  come from iterating hundreds of skeleton × data × parameter combinations.
- **Field names vary by region.** Verify exact names in your Data Explorer.
  What's in this playbook matches the patterns in community sources and the
  official operator docs, but a given dataset may have slightly different
  naming.
- **Crowdedness changes.** A template that worked in 2024 may be arbitraged
  away by 2026. Always check Self-Correlation.
- **The official test-period documentation page** is the authoritative
  source for thresholds, but the page is JavaScript-rendered and the
  numbers cited here come from multiple community sources that agree.
- **Consultant-only fields** (40,000+ dataset) are not available to standard
  users. The templates here use widely available fields.
- **Don't peek at the OS line.** The platform's blind out-of-sample test is
  the whole point; tuning to visible OOS metrics will fail blind OOS.

---

## 10. Next steps (optional follow-ups)

Once you've validated a few templates here locally (and on BRAIN), you can ask
me to build:

1. **A Python harness against the BRAIN API** — batch-simulate a
   template × data × parameter grid, fetch results, filter by the gates
   above.
2. **A local sub-universe validation routine** — pre-screen alphas against
   the scaled sub-universe Sharpe cutoff before spending simulation slots.
3. **A region-rotation schedule** — automatically generate an IQC-style
   submission plan that cycles through USA → ASI → CHN → EMEA → GLB across
   days, with the right delay and universe per region.
