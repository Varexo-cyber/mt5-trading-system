# Hypothesis: Claude market scout

> Pre-registered before implementation or any outcome inspection.

## Claim

An independent Claude review of compact, closed-bar D1/H4/H1/M15/M5 evidence
can identify a small set of directional opportunities or explicit wait
conditions that add information beyond the deterministic confluence engine.
It should improve paper-trade expectancy only when its direction, invalidation
and target survive the unchanged broker, risk, cost and execution gates.

## Economic rationale

The deterministic modules deliberately isolate a few simple mechanisms. A
multimodal language model may be able to combine their cross-timeframe context,
recognise conflicts, and express a conditional thesis without assigning a
fixed score to every visual pattern. The opposing side is assumed to be traders
reacting to a local move while ignoring higher-timeframe structure, liquidity
or cost. This is only a plausible mechanism, not evidence that an LLM predicts
returns.

The scout is useful only if it adds repeatable incremental information. Fluent
explanations, model size and apparent confidence do not count as evidence.

## Operational definition

- Inputs are secret-free and contain only closed OHLCV bars, current quote,
  asset class, current exposure and immutable risk constraints.
- Claude independently returns `WAIT`, `LONG`, or `SHORT`, confidence,
  observations, thesis, counter-thesis, invalidation price, target price,
  detected patterns, risks, and what it is waiting for.
- The scout may be called at most once per configured global cooldown and at
  most 24 times per UTC day. The durable cooldown survives restarts.
- Repeated calls for the same closed-bar signature are forbidden.
- A missing, malformed, timed-out or low-confidence response is `WAIT`.
- Claude cannot call MT5. It proposes intent and price levels; deterministic
  code normalises prices, verifies direction and reward/risk, sizes downward,
  reruns every hard gate, and alone may submit an order.
- In PAPER, a scout-originated candidate may proceed when confidence is at
  least 0.80, both price levels are valid and planned reward/risk is at least
  2.0.
- In real-money modes, the scout is observation and independent confirmation
  only. It cannot originate or resize a live trade before validation.
- The existing final Claude veto remains separate: scouting asks “what do you
  see?”, while the final review asks “may this exact executable proposal pass?”

## Predictions, stated in advance

- Expected paper expectancy after costs: **-0.10R to +0.20R per trade**. The
  range deliberately includes failure.
- Expected incremental expectancy versus deterministic-only candidates:
  **-0.05R to +0.10R**.
- Expected qualified scout ideas: **5 to 40 per liquid instrument per year**.
- Expected `WAIT` rate: **at least 70%**.
- Conditions where this should fail: stale bars, regime transitions, wide
  spreads, scheduled gaps, ambiguous ranges, unsupported asset classes,
  prompt sensitivity, model-version changes, or explanations that merely
  restate indicators without incremental predictive value.

## Test design

- Development: EURUSD, GBPUSD and USDJPY.
- Cross-instrument holdback: AUDUSD and USDCAD.
- Separate research buckets for crypto, equities, indices and metals; none may
  inherit an FX result.
- Data split: oldest 60% train / next 20% validation / final 20% untouched.
- Compare deterministic-only, scout-confirmed, scout-vetoed and scout-originated
  PAPER candidates using identical point-in-time prices and costs.
- Prompt variants: concise evidence, evidence plus counter-thesis request, and
  evidence plus explicit wait-condition request.
- Confidence thresholds: 0.75, 0.80 and 0.85.
- **Number of configurations tested: 9.** Correct the final Sharpe for all nine.
- Minimum sample: 100 closed paper trades per evaluated role; model upgrades
  start a new cohort and do not silently pool results.

## Result

Not tested. Results will be documented in
`docs/modules/claude_market_scout.md`.

- [ ] At least 100 trades per role
- [ ] Net costs and adverse slippage included
- [ ] Broad confidence/prompt plateau
- [ ] Held-back instruments retain the sign of the effect
- [ ] Deflated Sharpe remains positive
- [ ] Final 20% inspected exactly once

## Verdict

`PAPER origin enabled for evidence collection; real-money origin disabled`
