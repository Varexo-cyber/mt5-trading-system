# Hypothesis: horizon-correct Claude evidence brief

## Problem

The final AI reviewer receives closed bars and the deterministic engine's
proposal, but has to recompute basic multi-timeframe facts from compact rows.
The target-reachability block is fixed to 24 H1 bars even when the proposal is
an M15 intraday trade, and the review cache is also fixed to H1. This can make
different short-horizon questions look identical and encourages repetitive
answers that paraphrase the engine rather than independently reading the data.

Anonymous HTTP 400 lines also hide which advisory call failed. A failed
supervision, reflection, scout and pre-trade review currently look identical in
the console even though they have different consequences.

## Pre-registered change

1. Add a deterministic `evidence_brief` to every pre-trade review. It will be
   calculated only from supplied closed OHLCV bars and will report, per
   timeframe, measured drift, range location, volatility, candle anatomy and
   directional alignment. Facts, engine claims and unavailable data will be
   explicitly separated.
2. Calculate target reachability on the proposal's own planning timeframe and
   expected horizon instead of always using 24 H1 bars.
3. Key cached reviews on the proposal shape and its planning-timeframe close,
   not a fixed H1 close.
4. Log the advisory operation and sanitised API error for failed review, scout,
   supervision and reflection calls. No prompt, key or credential may appear in
   the console.

## Expected result

- Claude can verify or contradict the proposal from a compact fact table rather
  than merely repeat module prose.
- An intraday trade is judged against intraday travel and fresh intraday bars;
  a swing remains judged against its swing horizon.
- Repeated reviews remain cached when the actual question is unchanged, while a
  materially different proposal is not mistaken for the old one.
- The next API 400 names its call type and the API's safe request-shape message,
  making it diagnosable without opening raw ledgers.

## Guardrails

- No risk percentage, position limit, sizing, drawdown, stop or target rule is
  changed.
- Derived fields are labelled measurements, not predictions or guarantees.
- Missing/short history is reported as unavailable and never imputed.
- Tick data is identified separately from closed-bar evidence.
- Claude remains a veto-only second opinion; it cannot alter the order plan.

## Validation

- Unit tests assert horizon-correct reachability, fact provenance, conflict
  reporting and proposal-aware cache invalidation.
- Request-shape tests assert unsupported schema constraints are removed and
  failures are sanitised.
- Full tests, Ruff, Black check and compilation must remain green.
