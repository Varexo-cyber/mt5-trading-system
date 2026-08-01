# Hypothesis: market structure

> Pre-registered before implementation or backtesting. The ranges below are
> predictions, not results.

## Claim

On liquid FX majors during the London and New York sessions, an H1 candle close
beyond the most recently confirmed external swing, in the direction of the H4
structure, should predict continuation to 2R more often than the break-even rate
after conservative spread, slippage, commission, and swap costs.

The claim applies to a confirmed break of structure (BOS), not to every new high
or low. A change of character (CHoCH) is treated as a warning that the prior
direction may have ended; it is not pre-registered as a standalone entry trigger.

## Economic rationale

A confirmed swing concentrates stop orders from counter-trend positions and
entry orders from breakout participants. A close through that swing shows that
available liquidity was not sufficient to reject price immediately. When the
higher-timeframe structure points the same way, forced exits on the losing side
and slower institutional repositioning can sustain continuation beyond the
initial break.

The losing side is therefore traders holding against the higher-timeframe move
with stops beyond the swing, plus liquidity providers who temporarily absorb
the break and must hedge when price does not revert. This mechanism may be too
widely known or fully priced in; the test is allowed to reject it.

## Operational definition

- A swing is fractal-based and becomes visible only after its right-hand
  confirmation bars have closed. Its timestamp may be plotted on the pivot bar,
  but no decision may use it before the confirmation bar.
- External structure is the sequence of confirmed swings used for directional
  bias. Internal swings may describe pullbacks and CHoCH, but cannot independently
  satisfy the external BOS claim.
- A bullish BOS requires a closed candle above the latest confirmed external
  swing high; bearish is symmetric. Wick-only breaks do not qualify.
- A BOS against the current external direction is labelled CHoCH until a new
  external sequence is confirmed. It is not silently counted as trend continuation.
- Entry for validation is the next bar's open after confirmation. The structural
  stop is beyond the opposing confirmed swing plus an ATR buffer. Target is 2R.
- If one bar touches both stop and target, the stop is assumed to occur first.
- Equal highs/lows are descriptive key levels in this module, not extra evidence
  for the BOS score. They must not double-count a later liquidity-sweep module.

## Predictions, stated in advance

- Expected win rate at a 2R target after costs: **36% to 48%**.
- Expected average expectancy: **+0.05R to +0.30R per qualified signal**.
- Expected number of qualified signals: **40 to 120 per instrument per year**
  before the phase-3 risk and news filters reduce the tradable count.
- Expected directional lift: at least **5 percentage points** over an
  unconditional, session-matched continuation baseline.
- Conditions where this should **fail**: range-bound or very low-volatility
  regimes; wick breaks that do not close beyond structure; signals against H4
  structure; rollover and high-impact-news windows; symbols with discontinuous
  pricing or materially wider costs than the FX majors.

The hypothesis is rejected if validation expectancy is not positive after
costs, if the held-back instruments do not retain the direction of the effect,
if the deflated Sharpe is non-positive, or if performance exists only at one
parameter value rather than on a broad plateau.

## Test design

- Instruments (development set): **EURUSD, GBPUSD, USDJPY**.
- Instruments held back for the cross-instrument test: **AUDUSD, USDCAD**.
- XAUUSD is excluded from the primary claim because its session behaviour,
  contract economics, and stop scale are materially different. It requires a
  separate pre-registration rather than being used to rescue an FX result.
- Timeframes: **H1 signal and stop structure, H4 directional structure**.
- Target history: **2015-01-01 through 2025-12-31**, subject to obtaining
  complete, timezone-normalised bid/ask-aware data. A shorter sample must still
  produce at least 100 qualified signals per evaluated configuration.
- Data split: oldest 60% train / next 20% validate / final 20% untouched.
- Primary comparison: qualified BOS aligned with H4 structure versus a
  session-, direction-, and volatility-matched unconditional baseline.
- Parameters swept: symmetric fractal confirmation lookback **2, 3, or 4
  closed bars** on each side; BOS close buffer **0.00, 0.05, or 0.10 ATR**.
  ATR period and structural-stop buffer are fixed to the separately configured
  risk assumptions and are not tuned in this module's edge test.
- **Number of configurations tested: 9.** Any later variant increments this
  number before its result is viewed and is included in the deflated-Sharpe
  correction.
- Robustness checks: pessimistic costs, stop-first intrabar ambiguity, no forming
  bars, parameter plateau, per-year results, cross-instrument holdback, and
  separate trending/ranging volatility regimes.

## Result

Not tested yet. Results will be recorded in `docs/modules/market_structure.md`.

- [ ] At least 100 trades per evaluated configuration
- [ ] Broad parameter plateau, not a spike
- [ ] Holds on instruments it was not developed on
- [ ] Deflated Sharpe still positive after correcting for 9 configurations
- [ ] Final 20% looked at exactly once

## Verdict

`weight 0, awaiting implementation and validation`
