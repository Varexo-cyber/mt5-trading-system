# Hypothesis: trend_momentum

> Written before any backtest. Nothing here is a claim about measured
> performance; it is what we expect and what would prove us wrong.

## Claim

When the 20-EMA sits above the 50-EMA and is rising on both H4 and H1, price
continues in that direction often enough, and far enough, to pay for the spread
and a structural stop. Symmetrically for the short side.

Expected to work best on liquid FX majors during the London/New York overlap
and worst in low-ATR ranges — which is why `volatility_regime` gates it.

## Economic rationale

Time-series momentum is one of the few anomalies with a mechanism that does not
require anyone to be stupid:

- **Slow information diffusion.** A macro repricing does not complete in one
  candle. Institutions executing a large position work it over hours or days
  precisely to avoid moving the market, which mechanically produces continuation.
- **Risk-management flows.** Trend-following CTAs add to winners by mandate.
  Their buying is price-insensitive and predictable in direction.
- **Disposition effect.** Retail closes winners early and holds losers, which
  supplies liquidity into a trend and removes it against one.

**Who loses money on the other side:** counter-trend traders fading a move
before it is exhausted, and anyone whose stop sits at the obvious swing behind
the trend. Both are populated categories.

**Where this rationale is weak, stated up front:** the momentum literature is
strongest at monthly-to-annual horizons on cross-sectional equity, and much
thinner on intraday FX. At H1 the effect — if it survives at all — is small
relative to the spread. It is entirely plausible that this module has a real
edge on D1 and none on H1. That is a specific, testable failure mode.

## Predictions, stated in advance

- Win rate: 38–48%. This is a run-with-it module, not a high-hit-rate one.
- Average R: 0.15 to 0.35 expectancy if it works at all.
- Signals: 40–90 per instrument per year on H1.
- **Conditions where this should fail:** ranging regimes (ADX < 20), the Asian
  session, and any instrument whose typical spread exceeds ~15% of the ATR-based
  stop. If it does *not* degrade in those conditions, the result is suspect —
  a real momentum edge has to disappear where there is no momentum.

## Test design

- Development instruments: EURUSD, GBPUSD, USDJPY
- Held back for the cross-instrument test: AUDUSD, USDCAD
- Timeframes: H4 bias, H1 signal
- Split: oldest 60% train / next 20% validation / final 20% untouched
- Parameters swept: fast EMA 15/20/25, slow EMA 40/50/60, slope lookback 3/5/8
- **Configurations tested: 27** (needed for the deflated Sharpe)

## Result

Not run. See `docs/modules/trend_momentum.md`.

- [ ] ≥100 trades
- [ ] Broad parameter plateau, not a spike
- [ ] Holds on AUDUSD and USDCAD
- [ ] Deflated Sharpe positive after correcting for 27 configurations
- [ ] Final 20% looked at exactly once

## Verdict

**Research weight 1.0, live-enabled without measured evidence.** Accepted by the
owner as part of the explicitly written-off EUR 100 experiment. This is a
deviation from the validation protocol and is recorded as such, not hidden.
