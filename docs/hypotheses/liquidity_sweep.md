# Hypothesis: liquidity_sweep

> Written before any backtest.

## Claim

When a candle trades through the highest high (or lowest low) of the previous
20 bars and then **closes back inside** that range, price continues away from
the swept extreme often enough to pay for the spread and a stop placed beyond
the sweep wick.

## Economic rationale

This is the strongest mechanism of the three currently weighted modules, and
the only one where the counterparty is unambiguous.

Stop orders cluster immediately beyond obvious extremes — recent highs and
lows are where a human puts a stop, and where a broker's stop-hunting
algorithms and large participants alike know to look. Those stops are resting
*market* orders. When price reaches them they fire, providing a burst of
liquidity to whoever wants to accumulate the opposite side at a better price.
Once the cluster is exhausted, the artificial pressure ends and price returns
inside the range.

**Who loses money on the other side:** two named groups. Traders whose stop sat
at the obvious level and were taken out at the worst price of the move. And
breakout traders who entered on the break of the 20-bar extreme and are now
holding a position that closed back against them.

**Where this rationale is weak:** the mechanism justifies *a reaction*, not
necessarily a tradable one. The reaction may be entirely consumed by the spread
and the wide stop the setup requires. A sweep 0.2 ATR deep on a 1.4-pip spread
is not the same trade as one 1.5 ATR deep, and this module currently treats
both as signals with only a confidence difference.

**Overlap warning.** `market_structure` also detects equal highs and lows, which
are the same market feature seen from a different angle. If both modules fire
on the same candle the confluence engine will count one event as two
independent votes. The correlation between these two modules' signals must be
measured before either carries live weight, and the confluence score corrected
for it. This is exactly the double-counting the project brief warned about.

## Predictions, stated in advance

- Win rate: 45–58%. Higher than momentum; the entry is at a location.
- Average R: 0.20 to 0.40 expectancy.
- Signals: 25–60 per instrument per year on M15.
- **Conditions where this should fail:** during high-impact news, when the
  "sweep" is just the start of a genuine repricing rather than a stop raid
  (the news filter already blocks this window); on illiquid instruments where
  the 20-bar extreme is noise rather than a level anyone watches; and when the
  sweep depth is under ~0.3 ATR, which is indistinguishable from ordinary wick
  behaviour.

## Test design

- Development instruments: EURUSD, GBPUSD, USDJPY
- Held back: AUDUSD, USDCAD
- Timeframe: M15 signal (H1 fallback)
- Split: 60/20/20
- Parameters swept: lookback 15/20/30, minimum sweep depth 0.0/0.3/0.5 ATR
- **Configurations tested: 9**
- **Additional required measurement:** signal correlation with
  `market_structure` over the same window.

## Result

Not run. See `docs/modules/liquidity_sweep.md`.

- [ ] ≥100 trades
- [ ] Broad parameter plateau
- [ ] Holds on AUDUSD and USDCAD
- [ ] Deflated Sharpe positive after correcting for 9 configurations
- [ ] Correlation with market_structure measured and corrected for
- [ ] Final 20% looked at exactly once

## Verdict

**Research weight 0.8, live-enabled without measured evidence.** Accepted by the
owner as part of the written-off EUR 100 experiment, and recorded as a
deviation from the validation protocol.
