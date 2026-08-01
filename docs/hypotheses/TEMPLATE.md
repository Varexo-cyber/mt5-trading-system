# Hypothesis: <module name>

> Filled in **before** the first backtest. Writing the explanation after seeing
> the result is not evidence — it is the definition of data mining.

## Claim

One sentence: what edge do you expect, on which instruments, in which regime?

## Economic rationale

Who is on the other side of this trade, and why do they keep losing money on
it? If there is no answer here, stop. A pattern without a mechanism is noise
that has not been falsified yet.

## Predictions, stated in advance

- Expected win rate: ___%  (state a range, not a point)
- Expected average R: ___
- Expected number of signals per instrument per year: ___
- Conditions where this should **fail**: ___

That last line matters most. A hypothesis that cannot fail cannot be tested.

## Test design

- Instruments (development set): ___
- Instruments held back for the cross-instrument test: ___
- Timeframe(s): ___
- Data split: oldest 60% train / next 20% validate / final 20% untouched
- Parameters swept, and their ranges: ___
- **Number of configurations tested: ___**  (needed for the deflated Sharpe)

## Result

Filled in after. Link to `docs/modules/<name>.md`.

- [ ] ≥100 trades
- [ ] Broad parameter plateau, not a spike
- [ ] Holds on instruments it was not developed on
- [ ] Deflated Sharpe still positive after correcting for the config count
- [ ] Final 20% looked at exactly once

## Verdict

`weight > 0` / `weight 0, kept for research` / `removed`
