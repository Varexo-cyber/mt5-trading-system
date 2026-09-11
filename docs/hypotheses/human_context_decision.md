# Human-context decision — preregistration

## Question

Can one conservative decision process outperform isolated setup modules by
requiring a coherent, closed-candle market story before it proposes a trade?

This is not a codification of one successful discretionary XAUUSD short. That
trade is an example of the quality of explanation required, not the pattern to
fit. The decision process may recognise continuation, failed-auction reversal,
or post-break retest. It may also conclude that no story is coherent.

## Information available at the decision

Only completed M5, M15, H1 and H4 candles, the current executable tick, and
levels derivable from those completed candles. The forming candle and future
bars are unavailable. Existing calendar, session, spread, position-book and
risk gates remain outside the module and mandatory.

## One decision contract

Every non-neutral proposal must name:

1. the higher-timeframe auction state and directional control;
2. the event that changed or continued that state;
3. a completed M5 confirmation rather than an intrabar impression;
4. the price level whose failure makes the story false;
5. the next opposing liquidity/structure objective;
6. the reward available after the structural stop and execution costs;
7. which supported story fired: continuation, failed auction, or break-retest.

If any item is unknown, contradictory, or economically untradeable, the output
is neutral. More trades are not a goal.

## Frozen first test

- Market: XAUUSD only.
- Window: the most recent 30 complete UTC days.
- Mode: shadow/replay only; no live allowlist entry.
- Outcome: first touch of the structural stop or a 2R objective, with recorded
  broker spread and the ordinary account gates.
- Report each story separately as well as combined, including trades, net R,
  win rate, expectancy, maximum drawdown, and early/late half.

No threshold is changed after reading this window. A changed definition starts
a new hypothesis and needs a new untouched window.
