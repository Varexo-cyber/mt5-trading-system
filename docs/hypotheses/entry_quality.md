# Hypothesis — execution-price entry quality

## Registered before implementation

Date: 2026-08-10

The direction of a setup and the quality of the price available *now* are two
different questions. The current runner blocks price moving against a proposal,
but treats an arbitrarily large move with the proposal as confirmation. A market
order can therefore chase the end of an M5 impulse. A paid AI review adds another
20–30 seconds during which the fill can move away from the reviewed entry while
the original stop, target and volume remain unchanged.

## Expected mechanism

Late continuation buyers and sellers pay urgency to participants taking profit
after an impulse. Requiring a non-extended entry, or waiting until the setup is
offered again after a pullback, should reduce immediate adverse excursion without
changing the directional theory. Binding the order to the reviewed price shape
should prevent latency from silently changing reward-to-risk and monetary risk.

## Pre-registered intervention

1. Measure closed-M5 directional extension, last-candle body, distance from EMA,
   and location inside the recent range in ATR-normalised units.
2. Mark an entry `WAIT_RETEST`, rather than permanently vetoing it, when price is
   both directionally extended and at the directional edge of its recent range.
3. Require the latest closed entry bar not to be materially moving against the
   proposal.
4. After the AI response, refresh market/account/position state. Refuse the order
   when response latency or price drift makes the reviewed snapshot stale.
5. When the drift is small, rebuild sizing and rerun filters, spread, runway,
   risk and margin checks using the fresh executable price.

All thresholds are typed configuration. No risk percentage, position limit,
drawdown rule or open-position management rule changes as part of this work.

## Measurements

Record for every deep decision:

- favourable extension in ATR;
- last directional candle body in ATR;
- EMA distance in ATR;
- directional range location;
- post-review price drift in ATR and AI latency;
- explicit reason: enter now, wait for retest, or stale after review.

Existing MFE/MAE, shadow-trade and management-baseline resolution remain the
outcome layer. A future conclusion needs at least 100 resolved entries per setup
family and an untouched out-of-sample window. Until then this is an execution
integrity correction, not evidence of a positive trading edge.

## Falsification

The hypothesis fails if, on an untouched sample, waiting does not reduce early
MAE or materially worsens expectancy after costs. It also fails operationally if
the gate dominates decisions for normal liquid sessions; that indicates thresholds
are unreachable rather than that every market is bad. Parameter changes then follow
the repository's shadow/validation process, never a handful of recent trades.
