# Module: Claude market scout

## What it measures

Claude receives a compact point-in-time set of closed D1/H4/H1/M15/M5 bars and
must independently choose WAIT, LONG or SHORT. It records a human-readable
thesis, its strongest counterargument, concrete invalidation and target levels,
risks, detected patterns and the condition that would make it reconsider.

## Interface

- Direction: `WAIT`, `LONG`, or `SHORT`.
- Confidence: 0..1, parsed and bounded locally.
- Price levels: optional floats; invalid orientation or reward/risk fails closed.
- It has no broker client and cannot submit an order.
- Every request and response is written to the AI exchange audit ledger.

## Measured edge

| Split | Trades | Win rate | Avg R | Expectancy | Max DD |
|---|---:|---:|---:|---:|---:|
| Train (60%) | not tested | | | | |
| Validation (20%) | untouched | | | | |
| Holdout (20%) | untouched | | | | |

Configurations tested: 0 of 9. Deflated Sharpe: unavailable.

## Known limitations

An LLM can produce coherent but false explanations, is sensitive to prompt and
model changes, and sees only the supplied point-in-time evidence. It has no
order book, future data, private institutional flow or privileged news. Model
confidence is not a calibrated probability of profit.

## Weight

No static confluence weight. PAPER may collect originated candidates. Live can
only use the scout as logged context/confirmation until validation passes.
