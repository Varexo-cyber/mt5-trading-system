# m1_micro_breakout

Standalone quick-entry research module. It reads only closed M1 and M5 bars:
M5 supplies directional structure; the latest closed M1 bar must break a
compact micro-range with body, close-location and tick-volume confirmation.

The signal is an event and cannot repeat on every candle outside the range.
Its invalidation sits beyond the opposite side of the pre-break range. All
normal cost, spread, session, news, sizing, correlation, risk and Claude gates
still apply after it fires.

Status: experimental live observation permitted; OOS edge not yet established.
See `docs/hypotheses/m1_micro_breakout.md` for the registered claim and required
validation.
