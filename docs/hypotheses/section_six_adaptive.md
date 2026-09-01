# Section 6 adaptive nonlinear models

Section 6 was rebuilt after the original M1 candle lane lost after costs.  The
replacement procedure fits one nonlinear model per market on the preceding 60
days, chooses execution parameters without touching the final month, and then
replays that final month with recorded spread and the EUR 203 position sizer.

XAUUSD M5 survived.  The research resolver returned 29 trades, 55.2% wins,
+0.294R per trade and EUR +14.96.  The exact Jarvis order route was deliberately
run as a second check because it permits repeated non-overlapping signals and
holds fixed broker exits longer.  It returned 309 resolved trades in 30 days,
44.7% wins, +21.90R and EUR +63.00.  The generic break-even manager changed the
measured strategy, so `JARVIS-S6-AU-M5` retains its one-ATR stop and 1.5R target.

The SPX500 candidate did not survive the exact route: research showed +0.126R
per trade, but Jarvis returned -4.86R.  It remains disabled.  GER40 and US30
failed own-chart, fixed-clock, adaptive and relative-index searches.  They are
not enabled merely to increase activity.  NDX100 remains covered by Section 5,
and SPX500 by Sections 4 and 7.

This is not a profit guarantee.  The exact gold sample is only 0.82 standard
deviations above zero, so the live section has an independent breaker after six
consecutive losses and must earn a longer forward record.

With the other measured live families on XAUUSD, NDX100 and SPX500, the exact
30-day portfolio replay returned 510 resolved trades (17.0 per calendar day),
51.6% wins, +31.66R and EUR +90.68.  This combined figure is also only 0.98
standard deviations above zero and is reported as a measurement, not a promise.
