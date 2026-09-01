# Section 5 M5 nonlinear index model

Section 5 is a frozen nonlinear M5 classifier selected with chronological data
only.  It was trained on the oldest half of the available history, tuned on the
next quarter, and evaluated on the newest quarter.  Each market had to pass on
its own; pooling markets hid losses in individual symbols.

Only NDX100 survived.  SPX500, US30, GER40, XAUUSD and the FX symbols were not
enabled because their out-of-sample result was negative or too thin.  Adding
them would increase activity while reducing expected money.

The exact Jarvis replay over the newest 45 days, with spread and the broker
minimum-lot rules, produced 206 resolved trades (4.6 per calendar day), a 55.3%
win rate, +11.39R and +EUR 35.70 from EUR 203.  This is a positive historical
sample, not a promise of future profit: its daily result is only 0.71 standard
deviations above zero and August contributes 78% of the total.

The generic break-even manager must not manage this family.  In the same replay
it changed +11.39R into -16.31R by cutting 76 eventual winners while rescuing
37 losers.  Orders tagged `JARVIS-S5-NL-M5` therefore retain the measured fixed
one-ATR stop and one-R target at the broker.  All other strategy families keep
their existing management policy.

The reproducible search is `scripts/search_multimarket_section.py`; the frozen
live implementation is `analysis/section_five_m5.py`.
