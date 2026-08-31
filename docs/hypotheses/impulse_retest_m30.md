# Hypothesis: impulse_retest_m30

This is not a new detector or a new parameter search. It is the shipped
`impulse_retest` rule on M30, added as a separately named live instance so its
weight, journal record and breaker cannot be shared with M15.

The mechanism is unchanged: a decisive channel break consumes the resting
orders at a level, price returns to that frozen level, and the surviving queue
supports the retest. The losing side is the trader who buys the extended break
or sells the first return through the level.

The account sweep reported 99 resolved M30 trades over 100 days, 83% wins,
+9.20 managed R and EUR +122.51. That row predates the correction that charges
every taken trade its spread, so it is supporting evidence rather than a clean
net-profit claim. It is live at the owner's explicit instruction on 31 August
2026, with the same parameters, 1R target and breaker policy as M15.

