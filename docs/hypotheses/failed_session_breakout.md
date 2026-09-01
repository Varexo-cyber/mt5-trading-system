# Section 7: failed SPX500 session breakout

This section fades the first M5 close outside the SPX500 range formed from
07:00 through 09:00 broker time. It may trigger only until 12:00, uses half the
range as its stop, and targets 1.5R.

The parameters were selected on days 91-135 of the 180-day Eightcap archive.
The untouched days 136-180 produced 20 resolved trades, 50% winners,
+0.185R/trade and EUR +7.45 at EUR 203 equity after historical spread and the
configured cost model.

The exact Jarvis dry-run over the newest 45 days, including confluence,
position sizing and the live break-even exit, produced 23 trades, 20 positive
exits, +7.32R and EUR +26.90. The fixed-stop counterfactual was +10.52R, so the
account's break-even manager reduced this sample by 3.20R.

This is a small sample, not a profit guarantee. It is restricted to SPX500
because the same search did not produce an executable validation candidate on
GER40, NDX100, US30 or any available FX pair at EUR 203. Its independent
section breaker stops it after 12 trades if more than 75% lose, or after six
consecutive losses.
