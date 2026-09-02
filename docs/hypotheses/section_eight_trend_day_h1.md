# Section 8: SPX500 prior-day continuation

Section 8 follows SPX500 only when the previous UTC day closes in the outer
decile of its own range. The entry is evaluated from closed H1 bars during
00:00-02:00 UTC, the stop is one H1 ATR and the target is 1R.

The mechanism is short-horizon continuation: a day that holds its close at an
extreme has left one side unable to mean-revert before settlement, and the
next session is expected to continue that imbalance briefly.

On the 180-day Eightcap archive, after historical spread and EUR 203 position
sizing, the frozen rule produced 26 resolved trades, 65.4% target winners and
+6.99R. The chronological early/late split was +0.44R and +6.55R; maximum
drawdown was -3.20R.

This is a positive but very small sample and is not a profit guarantee. It is
live only at the owner's explicit instruction after seeing that limitation.
Its independent breaker evaluates after 15 trades and stops it above 60%
losses or after six consecutive losses.
