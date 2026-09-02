# Section 10: XAUUSD M1 first-retest continuation

Section 10 trades XAUUSD only between 03:00 and 19:00 UTC. A closed M1 bar
must break the preceding 20-bar channel by at least 0.75 ATR. The first later
closed-bar retest of that level is entered in the break direction, with the
stop 0.75 ATR beyond the level and a 1.5R target.

The proposed counterparty is the trader fading a meaningful channel break.
If the first retest holds, the old boundary has changed sides and trapped
inventory is expected to fuel continuation. Later retests are excluded
because the resting liquidity has already been consumed.

The exact 180-day Eightcap audit uses historical spread, EUR 203 sizing, one
position at a time and Jarvis' conservative live break-even rule. It produced
565 resolved trades (about 3.1 per calendar day), 79.3% positive-or-protected
gross exits, +82.00R and EUR +132.27. Its chronological halves were +41.86R
and +40.14R; maximum drawdown was -5.41R. After transaction costs, fewer exits
remain positive than the 79.3% gross figure.

These are backtest results, not a promise of live profit. Its independent
breaker evaluates after 40 trades and stops it above 55% losses or after eight
consecutive losses.
