# Section 10: XAUUSD M1 first-retest continuation

Section 10 trades XAUUSD in two windows, 03:00-07:00 and 13:00-19:00 UTC. A closed M1 bar
must break the preceding 20-bar channel by at least 0.75 ATR. The first later
closed-bar retest of that level is entered in the break direction, with the
stop 0.75 ATR beyond the level and a 1.5R target. The EMA20 slope over three
already-closed M5 bars must agree: M1 finds the quick setup and M5 confirms
the immediate trend.

The proposed counterparty is the trader fading a meaningful channel break.
If the first retest holds, the old boundary has changed sides and trapped
inventory is expected to fuel continuation. Later retests are excluded
because the resting liquidity has already been consumed.

The M5 slope and the no-entry block from 07:00 through 12:59 UTC were selected
from the same 180-day Eightcap history. The exact engine replay from 3 March
through 31 August, with historical spread, EUR 203 sizing, fixed SL/TP and the
20:50 pause flatten, resolved all 446 trades: 46.9% wins, +50.88R and +2.27
day-clustered sigma. All six months were positive. The chronological halves
were +42.21R and +8.68R. The weaker late half and the fact that the hour filter
was calibrated on this sample matter: this is not an independent holdout and
is not a promise of live profit.

The account now keeps Section 10 on its measured fixed SL/TP because the
dry-run found break-even reduced the same entries by about 15R. It is still
flattened at 20:50 UTC before the daily gold pause. Section 6 is not changed
by this detector; moreover S6's old 20:00-02:00 entry window and S10's
03:00-19:00 outer entry window never overlap.

These are backtest results, not a promise of live profit. Its independent
breaker evaluates after 40 trades and stops it above 55% losses or after eight
consecutive losses.
