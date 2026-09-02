# Section 9: USDJPY session-VWAP reversion

Section 9 trades USDJPY only. A closed M30 bar must finish at least two M30
ATR from the current UTC-session volume-weighted average price. The trade
fades that displacement with a one-ATR stop and a 1.5R target.

The proposed counterparty is a late directional buyer or seller entering
after an unusually large intraday displacement. The hypothesis is that this
inventory is poorly positioned and price partially reverts toward the
session's traded centre.

On the 180-day Eightcap archive, after historical spread and EUR 203 position
sizing, the frozen rule produced 22 resolved trades, 63.6% target winners and
+10.71R. The chronological early/late split was +5.93R and +4.78R; maximum
drawdown was -4.15R.

Twenty-two trades cannot establish a durable edge. This section is live only
at the owner's explicit instruction after seeing that limitation. Its own
breaker evaluates after 15 trades and stops it above 60% losses or after six
consecutive losses.
