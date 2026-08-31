# Hypothesis: order_block_h1

This is the shipped `order_block` detector on H1, kept as a separately named
instance so an H1 losing run cannot stop or contaminate M1, M15 or M30.

The mechanism is unchanged: the final opposing candle before a violent move
is treated as unfinished local inventory. A return to that area is entered
with the same stop construction and 1R target as the measured detector. The
losing counterparty is the side whose resting order was overrun by the impulse.

The account sweep reported 409 resolved H1 trades over 100 days, 79% wins,
+18.70 managed R and EUR +199.90. That output predates the correction that
subtracts spread from every taken trade, so it does not establish final net
expectancy. It is live at the owner's explicit instruction on 31 August 2026,
with unchanged thresholds and an independent breaker.

