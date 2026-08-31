# Hypothesis: order_block_m15

This is the shipped `order_block` detector on M15, not a new rule. It has a
separate module name because each clock must keep its own weight, history and
automatic breaker.

The mechanism is unchanged: the last opposite candle before a large impulse
marks an area where the losing side may have unfinished orders. The strategy
rests an order in that block and exits at 1R before that local inventory is
exhausted.

The account sweep reported 1,301 resolved M15 trades over 100 days, 76% wins,
+40.80 managed R and EUR +12.17. The small euro result despite the positive R
and the later cost-accounting correction make this materially weaker evidence
than the original FX research. It is live at the owner's explicit instruction
on 31 August 2026, without changing detector thresholds and with its own
breaker.

