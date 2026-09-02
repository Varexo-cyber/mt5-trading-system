# Section 6 adaptive nonlinear models

Section 6 was rebuilt again after its first live week contradicted the earlier
research resolver.  The production-semantics search is now reproducible with
`scripts/search_section_six_exact.py`.  It uses the same rolling 260-bar model
frame, bid/ask entry, recorded spread, EUR 203 sizing, one-position rule and M1
first-touch resolution as Jarvis.  An earlier version accidentally sized in the
1% default mode; that result was rejected and the tool now explicitly selects
the 2% micro-live contract.

The deployed XAUUSD M5 route is deliberately simple: model magnitude at least
0.15, long only, one contiguous 20:00-02:00 UTC session, 0.8 ATR stop and 3R
target. It is a current-regime forward-test candidate, not a validated strategy.
The exploratory session search inspected the newest period, so that period may
not subsequently be called an untouched holdout.

The full Jarvis replay, after putting those rules in the production module,
returned 171 resolved trades in 30 days (5.7 per calendar day), 34.5% wins,
+0.313R per trade, +53.60R and EUR +165.72 at a fixed EUR 203 starting equity.
Eighteen of 25 trading days were positive and the daily result was +2.15 sigma
from zero.  The fixed stop and configured break-even route were identical on
this sample.

The strategy owns its complete entry plan.  For this family the generic HTF,
chase, confirmation-lifecycle and target-reach gates are advisory or bypassed;
otherwise the live system would no longer be trading what was measured.  News,
spread/cost, account exposure, position sizing and broker validation remain hard
safety checks.

The SPX500 candidate did not survive the exact route: research showed +0.126R
per trade, but Jarvis returned -4.86R.  It remains disabled.  GER40 and US30
failed own-chart, fixed-clock, adaptive and relative-index searches.  They are
not enabled merely to increase activity.  NDX100 remains covered by Section 5,
and SPX500 by Section 7.  Section 4 was taken offline after the later exact
seven-day portfolio replay returned -2.21R for it.

This is not a profit guarantee.  The exact sample clears +2 sigma but contains
only 171 trades and one month, short of the predeclared 200-trade/three-month
evidence bar.  It therefore remains explicitly unproven and has its own breaker
at more than 75% losses over 30 trades or ten consecutive losses.

The 180-day audit split the history into an old 120-day selection block, a later
30-day validation block and the newest 30-day holdout. No plain stop/target
candidate survived the selection requirements. Session routes that were
positive in both selection and validation all failed or produced no trades in
the newest holdout; adding a causal multi-day trend agreement did not repair
that failure. This negative result is why the live route is called experimental
despite its strong most-recent-month replay.
