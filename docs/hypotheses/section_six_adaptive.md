# Section 6 adaptive nonlinear models

Section 6 was rebuilt again after its first live week contradicted the earlier
research resolver.  The production-semantics search is now reproducible with
`scripts/search_section_six_exact.py`.  It uses the same rolling 260-bar model
frame, bid/ask entry, recorded spread, EUR 203 sizing, one-position rule and M1
first-touch resolution as Jarvis.  An earlier version accidentally sized in the
1% default mode; that result was rejected and the tool now explicitly selects
the 2% micro-live contract.

The repaired XAUUSD M5 route keeps model magnitude at least 0.15, long only,
one contiguous 20:00-02:00 UTC session, a 0.8 ATR stop and 3R target. A long is
now admitted only when the latest closed M5 price is also above the close from
12 closed M5 bars earlier. That causal confirmation was selected on the oldest
90 days, stayed positive on the next 45-day validation block and on the newest
45-day holdout. This is still calibration evidence, not a profit guarantee.

The old fixed-exit route was rejected by the 180-day audit: 856 trades, 25.0%
wins, -53.23R and EUR -137.78. The earlier strong 30-day result was one recent
regime and could not authorize that unchanged route.

The repaired exact replay uses the same latest-closed H1 ATR offset as the live
manager when it moves the stop to break-even. The earlier dry-run incorrectly
treated `0.10 ATR` as `0.10R`; on an M5 stop those are different distances. The
first corrected route was only +8.92R after the real concurrent-position cap.
A separately screened four-hour confirmation was then added alongside the
one-hour confirmation. The exact 180-day replay of that final route returned
331 resolved account trades (1.8 per calendar day), 61.0% positive exits,
+20.61R and EUR +63.59 at EUR 223.16 equity. Break-even added +8.18R versus the
same entries on fixed exits.

The result is not stable enough to call proven: only +0.98 sigma, four of six
calendar months positive, March -5.66R and June -10.48R. The best month carried
88% of the total. More importantly, the next VPS-only seven-day dry-run through
3 September lost 10.31R on 35 S6 trades; the local archive ends on 31 August,
so those newest days cannot honestly be re-tested here. The strategy therefore
has a zero-expectancy breaker over its latest 30 resolved live trades, judged
after 20, plus an eight-loss streak stop. That can limit a future bad regime;
it cannot guarantee profit.

The strategy owns its complete entry plan. For this family the generic HTF,
chase, confirmation-lifecycle and target-reach gates are advisory or bypassed;
otherwise the live system would no longer be trading what was measured.  News,
spread/cost, account exposure, position sizing and broker validation remain hard
safety checks.

Position management is deliberately local to S6: only its measured break-even
move may alter the original stop/target. Partial closes, trailing, give-back,
health and generic time exits return early for this broker comment. The dry-run
uses that same H1-ATR conversion. Section 10 remains on fixed SL/TP and therefore
does not inherit S6's break-even rule.

The SPX500 candidate did not survive the exact route: research showed +0.126R
per trade, but Jarvis returned -4.86R.  It remains disabled.  GER40 and US30
failed own-chart, fixed-clock, adaptive and relative-index searches.  They are
not enabled merely to increase activity.  NDX100 remains covered by Section 5,
and SPX500 by Section 7.  Section 4 was taken offline after the later exact
seven-day portfolio replay returned -2.21R for it.

This remains an experimental live route. The filter and management choice were
developed from these same 180 days, so the three blocks reduce obvious curve
fitting but do not constitute a fresh future sample. Forward results and the
automatic breaker decide whether it deserves to remain live.
