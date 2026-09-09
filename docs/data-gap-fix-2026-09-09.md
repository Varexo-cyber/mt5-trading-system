# M5 quarantine on NDX100, SPX500 and XAUUSD

The VPS reports 73 missing M5 bars on both indices and 55 on gold. All
1,478 decisions per instrument were stopped by data availability/quarantine.
The actual timestamps have not yet been exported from the VPS.

Eightcap's published Labor Day schedule for 7 September 2026 lists an early
close at 19:50 GMT+3 for NDX100/SPX500 and 21:20 GMT+3 for XAUUSD:
https://www.eightcap.com/en/traders/trading-hours/ (checked 9 September 2026).
The 90-minute difference matches the 18-bar difference in the screenshots.
This is strong evidence of holiday misclassification, not confirmation of
the precise VPS gap boundaries.

The overlay now declares only those three exact symbols' early-close periods
through midnight of the published server date (21:00 UTC). Normal overnight
hours beyond that endpoint are deliberately not invented. The gap counter
deducts only absent slots entirely covered by these intervals, once even if
intervals overlap. It preserves actual bars, staleness checks, required history,
and the existing gap tolerance. Other symbols and dates are unaffected.

Regression tests reproduce 73/55 absent slots: the bounded calendar removes
50/32 respectively, leaving 23 subject to the normal tolerance. Longer outages
still fail. These are synthetic fixtures, not recovered broker data.

Run `datacheck.cmd` on the VPS after updating. It reads MT5 only and prints
the timestamp offset, actual largest M5 gaps, counts before/after the calendar,
and PASS/FAIL for the full live context. A failure remains a failure and prints
its complete reason. Restarting Jarvis clears its in-memory quarantine and
loads the new settings. Existing 24-hour journal reports retain past refusals.

Validation: 162 tests passed across history closures, data manager, quarantine,
configuration and partial timeframe context. No VPS execution was performed.
