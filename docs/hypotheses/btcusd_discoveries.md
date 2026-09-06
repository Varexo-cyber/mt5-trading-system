# Sections 15 / 16 / 17 — BTCUSD, three frozen discoveries

> The shared record for `section_fifteen_btc_m1.md`,
> `section_sixteen_btc_m5.md` and `section_seventeen_btc_m15.md`. Each of
> those carries its own parameters; this carries what is true of all three,
> including the part that is uncomfortable.

## What these are

Three detectors found by searching 180 days of actual Eightcap BTCUSD bars.
All three run the same class, `analysis/gold_cross_discoveries.py`, on one
market and one clock each, with a frozen parameter set. Each owns its own
trigger and stop — they are in `strategy_owned_entry_families`, so the generic
higher-timeframe trend veto does not run on top of them.

| section | clock | mechanism | polarity | stop | target | session (UTC) | break-even |
|---|---|---|---|---|---|---|---|
| 15 | M1 | `adaptive_channel_100` | −1 | 3.0 ATR | 2.0 R | 00:00–07:00 | 1.5 R |
| 16 | M5 | `adaptive_channel_25` | −1 | 6.0 ATR | 2.0 R | all day, weekdays | none |
| 17 | M15 | `trend_pullback` | +1 | 4.0 ATR | 4.0 R | all day, all week | 1.0 R |

## The thing that has to be said first

**All three periods were inspected to discover them.** There is no untouched
holdout, and there cannot be one for the data they were found on — that data
is spent. The rule this repository wrote for itself when they were added said
so in the config, in these words:

> These are deliberately shadow-only: all three periods were inspected to
> discover them, so only later forward data may grant real-money permission.

On **6 September 2026** the owner promoted all three to
`live_enabled_modules` after running his own replays and finding them
positive. That is his account and his decision, and this file exists so the
decision is on the record with its condition next to it rather than quietly
overwritten.

So: the only genuine out-of-sample these three will ever get is what happens
from the promotion date onward. Every figure predating it was measured on
data that was looked at first.

## What actually limits the downside

Not a measurement — a breaker. `risk.section_breakers` gives each of the
three a 60-trade window, 40 minimum trades, more than 60 % losers or eight
losses in a row, and the section switches itself off with nobody present.
That, plus the account position cap and the per-trade risk percentage, is the
whole containment.

## What promoting them changed besides the allowlist

Their `weights` went from 0.0 to 1.0. That is not cosmetic and it is not a
second decision: `ConfluenceEngine` tests `if weight > 0`, so a section on the
allowlist at weight zero is computed, logged, and counted by nothing. It would
have taken zero trades forever with nothing anywhere saying why — the failure
mode this codebase has produced more often than any other. The allowlist entry
without the weight would have been a change that did nothing while reading
like a change that did something.

## Known gaps, so they are not discovered later

**The entry-timing gate still runs on them.** Every other live section is in
`entry_timing_exempt_families`; these three are not. `AWAITING_CONFIRMATION`
was the single largest refusal reason on the one fully-counted live day (140
of 414 setups), and it applies a generic three-bar M5 adverse-drift test to
detectors whose thesis is a closed-bar event on their own clock. Expect it to
refuse setups the replay took. Whether that is protection or interference is
an open question and a one-line change either way.

**The replay charged a research execution allowance.** `RAW_BTC_EXECUTION_
ALLOWANCE_R` plus the recorded spread. Live pays the real fill, and BTCUSD
spread widens exactly when these detectors fire.

**Section 16 has no break-even move.** `shadow_break_even_at_r: 0.0` means the
broker stop stands until it is hit or the target is. On a 6.0 ATR stop that is
a wide thing to leave alone.
