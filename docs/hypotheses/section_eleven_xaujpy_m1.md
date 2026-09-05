# Sections 11 / 12 / 13 — XAUJPY, one searched mechanism, three clocks

> Written **before** the search runs. That is the only order in which this
> document is worth anything: an explanation written after seeing the result is
> not evidence, it is the definition of data mining. Sections 12 (M5) and 13
> (M15) point here.

## What replaced what

The previous section 11 was a fitted model per metal on the four gold crosses.
It failed on the one test that had no choices in it — the untouched holdout,
negative in four markets out of four:

| market | holdout R | per trade |
|---|---|---|
| XAUAUD | −193,22 | −0,052 |
| XAUEUR | −189,77 | −0,052 |
| XAUGBP | −148,93 | −0,041 |
| XAUJPY | −115,53 | −0,031 |

Walk-forward read +0,005 to +0,030 per trade. A sign flip on nearly fifteen
thousand trades. It is deleted, not disabled.

This is a different kind of thing on a narrower question: **one instrument,
XAUJPY, and a mechanism chosen by search rather than fitted** — a rule with a
name you can say out loud, from the same registry the search measures.

## Claim

On XAUJPY there is a repeatable intraday mechanism, on M1, M5 or M15, whose
edge survives the real Eightcap cost model, one-position-at-a-time, a
rate-matched random control and an untouched holdout — and which fires often
enough to give the account three to five trades a day.

Both halves of that sentence are the hypothesis. A mechanism that clears the
statistics at 0,5 trades a day does not answer what was asked; a mechanism at
five a day whose edge is inside its costs answers it wrongly.

## Economic rationale

XAUJPY is gold priced through a currency, so it has **two order books** and
they are open at different times. That is the whole reason to expect anything
here that XAUUSD does not have, and it is also the reason to expect the hour of
day to matter more than the mechanism.

The proposed counterparty: participants pricing one leg while the other leg is
thin. During Tokyo the metal leg has no informed flow; between 16:00 and 18:00
UTC London is gone while gold still trades, so the currency half of the quote
widens. A move made in a thin book by somebody who has to trade is a move with
nobody informed behind it, and that is what a fade is buying.

If the search finds only continuation mechanisms, that rationale is wrong and
should be said so rather than rewritten to fit.

## Predictions, stated in advance

- The surviving mechanisms will be **fades, not continuations**.
- Expected win rate at 1,5 R: **38–46%**
- Expected average R: **+0,02 to +0,06** per trade — thin, and thin on purpose
- Expected rate: the owner wants **3–5 trades a day**; I expect the cells that
  clear the statistics to come in **under** that, and for the trade count and
  the edge to trade off against each other.
- Conditions where this should **fail**:
  - if the best cells are continuations, the two-book rationale is wrong
  - if the edge lives in one session only and no mechanism explains which
  - if cost share exceeds ~12% of the stop, which M1 on a cross may well do
  - if early and late halves disagree in sign

### The pre-registered first reading

Run before the real search, on **synthetic** XAUJPY (XAUUSD × USDJPY, M15,
2012–2022, **no spread**, placeholder cost). This is a test of the machinery,
not a result, and it is written down so the real search can contradict it:

- The twelve best cells were **all fades**, no continuations. Consistent with
  the rationale above.
- Nothing cleared the bar: best `quiet_stretch_fade` +2,79 σ against 3,30.
- **The rate/edge trade-off showed up immediately.** The best cell ran at 0,5
  trades a day. The cells at the owner's 3–5 a day (`streak_reversal` 3,9,
  `stretch_fade` 2,4) had the smallest per-trade edge and **negative** holdouts.

## Test design

- Instrument: XAUJPY. Clocks: M1, M5, M15.
- Grid: 28 mechanisms × 3 clocks × 2 ratios (1,0 and 1,5) = **168 cells**.
- Multiplicity: **Bonferroni, 3,62 σ**. Selecting a session raises it to
  **4,01** and `--hours` is off by default so the table can be read without
  paying for it.
- Significance: **day-clustered** sigma. Hours of one morning are one
  observation.
- Control: a **rate-matched random entry** on the same bars, same stop, same
  target, same costs.
- Costs: the account's own `PositionSizer._cost_share` — commission, slippage
  and spread against the stop width.
- Holdout: the newest **20%**, never searched, and a positive holdout is a
  hard requirement rather than a tie-breaker.
- Hours 07:00–13:00 UTC are **blocked before the search starts**. That is not a
  free parameter: section ten measured −0,101 R per trade in that window on
  these crosses over 1 664 trades against +0,064 everywhere else.

## What the search does not model

Named here because an unnamed omission is an assumed zero. Applied: the cost
model, one position at a time, blocked hours, and a bar spanning both barriers
booked as a loss. **Not** applied: `AWAITING_CONFIRMATION`, `NEWS_BLACKOUT`,
`TARGET_RARELY_REACHED`, `SPREAD_EATS_THE_STOP`, `MARKET_TOO_QUIET`,
`AWAITING_PULLBACK`, `VOLUME_SPIKE`, `ENTRY_OVEREXTENDED`, the four-position
cap, and the lone-module confluence floor. Every one of them only ever removes
trades.

`zoekjpy.cmd` decides what is worth replaying. `sectie11.cmd` — the replay with
the gates on, sharing the position cap with sections 6, 7, 8 and 10 — is what
decides.

## Result

Not run yet. `zoekjpy.cmd 720` fills this in.

- [ ] ≥100 resolved trades in the winning cell
- [ ] Clears 3,62 σ day-clustered
- [ ] Beats its own rate-matched coin flip
- [ ] **Positive untouched holdout**
- [ ] Early and late halves agree in sign
- [ ] Cost share under 12% of the stop

## Verdict

`no mechanism named; all three sections silent and off the live allowlist.`

A section with no mechanism emits no read at all rather than a zero, so this
state cannot be mistaken for a search that ran and found nothing.
