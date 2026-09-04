# Section 11 — XAU crosses M5, one fitted model per market

## Read this first

**The untouched holdout is negative in four markets out of four.** That is the
result, and it arrived after the models were written:

| market | holdout trades | holdout R | per trade |
|---|---|---|---|
| XAUAUD | 3 685 | **−193,22 R** | −0,052 |
| XAUEUR | 3 621 | **−189,77 R** | −0,052 |
| XAUGBP | 3 643 | **−148,93 R** | −0,041 |
| XAUJPY | 3 783 | **−115,53 R** | −0,031 |

The walk-forward read **+0,005 to +0,030 R** per trade; the holdout reads
**−0,031 to −0,052**. A sign flip, on nearly fifteen thousand trades, in the
one measurement containing no choices at all — no threshold picked, no market
picked, no penalty argued. **The recommendation is that this section does not
trade real money.** It is on the allowlist to be replayed by
`dryrun-live.cmd 180`, which costs nothing; it should come off before Jarvis
starts, and a positive dry run over a shorter window does not overturn the
table above.

**Nothing in this section cleared the statistical bar either.** The best cell in the
whole search came out at **+2,56 day-clustered sigma against a Bonferroni bar
of 2,96**, and the trainer refused to write a single model file on its own.
The four model files that exist were written with `train11.cmd 720 forceer`
(`--write-anyway --threshold-for-all 0.30`) on the owner's explicit
instruction, with that number in front of him, and each file carries
`cleared_the_bar: false`, the searched sigma and the bar it missed stamped
inside it so the provenance travels with the model instead of living only in
this document.

This module is live to be **measured by `dryrun-live.cmd 180`** before Jarvis
is started, not because it is proven. If that replay does not come out
positive, the entry comes off `live_enabled_modules`.

## Claim

The same mechanism as section six — a small fitted model reading the last
closed M5 bar's feature row and predicting whether a 1,5 R target is reached
before a 1,0 ATR stop — applied to the four XAU crosses (XAUEUR, XAUGBP,
XAUAUD, XAUJPY) instead of XAUUSD, with **an own model trained per market**
rather than one model shared across them.

One model per market is the point. A XAU cross is gold priced through a second
currency; XAUJPY and XAUEUR do not share a session profile, a tick size or a
spread regime, and a shared fit would average four different animals into one
that describes none of them.

## Economic rationale

The counterparty is the same one section six proposes on XAUUSD: participants
reacting to the last bar of a metal move without pricing the state that
preceded it. The crosses add a second, measured effect — the FX leg. Between
16:00 and 18:00 UTC London is gone while gold still trades, so the currency
half of the quote thins out and the spread on the cross widens. That is not a
theory here; it was measured on section ten across the same four crosses:
**16:00 negative in both chronological halves in four crosses out of four**,
while the same hour is positive on XAUUSD. Hours 16 and 17 are therefore
blocked per symbol for all four crosses.

If there is no answer to "who keeps losing money here", the honest reading of
this section is that its mechanism is inherited from section six and its
evidence on these four markets is *weaker than section six's on gold*.

## Predictions, stated in advance

- Expected win rate: **38–43%** at a 1,5 R target (the fit predicts 40,6%)
- Expected average R: **+0,03 R** per trade — thin, and thin on purpose
- Expected signals: roughly **18 per market per day** at the 0,30 threshold
- Conditions where this should fail:
  - a regime where the M5 bar-to-bar structure of a cross is dominated by its
    FX leg rather than by gold (a central-bank morning on the quote currency)
  - any widening of cross spreads beyond the 0,4–0,8% of stop measured here
  - the loss share exceeding 55% over a real sample — 40,6% expected wins
    means about 59% losers is *normal*; 65% over forty trades is more than
    three sigma the wrong way and is the breaker's trigger

## Test design

- Instruments: XAUEUR, XAUGBP, XAUAUD, XAUJPY (M5), Eightcap history
- Procedure: **walk-forward** — the model predicting fold *i* is fitted only on
  bars strictly before fold *i*. A final segment was held back and looked at
  once.
- Significance: **day-clustered** sigma. Four crosses breaking on one morning
  is one observation, not four.
- Multiplicity: Bonferroni over the cells actually searched. **16 cells** (4
  markets × 4 thresholds) → bar 2,96 sigma.
- Control: a **rate-matched random entry** taking the same number of trades on
  the same bars with the same stop and target.
- Costs: commission, slippage and spread charged against the stop width, the
  same cost model the account uses live.

## Result

- **Nothing cleared 2,96 sigma.** Best cell XAUJPY **+2,56**.
- All 16 cells came out **positive**.
- All 16 beat their rate-matched coin flip on the same bars.
- Costs ran **0,4–0,8% of the stop** — small enough that they hide nothing and
  do not explain the sign either way.
- XAUJPY's sigma rose **monotonically with the threshold** (0,82 → 0,86 → 1,67
  → 2,56), which is the shape a real edge has and noise usually does not. It is
  suggestive. It is not significance.

Sixteen positive cells is a weaker statement than it sounds: the cells share
bars, share gold as a common factor, and are therefore nothing like sixteen
independent coin flips. That is exactly why the bar is 2,96 and why it was not
lowered to fit the result.

## What bounds it live

- Own section breaker: window 40, minimum 40 trades, stop above **65% losses**
  or **6 consecutive losses**. At ~18 signals per market per day this judges
  within days, not months.
- Hours 16 and 17 blocked on all four crosses.
- Lone-module floor 0,55 (it is its own thesis; the generic 0,65 lone gate
  would refuse every valid signal).
- Exempt from the entry-timing gate for the same reason section six is: a model
  that reads the last closed bar would have its own thesis vetoed by a gate
  that refuses entries running against the last three M5 bars.
- The hard account, news, cost and broker checks are **not** relaxed.
- No model file for a market ⇒ that market is silent. The startup guard refuses
  to start at all if section eleven is live with no models, so "took zero
  trades" can never be mistaken for "found nothing".

## Verdict

`weight > 0, live for a measured forward test only, on the owner's instruction
and against both the search result and the holdout.`

Off the allowlist if `dryrun-live.cmd 180` does not come out positive — and the
standing recommendation is that it comes off regardless, because the untouched
holdout is negative in four markets out of four.
