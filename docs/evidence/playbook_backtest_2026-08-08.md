# Playbook backtest evidence — 2026-08-08

## Status

**Negative evidence. The tested entry approach is not a proven edge.**

The owner supplied the console output of `backtest.cmd` on 2026-08-08. The
command replayed 90 days of real MT5 history for EURUSD.i, GBPUSD.i, USDJPY.i
and AUDUSD.i, one closed M5 bar at a time. It also shuffled trade directions at
the same moments, symbols, stops and targets as a coin-flip control.

The command and its source file are not present in this linked repository, so
the result is recorded as operator-supplied evidence and has not yet been
reproduced from this checkout. That limits provenance; it does not justify
ignoring a strongly negative result.

## Reported result

| theory | proposals | trades | won | net | R/trade | worst DD |
|---|---:|---:|---:|---:|---:|---:|
| range_fade | 1,721 | 464 | 29% | -114.11R | -0.246R | 56.30R |
| momentum_scalp | 292 | 220 | 33% | -103.78R | -0.472R | 37.10R |
| failed_break | 783 | 218 | 26% | -65.66R | -0.301R | 37.18R |
| range_break | 638 | 231 | 25% | -62.33R | -0.270R | 25.84R |
| trend_pullback | 434 | 119 | 24% | -55.42R | -0.466R | 23.54R |

The reported matched coin-flip comparison concluded that none of the five
theories beat guessing outside chance. Costs included spread and slippage; the
output says commission was not included and estimates subtracting roughly
0.03R–0.11R per trade would make the result worse.

## What this proves—and what it does not

- It is evidence against forcing more trades from these entry patterns.
- It is evidence that Claude cannot rescue a bad upstream candidate generator
  merely by writing a stronger explanation.
- It does not validate the exact production path until the script is added to
  this repository and the run is reproduced with immutable data/config hashes.
- It does not identify a profitable replacement strategy.
- It does not justify tuning the same 90-day sample until it becomes green.

## Required next research step

Change one falsifiable idea at a time, pre-register it, and compare it with the
same matched-time coin baseline on train/validation data. Keep the final 20%
untouched. No live weight or entry gate should be promoted because it makes
this already-seen sample look better.

## Production response

The supplied negative result is now enforced in configuration and code:

- the playbooks still run and remain visible as research context;
- in Experimental Live they may not create an order or veto the independently
  routed swing/confluence candidate;
- paper and backtest operation can continue collecting their outcomes;
- restoring live execution authority requires new, uniquely identified
  out-of-sample evidence, not a rerun or tuning pass over this 90-day sample.

This quarantine does not disable the production swing/confluence route and it
does not alter position risk. Its purpose is to prevent known-negative activity
from being mistaken for useful trade frequency.
