# MT5 trading system

A research-driven autonomous trading system for MetaTrader 5, built in phases.
**Phases 1-2 are complete** (foundation, risk layer, journal); there is no strategy and no order loop yet.

Read `PLAN.md` for the roadmap, the honest account-size arithmetic, and the open
questions. Read `CLAUDE.md` for the conventions this codebase is held to.

## What is built

- Typed, validated configuration where the hard risk rules are **validators**,
  not comments — you cannot enable martingale or a fail-open news filter by
  editing YAML.
- An MT5 connector with reconnect backoff, filling-mode negotiation, and retry
  logic that distinguishes transient rejections from fatal ones.
- Full execution telemetry on every order: requested vs filled price,
  direction-aware slippage in pips, latency, spread at send, raw return code.
- A data layer that never exposes the currently forming bar, and refuses stale,
  gapped or malformed series rather than analysing them.
- A startup guard that computes, per symbol, the widest stop your account can
  actually afford at your configured risk — and says so out loud when the
  answer is "none".
- A position sizer that **skips** trades it cannot size rather than rounding up
  to the broker's minimum lot, and says in plain numbers how far off it was.
- A risk manager with daily/weekly loss limits measured on equity, a drawdown
  circuit breaker that trips the kill switch, anti-martingale risk reduction
  after a losing streak, and assertions that crash on averaging down, hedging,
  or any post-loss risk increase.
- A SQLite journal that records **every** analysis cycle — including the
  overwhelming majority that produce no trade — plus per-module scores, full
  execution telemetry, MAE/MFE in R, and shadow trades for blocked setups.
- Structured JSON logging, a filesystem kill switch, and 208 tests that run
  without a terminal on any platform.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"        # add ".[live]" on Windows for MetaTrader5
cp config/.env.example config/.env       # then fill it in
```

`config/.env` is gitignored. Keep it that way.

## Use

```bash
python main.py --check-config     # validate configuration offline
python main.py --status           # connect, run the startup guard, print the report
python main.py --risk             # risk state: every limit and how close we are
python main.py --data EURUSD      # multi-timeframe summary with ATR
python -m pytest                  # the test suite
```

The Phase 1 acceptance test — one demo order, placed, verified and closed, with
every execution detail printed — is:

```bash
python scripts/phase1_acceptance.py --symbol EURUSD
```

It refuses to run against a live account.

## Kill switch

Create a file named `STOP` in the project root. The system halts. Delete it to
resume. No API, no network, no authentication — it works when nothing else does.

## Platform note

`MetaTrader5` ships Windows-only binaries, so live and paper modes need a
Windows host or VPS. Everything else — configuration, instrument maths, the data
layer, backtesting, and the whole test suite — runs anywhere. The MT5 constants
are mirrored in `core/mt5_codes.py` and verified against the real package on
connect.
