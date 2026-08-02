# MT5 trading system

A research-driven autonomous trading system for MetaTrader 5. The scanner,
multi-timeframe analysis, paper/demo execution, position management, monitoring
and reporting loops are implemented. Live execution remains evidence-locked;
the current research sample does not justify risking money.

**Continuing this project in a new session? Start with `HANDOFF.md`.**

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
- A mandatory news filter that **fails closed**: two independent calendar
  providers, an expiring disk cache, and a hard block when none of them can
  answer. No calendar means no trade.
- Session, spread and correlation filters. The spread filter learns its own
  per-instrument, per-hour baseline from observation; the correlation filter is
  direction-aware, so long EURUSD against short GBPUSD passes while long
  against long does not.
- Whole-catalogue rotation, asset-aware profiles, bounded optional AI vetoes,
  persistent paper execution and a hard-gated MT5 demo mode.
- Exact deal reconciliation, broker-calculated margin, partial-close crash
  recovery, news de-risking and evidence-based live promotion.
- Look-ahead-safe historical replay, registered parameter sweeps, deflated
  Sharpe and Monte Carlo drawdown diagnostics.
- Daily, weekly and execution reports plus guarded learning proposals,
  30-day shadow configuration and config rollback.
- Structured JSON logging, a filesystem kill switch, and 300+ tests that run
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
python main.py --filters EURUSD   # each filter's verdict for one symbol
python main.py --data EURUSD      # multi-timeframe summary with ATR

# Eightcap-Live (FX symbols use `.i`; XAUUSD does not)
python main.py --overlay config/eightcap.yaml --status
python -m pytest                  # the test suite

# Explain every live gate; cannot arm until all evidence passes
python scripts/audit_live_promotion.py
```

## Local dashboard

On Windows, install the dashboard extra once and then double-click
`launch_dashboard.cmd`:

```powershell
.venv-live\Scripts\python.exe -m pip install -e ".[live,dashboard]"
```

The app runs on `127.0.0.1` only. It reads the connected MT5 account, all broker
catalogue instruments, active positions and up to four of MT5's 21 timeframes
at once. It also generates a PDF report and exposes the real filesystem-backed
`STOP` switch. Live execution remains locked until the strategy, backtester and
demo loop in phases 4–6 are validated.

Eightcap-specific symbol names live in `config/eightcap.yaml`; FX uses `.i`,
while BTCUSD, XAUUSD, stocks and indices keep their catalogue names.

## Autonomous Jarvis service

```powershell
# One bounded, read-only whole-market batch
.venv-live\Scripts\python.exe jarvis.py --operation monitor --once

# Continuous read-only scanner
.venv-live\Scripts\python.exe jarvis.py --operation monitor

# Persistent paper trading on live Eightcap quotes
.venv-live\Scripts\python.exe jarvis.py --operation paper

# Real MT5 execution, accepted only when MT5 reports a demo account
.venv-live\Scripts\python.exe jarvis.py --operation demo
```

The scanner rotates through the complete supported broker catalogue rather
than hammering all symbols simultaneously. It ranks each batch cheaply,
deep-reads the best candidates across D1/H4/H1/M15/M5, applies confluence,
filters, risk sizing and journalling, and then advances its persistent cursor.
See `docs/JARVIS.md` for controls, AI consensus and Windows autostart. Use
`install_autostart_paper.cmd` to start the saved MT5 terminal minimized at
login, start PAPER sixty seconds later, and open the local dashboard after
ninety seconds; it disables monitor autostart and never arms LIVE.

The Phase 1 acceptance test — one demo order, placed, verified and closed, with
every execution detail printed — is:

```bash
python scripts/phase1_acceptance.py --symbol EURUSD
```

It refuses to run against a live account.

## Before going live: verify the calendar

Both calendar response shapes were verified against live feeds on 2026-08-01.
Continue archiving weekly so multi-year backtests do not depend on short-lived
free-feed history:

```bash
python scripts/verify_calendar.py --raw
```

Zero high-impact events in a normal week is the signature of a parser that lost
the impact field. Treat the remote providers as unverified until this has
passed.

## Kill switch

Create a file named `STOP` in the project root. The system halts. Delete it to
resume. No API, no network, no authentication — it works when nothing else does.

## Platform note

`MetaTrader5` ships Windows-only binaries, so live and paper modes need a
Windows host or VPS. Everything else — configuration, instrument maths, the data
layer, backtesting, and the whole test suite — runs anywhere. The MT5 constants
are mirrored in `core/mt5_codes.py` and verified against the real package on
connect.
