# Jarvis operator guide

## Modes

- `MONITOR`: rotates through the complete supported MT5 catalogue, ranks live
  candidates, performs deep multi-timeframe analysis and journals decisions.
  It never sends an order.
- `PAPER`: the same service plus persistent simulated positions filled from
  live Eightcap bid/ask quotes. Paper balance and positions survive restarts.
- `DEMO`: sends real MT5 orders through the full execution and reconciliation
  path, but hard-refuses to start unless MT5 reports a demo account.
- `LIVE`: the real MT5 execution adapter. It starts only when the config is
  micro-live, the account matches `runtime/LIVE_ARMED.json`, and at least two
  analysis modules are explicitly listed as independently validated.
- `EXPERIMENTAL_LIVE`: a separate, explicitly accepted real-money experiment.
  It is bound to one login/server/currency contract, fixes both requested and
  maximum risk at 1% per trade, uses the 15% peak drawdown breaker, and also
  stops at 85% of the equity recorded when the contract was armed. It does not
  pretend the current research modules have passed normal live promotion.

Opening MT5 alone does not start Jarvis. Start it through the dashboard or one
of the launchers. MT5 must remain open and logged in.

## Recommended first run

1. Double-click `launch_dashboard.cmd`.
2. Open the Control tab and click **Start MONITOR**.
3. Let it rotate through the catalogue for at least a week.
4. Stop it, clear STOP by typing the confirmation, then start **PAPER**.
5. Leave paper running for thirty days and inspect `runtime/reports`.
6. Log MT5 into a demo account and run **DEMO** to validate broker execution.

Install `install_calendar_archive_task.cmd` once so the economic-calendar
history grows every Sunday. Free feeds cannot reconstruct past years later.

The service writes `runtime/heartbeat.json`. `runtime/jarvis.pid` exists only
while the service is running. The dashboard STOP button writes the durable
`STOP` file. Jarvis stops new entries, closes positions carrying its own magic
number, and exits only when those positions are flat. A rejected closure is
retried; manual positions are never touched.

The dashboard's **Live scanner** tab refreshes every five seconds from
`runtime/scan_activity.json`. It shows the rotating catalogue coverage, every
recent cheap inspection, the exact rejection stage and the final deep-analysis
decision. The ledger retains the latest status per symbol and the most recent
500 inspection records; it is bounded so it cannot grow forever.

With the default settings Jarvis inspects 25 symbols every roughly 30 seconds
and deep-analyses at most five. A catalogue of 847 symbols therefore takes
about 17 minutes per full rotation. Scanning all 847 symbols every second is
intentionally unsupported: it would overload the terminal, multiply stale and
partial reads, and does not create better signals.

## AI review

AI is optional in the base research configuration. The Eightcap overlay makes
Claude mandatory for `EXPERIMENTAL_LIVE`: it is a final second-opinion veto,
not an unbounded order tool. The deterministic engine must first clear account,
news, session, spread, correlation, sizing, margin, stoploss and reward:risk
rules. Only then is a compact snapshot of the last three closed bars on each
configured timeframe sent to Claude. Claude can approve or veto; it cannot
change size, stop, target, risk, filters or account settings.

Install `.[ai]` and set the selected provider secret in `config/.env`, never in
YAML, Git, a screenshot, or chat:

```text
ANTHROPIC_API_KEY=...
# OPENAI_API_KEY=...  # unused in the current Claude-only Eightcap setup
```

The Eightcap overlay currently pins `claude-sonnet-4-6`. Verify authentication
without placing a trade:

```powershell
.venv-live\Scripts\python.exe scripts\verify_ai_advisor.py
```

A timeout, rate limit, authentication error, malformed response, missing key,
unfinished response, confidence below 0.65 or failed audit write blocks the
setup. Every proposal and decision is stored in
`runtime/ai_reviews.jsonl` and shown in the dashboard. A closed trade is sent
back once for a process reflection; that reflection is research-only and can
never alter production parameters automatically. API usage is billed by the
provider separately from a Claude or ChatGPT chat subscription.

## Windows autostart

Run `install_autostart_monitor.cmd` once. It registers a per-user task named
`JarvisTradingMonitor` that starts monitor mode when you log in. It does not
configure live mode. MT5 still needs to be running and logged in.

For an evidence-collection workstation, run `install_autostart_paper.cmd`
instead. It registers `JarvisMetaTrader5`, starts the saved MT5 terminal
minimized at login, starts `JarvisTradingPaper` after a 60-second delay, and
opens `JarvisDashboard` after 90 seconds. The single-instance dashboard
launcher reopens the existing local app instead of starting duplicates. The
installer disables `JarvisTradingMonitor` so only one runner can start. This
changes only automatic PAPER operation; LIVE remains evidence-locked.

After an experimental contract has been created, run
`install_autostart_experimental_live.cmd` to replace PAPER autostart with the
account-bound real-money runner. It registers `JarvisTradingExperimentalLive`,
disables `JarvisTradingPaper`, and keeps the MT5 and dashboard startup tasks.
If a durable STOP exists at login, the runner stays safe and exits. In the
dashboard type `clear stop` and choose **CLEAR STOP + START REAL TRADING** to
clear it and start the account-bound experimental runner in one explicit step.

## Live promotion

There is deliberately no one-click live button. Live requires all of:

1. 100+ out-of-sample trades per enabled module.
2. Parameter stability and cross-instrument evidence.
3. Thirty days clean paper operation.
4. A demo execution run with zero unexplained reconciliation differences.
5. Explicit module names in `analysis.confluence.live_enabled_modules`.
6. An account-specific arming file created after the preceding evidence exists.

Until then the code can scan, reason and paper trade automatically without
being able to spend the live balance.

Inspect the gates at any time:

```powershell
.venv-live\Scripts\python.exe scripts\audit_live_promotion.py
```

The command reports failures and exits without arming. Only after every gate
passes can `--arm` bind approval to the currently connected live account.

The owner-authorized `EXPERIMENTAL_LIVE` path is deliberately separate from
this normal promotion. It scans the supported broker catalogue but can place
entries only in the micro-live whitelist (`EURUSD.i`, `GBPUSD.i`, `USDJPY.i`,
`AUDUSD.i`). The analysis is deterministic confluence across D1/H4/H1/M15/M5;
in the current Eightcap configuration Claude is a mandatory fail-closed veto
immediately before an executable proposal can become an order. No win rate or
profit is guaranteed.

## Historical validation

`validate_strategy.py` fetches long MT5 ranges in safe chunks, archives them
locally, recreates D1/H4/H1/M15/M5 context using closed bars only, and runs the
pre-registered 3x3 market-structure sweep. The last 20% is not evaluated unless
`--unlock-holdout` is supplied, and a durable ledger refuses a second look.

```powershell
.venv-live\Scripts\python.exe scripts\validate_strategy.py `
  --symbol EURUSD.i --symbol GBPUSD.i --symbol USDJPY.i `
  --symbol AUDUSD.i --symbol USDCAD.i `
  --start 2015-01-01 --end 2026-01-01 --configurations-tested 9
```

## Guarded learning

Weekly reports may propose weight changes but never apply them. Proposals are
capped at +/-15%, need 100 trades spanning 30 days, and then run as a no-order
shadow configuration for another 30 days. Use `scripts/config_control.py` for
snapshots, starting a shadow, explicit promotion and rollback. Every approved
change must also be recorded in `learning/changelog.md`.
