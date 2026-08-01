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
`STOP` file; the service then exits and cannot restart until a human clears it.

## AI review

AI is optional and disabled by default. It is a second-opinion veto, not an
unbounded order tool. Enable only after installing `.[ai]` and setting secrets
in `config/.env`, never in YAML or Git:

```text
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

Also set an Anthropic model available to your own API account. Consensus mode
requires both providers to approve. A timeout, malformed response, missing key
or disagreement blocks the setup.

## Windows autostart

Run `install_autostart_monitor.cmd` once. It registers a per-user task named
`JarvisTradingMonitor` that starts monitor mode when you log in. It does not
configure live mode. MT5 still needs to be running and logged in.

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
