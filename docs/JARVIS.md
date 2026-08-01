# Jarvis operator guide

## Modes

- `MONITOR`: rotates through the complete supported MT5 catalogue, ranks live
  candidates, performs deep multi-timeframe analysis and journals decisions.
  It never sends an order.
- `PAPER`: the same service plus persistent simulated positions filled from
  live Eightcap bid/ask quotes. Paper balance and positions survive restarts.
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
