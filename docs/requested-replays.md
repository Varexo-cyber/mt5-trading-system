# VPS measurements

`us30ruw.cmd 180` runs the existing impulse_retest and order_block_fast together
on US30 M1, then independently on US30 M5. These are the detectors used in
the earlier measurement, with their existing configuration. No new strategy
sections or live permissions are added. Results are in runtime/us30-m1-180.csv
and runtime/us30-m5-180.csv.

NOT THE SAME MEASUREMENT AS `us30.cmd`. That launcher drives the four new
experimental sections (section_us30_impulse_m1/m5, section_us30_orderblock_m1/m5)
through scripts.dry_run_sections. This one drives the existing detectors with
no section wrapper: no US30 symbol gate, no one-entry-per-setup rule, no
per-section clock restriction. Two different US30 numbers. Do not compare them
to each other.

`ndx100be.cmd 180` runs section_five_ndx100_m5 on its configured M5 clock twice:
fixed SL/TP, then SL/TP with the configured break-even rule. Files are
runtime/ndx100-s5-fixed-180.csv and runtime/ndx100-s5-break-even-180.csv.
Use 90 or 30 instead of 180 for shorter measurements.

Both launchers use scripts.dry_run_sections and --jarvis-replay, the engine
behind hoeveel.cmd. MT5 must be running and logged in on the VPS. Each
scenario has its own chronological position book; other markets/sections
are not included. The account replay's historical gates, sizing and cost
model apply. News, historical AI decisions and exact tick fills retain the
limitations printed by hoeveel.cmd.

The S5 comparison changes only in-memory replay settings. It deliberately
compares fixed exits with BE only, without partial/trailing/profit-lock.
The configured H1-ATR offset uses the last fully closed H1 bar. Its section
availability and portfolio exit timestamp follow the exit selected for that
scenario. The existing general replay path is preserved. Consequently the
explicit fixed comparison can differ from old hoeveel output, where section
availability could follow a managed exit even for a fixed-exit family.

The two runs are independent strategy replays, not a paired-trade experiment:
BE can free the symbol sooner and permit different subsequent entries.
Printed 'live' column headings in the shared report denote the exit selected
for the scenario, not permission to place real orders.

Offline developer smoke checks are supported by:
python -m scripts.replay_requested_markets s5 2 --database PATH --equity 258.90
Replace s5 with us30 for the other launcher. These short checks only validate
execution, not profitability. No YAML or live trading modules are modified.
