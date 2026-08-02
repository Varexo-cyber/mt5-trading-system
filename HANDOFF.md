# Handoff prompt — MT5 autonomous trading system

> Paste this whole file as the opening prompt of a new session, with the
> repository `Varexo-cyber/mt5-trading-system` attached. It is written for an
> AI assistant picking the project up cold.

---

## 0. Your role and the ground rules

You are continuing a long-running, research-driven trading system in Python
that trades through MetaTrader 5. Phases 1–3 are finished, tested and pushed.
Your job is Phase 4 onward.

Before writing any code: **read the repository.** Specifically `CLAUDE.md`
(conventions), `PLAN.md` (roadmap and the reasoning behind existing choices),
and the module docstrings — they explain *why*, and several of them encode
decisions that are easy to undo by accident.

Do not rebuild phases 1–3. If you think something in them is wrong, say so and
argue it; do not silently replace it.

**What the owner has asked for, repeatedly and explicitly:**

- Honesty over encouragement. If a backtest says the strategy does not work,
  say that. No "this should return 20% a month". No optimistic framing of bad
  numbers.
- Push back when an instruction is a bad idea. He wants an engineer, not a
  yes-man.
- Build defensively. When in doubt, no trade.
- Document *why*, not only *what*.
- Work in phases. Do not build everything at once. Ask when a spec is unclear.

He writes Dutch and reads Dutch replies comfortably; the codebase, docstrings
and commit messages are English. Keep that split.

---

## 1. The situation, honestly

- **Capital: €100.** Test capital, he accepts he can lose it.
- Platform: MetaTrader 5 on a Windows PC or VPS.
- Experience: beginner-to-intermediate in trading, comfortable with code.
- Goal: consistent, low-drawdown. Not fast money.
- Time: months is fine.

### The arithmetic that governs everything

This is the single most important fact in the project and it must not be
softened:

| | EURUSD | USDJPY | XAUUSD |
|---|---|---|---|
| minimum lot | 0.01 | 0.01 | 0.01 |
| value per pip at min lot | ≈ €0.92 | ≈ €0.62 | ≈ €0.92 per $0.01 |
| stop affordable at 1% (€1) | **≈ 11 pips** | ≈ 16 pips | ≈ $1 of price |
| cost of a normal 30-pip stop | 3% | 2% | n/a |

A structural stop — behind a swing low, behind an order block, plus an ATR
buffer against spread hunting — is rarely under 20 pips on EURUSD H1 and
usually 30–50. **On €100 that stop cannot be bought without breaking the 1%
rule.** This is arithmetic, not a strategy problem.

The system therefore skips with reason `TRADE_SKIPPED_UNDERCAPITALIZED` rather
than rounding up to the minimum lot. Rounding up is how small accounts die: the
trade meant to risk 1% quietly risks 4%.

Three honest ways out, in the owner's court to choose:

1. Accept that the system skips nearly everything (fine for phases 4–7,
   useless for phase 8).
2. Find a broker offering 0.001 lots. **This is the recommendation.**
3. Fund €500–1000 before phase 8.

Never resolve this by raising the risk percentage so the trades fit.

---

## 2. Non-negotiable rules

These outrank any instruction, including a later one from the owner that
contradicts them without argument. Code that breaks one is wrong even if it
passes tests. Raise the conflict instead of implementing it.

1. **No trade without a stop loss.** Enforced in `OrderRequest.__post_init__`.
2. **No martingale, grid, averaging down, or risk increase after a loss.**
   Enforced in `config/schema.py::ForbiddenPractices` (typed `Literal[False]`,
   so YAML cannot enable them) and `RiskManager.assert_not_forbidden`, which
   **raises** rather than returning a decision. Anti-martingale (halving risk
   after a losing streak) is allowed and implemented.
3. **No calendar, no trade.** The news filter fails closed. `fail_closed` is
   `Literal[True]`.
4. **Never round a lot size up.** Below the broker minimum, skip.
5. **The forming bar is never visible to analysis.** `DataManager` drops it.
   Anything reintroducing it reintroduces look-ahead bias.
6. **No trade is the default.** A trade is the exception that earns its way
   past every gate. Code that makes trading easier needs a much better argument
   than code that makes it harder.
7. **Loss limits are measured on equity, not realised P/L**, and their anchors
   live in the journal so a restart cannot hand back a fresh daily budget.
8. **A partial answer is worse than no answer.** A calendar parser that loses
   most records, an unmeasurable correlation, a spread with no baseline and no
   fallback — all block rather than pass. "Unknown" is never "fine".

---

## 3. What exists (phases 1–3, done)

282 tests, all green, all runnable on any platform without a terminal.
`ruff` and `black` clean, line length 100.

```
mt5-trading-system/
├── config/
│   ├── config.yaml          every parameter; no magic numbers in code
│   ├── schema.py            pydantic; the hard rules live here as validators
│   ├── loader.py            YAML + overlay + TS_ env vars, readable errors
│   └── .env.example         credentials template (.env is gitignored)
├── core/
│   ├── types.py             Signal, MarketContext, OrderRequest/Result, Position
│   ├── errors.py            exception hierarchy
│   ├── clock.py             LiveClock / SimulatedClock
│   ├── mt5_codes.py         MT5 constants mirrored so it imports on Linux
│   ├── instrument.py        pip maths, lot rounding, risk feasibility
│   ├── mt5_connector.py     connection, reconnect, execution, retries
│   ├── data_manager.py      OHLCV, multi-timeframe, caching, data quality
│   └── startup.py           startup guard + per-symbol feasibility report
├── filters/
│   ├── base.py              Filter protocol, FilterVerdict, FilterChain
│   ├── news_filter.py       MANDATORY, fail-closed
│   ├── calendar/            events, providers (2 remote + file), service
│   ├── session_filter.py    London/NY/Asia, rollover, weekend edges
│   ├── spread_filter.py     self-learning baseline per instrument per hour
│   └── correlation_filter.py  rolling correlation, direction-aware
├── risk/
│   ├── reasons.py           CLOSED vocabulary of reasons (goes into the journal)
│   ├── position_sizer.py    lot sizing + the undercapitalized check
│   └── risk_manager.py      limits, circuit breaker, forbidden-practice asserts
├── journal/
│   ├── database.py          SQLite schema (v2), migrations, risk queries
│   └── recorder.py          writes cycles, trades, executions, shadow trades
├── infra/
│   ├── logging.py           JSON to file, human-readable to console
│   └── killswitch.py        the STOP file
├── scripts/
│   ├── phase1_acceptance.py demo-order test (run on Windows)
│   └── verify_calendar.py   verify the calendar feeds — SEE §5
├── tests/                   282 tests, FakeMT5-driven
└── main.py                  --check-config / --status / --risk / --filters / --data
```

### Conventions you must keep

- **Config, never constants.** Any number the system acts on lives in
  `config.yaml` and is typed in `schema.py`. `extra="forbid"` is set, so a
  typo'd key is a startup failure.
- **Hard rules are validated, not documented.** Encode a rule as a pydantic
  validator or a `__post_init__` check, with a test proving it fires. Pattern:
  `tests/test_config.py::TestHardRules`.
- **Ask the clock, never the wall.** Nothing calls `datetime.now()` outside
  `core/clock.py`. All times tz-aware UTC. Broker server time is a separate,
  explicitly converted concept.
- **Fail loudly, never degrade quietly.** The one acceptable fallback is
  "do not trade".
- **Immutable domain objects.** `Signal`, `MarketContext`, `OrderRequest` are
  `frozen=True`.
- **MT5 is Windows-only; the codebase is not.** `MetaTrader5` is imported
  lazily in `core/mt5_connector.py::import_mt5`. Constants mirrored in
  `core/mt5_codes.py` and verified against the real package at connect time.
  **Never import `MetaTrader5` at module scope.**
- **Test through `FakeMT5`** (`tests/fakes/fake_mt5.py`). It scripts return
  codes, fill offsets and connection failures.
- **`risk/reasons.py` is a closed vocabulary.** Those strings land verbatim in
  the journal and get grouped by in reports. Renaming a member breaks
  historical queries — add a new one instead.
- **Docstrings explain why.** Comments restating the line below are noise.
- Log with `extra={"event": "...", ...}` so the JSON log stays queryable.

### Commands

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # add ".[live]" on Windows
.venv/bin/python -m pytest
.venv/bin/python -m ruff check . && .venv/bin/python -m black --check .
python main.py --check-config     # offline
python main.py --status           # needs a terminal
python main.py --risk
python main.py --filters EURUSD
```

---

## 4. Remaining work

### Phase 4 — Analysis modules (the big one)

Build **one module at a time**, each with a visualisation so the owner can
confirm it sees what he sees on a chart. Interface is the `AnalysisModule`
protocol in `core/types.py`:

```python
class AnalysisModule(Protocol):
    name: str
    def analyze(self, ctx: MarketContext) -> Signal: ...
```

`Signal` carries score (−100..+100), confidence (0..1), reasoning, key_levels,
invalidation_price. **Return `Signal.neutral()` when the module has no read** —
"no opinion" and "slightly bullish" are different statements.

Modules receive a `MarketContext` and nothing else: no connector, no clock, no
filesystem. That constraint is what makes them replayable and testable.

**Recommended order** (this is a deliberate narrowing of the owner's original
much longer list — see §6):

1. `analysis/market_structure.py` — swing points (fractal, configurable
   lookback), BOS vs CHoCH, internal vs external structure, equal highs/lows.
   **This produces stop placement**, so it comes first.
2. `analysis/levels.py` — S/R with touch count, age, and whether it flipped;
   daily/weekly/monthly opens; round numbers.
3. `analysis/smc.py` — liquidity sweeps and order blocks, with a quality score
   (untouched, has an FVG, swept liquidity beforehand). FVG and premium/discount
   zones. Note: many SMC concepts are not independent of each other — measure
   the correlation between module signals and correct for double counting.
4. `analysis/volatility.py` — ATR regime, ADX, trending vs ranging. Not a
   signal; the switch that enables the right setup set.
5. `analysis/indicators.py` — EMA stack, RSI divergence, MACD, Bollinger
   squeeze. **Confirmation only, never a trigger.**
6. `analysis/patterns.py` — candlestick patterns scored only in the context of
   a level, never standalone.

Lower priority, weight 0 unless a backtest proves otherwise: Wyckoff schematics,
Elliott Wave, volume profile, COT, seasonality, sentiment. They may be built
for research; they may not carry weight without evidence.

**Per module, in this order — the order matters:**

1. Write `docs/hypotheses/<name>.md` **before** backtesting. Template exists.
   State what you expect, why, and **who is losing money on the other side**.
   No mechanism, no module. Inventing the explanation after seeing the pattern
   is the definition of data mining.
2. Implement against the protocol.
3. Visualise (matplotlib is fine) so the owner can eyeball it.
4. Backtest out-of-sample. Document in `docs/modules/<name>.md`.
5. Weight stays 0 until it clears §7's validation protocol.

### Phase 5 — Confluence engine + backtester

`strategy/confluence_engine.py`:
- Weighted sum of module scores; weights from config.
- Trade only when **all** of: |weighted score| > threshold (start ~65);
  **and** at least N independent *categories* agree (structure + level +
  timing), not N modules — three SMC modules agreeing is one opinion;
  **and** structural R:R ≥ 1:2; **and** every filter green.
- Build in explicitly that no-trade is the default and the system should be
  doing nothing most of the time.

`backtest/engine.py` — **pessimistic, not flattering**:
- Real historical spreads, or 1.5× average as a conservative estimate.
- Slippage on entry and exit, higher during news and low liquidity.
- Commission and swap.
- **No look-ahead, and test for it explicitly.** The `SimulatedClock` and the
  forming-bar rule exist for this. Write a test that fails if a module can see
  a bar that has not closed.
- Intrabar ambiguity: if high and low both touch SL and TP in one candle,
  assume **SL first**.
- Report: total return, profit factor, expectancy in R, win rate, max drawdown
  (% and duration), Sharpe, longest losing streak, and a Monte Carlo
  (shuffle trade order 1000×, report the probability of a 20% drawdown).

`backtest/walkforward.py` — optimise on window A, test on B, roll forward.
Results good only on the optimisation window means overfit. Throw it away.

`backtest/monte_carlo.py`.

### Phase 6 — Trade management + live loop

`risk/trade_manager.py`:
- Stop always **structural** (behind swing/order block) with an ATR buffer.
  Never a fixed pip number.
- TP to the next structural level, not a fixed number.
- Break-even at +1R (config: `break_even_at_r`), offset by
  `break_even_offset_atr` to cover spread and commission.
- Partial close 50% at +1.5R, remainder trailed.
- Trailing: ATR or structure (follow swing lows in an uptrend).
- Time exit: flat after X hours with no momentum → close. Dead capital is
  capital at risk.
- News exit: `NewsFilter.position_action` already returns `none` /
  `break_even` / `close`. Wire it in.

`main.py` live loop:
- Kill switch checked at the top of every iteration **and** immediately before
  every order (`pre_send_guard` is already wired).
- **Reconciliation every cycle**: MT5 positions must match what the system
  believes. On mismatch → alert and stop.
- Graceful shutdown on Ctrl+C (handlers exist).
- Journal the cycle **before** sending an order.

### Phase 7 — Monitoring, reporting, postmortem

- `monitoring/alerts.py` — Telegram or Discord. Needed for the circuit breaker.
- `monitoring/healthcheck.py` — heartbeat, connection state.
- `journal/reporter.py` — weekly HTML/markdown: equity curve, win rate, profit
  factor, expectancy in R, max drawdown, average R, traded vs skipped setups,
  top 3 improvements.
- `learning/postmortem.py` — weekly analysis of losers. Common features?
  Report like: "setups scoring 65–70 win 31%, setups above 80 win 58% →
  consider raising the threshold."
- MFE analysis (taking profit too early?) and MAE analysis (stopped out just
  before being right?). Both columns already exist in the journal.
- `sample_size_check` — every report states how many more trades are needed
  before a conclusion means anything.

### Phase 8 — Micro-live shakeout (€100, 1–2 weeks)

Goal is **validating execution, not judging the strategy**.

Build an `EXECUTION_REPORT.md` generator from the `order_attempts` table:
- Orders sent / filled / rejected, with every return code.
- Slippage: mean, median, worst — separately for entry, SL hit and TP hit.
- Difference between backtest assumption and reality for spread and slippage.
  **Feed this back into the backtester.**
- Did the actual lot size exactly match the calculation? Any deviation is a bug.
- Did the news filter block correctly? Log every blocked moment and verify by
  hand against the calendar.
- Were SL and TP placed where the analysis wanted, or did the broker move them
  (stop level / freeze level)?
- Uptime, disconnects, reconnect duration, were positions unmanaged during it?

**Success criterion is explicitly NOT P&L.** It is zero unexplained
discrepancies between what the system thought it did and what MT5 did. With
~10 trades the P&L is statistically meaningless — a 55%-win-rate system loses
5 in a row 2.5% of the time. Report the P&L, but always with its confidence
interval, so nobody steers on noise.

### Phase 9 — Evaluation and scaling

Only after 100+ live trades are win rate and expectancy usable.

### The learning system (three layers, increasing caution)

Do **not** build a neural network that adjusts its own weights live. That is
the fastest route to a system that optimises on noise and falls over at the
worst moment. Instead:

**Layer 1 — fully automatic, safe (runs continuously).** Context-dependent, not
performance-dependent, so it cannot overfit to recent outcomes:
- Volatility adaptation: SL/TP/trailing scale with current ATR vs its history.
- Regime detection: trending vs ranging enables the matching setup set.
- Spread adaptation — **already built** in `filters/spread_filter.py`.
- Session adaptation: learn per instrument when it actually moves.

**Layer 2 — proposals, the owner approves (weekly).**
- Bayesian per setup type: Beta posterior on win rate, weak prior, report the
  90% interval. 3 wins from 4 is not a 75% setup and the interval shows it.
- Weight proposals: **hard limit ±15% shift per quarter, minimum 100 trades of
  evidence.** No weight goes to 0 or to dominance on one bad month.
- Threshold optimisation: performance per score bucket (60-70, 70-80, 80+).
- Filter effectiveness: what happened after the setups that were blocked?
  Shadow trades are already recorded (`shadow_trades` table) — resolve them.
- Every proposal states the data, the confidence interval, the number of trades
  behind it, and what the effect would have been historically.

**Layer 3 — quarterly, full revalidation.** Walk-forward reoptimisation, add or
retire modules, full backtest over all history, only after §7's checks.

**Guardrails on all layers:**
- `learning/changelog.md` — every parameter change with date, old value, new
  value, rationale, and performance 30 days before/after.
- Rollback: every config version stored and restorable with one command.
- A/B shadow running: a new config runs 30 days generating signals without
  trading, compared against live, before switching.
- **Anti-recency assertion**: never change a parameter on fewer than 30 trades
  or 30 days.
- Meta-monitoring: continuously compare the current config against the original
  frozen one. If the learning has added nothing after a year, that is a result
  and must not be buried.

---

## 5. Immediate TODO before phase 8

**Verify the calendar providers against live feeds.** The remote parsers in
`filters/calendar/providers.py` were written against each feed's documented
shape but were never run against a real response — the environment they were
written in blocks outbound HTTPS to third-party hosts (`403 on CONNECT`).

```bash
python scripts/verify_calendar.py --raw
```

What to check:
- Both providers reachable.
- Neither reporting **zero high-impact events for a normal week** — that is the
  signature of a parser that silently lost the impact field.
- The two roughly agreeing on the high-impact count.
- Blackout windows for a known release (this week's NFP or CPI) starting and
  ending where expected.

Also: run `verify_calendar.py --archive` weekly from a scheduled task. The free
feeds publish only the current and next week, so the archive is the only way to
get a calendar the backtester can use over a multi-year window.

---

## 5b. Changes made after the ChatGPT build (audit round)

- **AI model IDs corrected.** `claude-sonnet-4-6` and `gpt-5.6-terra` do not
  exist. With `ai.fail_closed: true` every trade would have been vetoed by an
  API error and the system would have run forever without trading. Now
  `claude-sonnet-5` / `gpt-5.1`.
- **`apply_experimental_live_limits` no longer auto-promotes modules to live.**
  It used to set `live_enabled_modules` to every module with weight > 0, which
  meant arming the experiment silently put unvalidated analysis on real money.
  Promotion is now an explicit line in `config/eightcap.yaml`.
- **`PromotionAudit` now runs for `EXPERIMENTAL_LIVE` too** — reporting, not
  blocking. Blocking would make the experiment impossible (the audit needs 100
  OOS trades that cannot exist before trading). Every failing check is logged
  and alerted at startup, so the missing evidence is visible rather than
  silently skipped. `LIVE` still requires every check to pass.
- **`instruments.universe_mode`** added: `whitelist` (default, unchanged) or
  `affordable`, where any broker symbol may be considered and the position
  sizer decides what the account can express. The equity floors and the
  blocklist still apply in both modes.
- **`scanner.batch_size` / `scanner.deep_candidates`** moved from a hardcoded
  5 into config; the Eightcap overlay uses 20.
- **Trade-count ceilings raised** to 6/day, 20/week. The daily *loss* limit is
  the binding constraint (4% at 1% risk = four losers); the count is a backstop
  against a signal-generation bug, not the throttle on trading.
- **Duplicate `instruments:` key in `config/eightcap.yaml` fixed** — the second
  block was silently discarding `symbol_suffix: ".i"`, so every live symbol
  name would have been wrong. A test now loads the overlay and asserts on it.

### The Anthropic request shape (found by running `verify_ai_advisor.py`)

`scripts/verify_ai_advisor.py` returned `BadRequestError:http_400`. Three
separate problems, all with the same signature — because the adviser is
fail-closed, none of them crashes anything. Each one turns into a permanent
veto on every candidate, so the system runs all day, scans the whole
catalogue, and never trades, with no error anywhere.

1. **`temperature=0` is rejected.** Claude Sonnet 5 and the Opus 4.7+ family
   return HTTP 400 for any non-default `temperature`, `top_p` or `top_k`.
   Removed from both `review` and `reflect`. `temperature=0` never guaranteed
   identical outputs on the older models either, so nothing is lost.
2. **`max_tokens` was too small for a model that thinks.** From Sonnet 5
   onward, omitting `thinking` runs *adaptive* thinking, and thinking tokens
   count against `max_tokens`. At 600 the reply truncates before the JSON,
   which arrives as `stop_reason="max_tokens"` — another silent veto. Now
   4000 with `output_config.effort: "medium"`, which is what actually bounds
   the spend.
3. **Thinking blocks were being concatenated into the JSON parser.** The text
   extraction selected on `hasattr(block, "text")`; it now selects on
   `block.type == "text"`.

Two diagnostics changed so the next failure of this kind is not silent:
`_safe_error` now appends Claude's own message for 400 and 404 (the statuses
that name the offending field or model) and nothing for 401/403/429/5xx, whose
bodies can carry organisation detail; and a non-`end_turn` stop reason is now
reported as `incomplete_response:<reason>` instead of a bare
`incomplete_response`. `ai.timeout_seconds` went 30 → 60, since a timeout is
also a veto and a thinking model is slower than a plain completion.

`tests/test_advisory_reporting.py` now asserts the request shape directly
rather than relying on a live call: no sampling parameters, no `budget_tokens`,
`max_tokens` ≥ 2000, and the thinking block excluded from the parsed verdict.

### First run against the live broker (2026-08-02)

Three verification scripts were run against the real Eightcap terminal for the
first time. The AI gate came back `READY`. The rest produced findings.

**The calendar is single-source right now.** `faireconomy` fails on
`ff_calendar_nextweek.json`; `tradingview` returns a full week (241 events, 14
high impact) and the service falls through to it, so the news filter works.
That is the fallback design behaving correctly, but the redundancy the design
asks for is gone until faireconomy is fixed. Changes made: the wrapped error now
carries the underlying reason (a 429 needs backing off, a 404 means the feed
moved, a timeout means the network — all three previously printed "failed"); the
two weekly files are spaced apart and retried twice, since fetching one CDN
origin back-to-back is the request pattern most likely to be rate limited; and
the bespoke User-Agent was replaced with a browser one. **Retrying does not
soften fail-closed** — a real outage still raises and still stops trading.

**`main.py --status` crashed instead of reporting.** The mode was `backtest`
while the terminal was on a live account, so `enforce()` raised and no
diagnostics ran. A read-only tool that refuses to run precisely when the
configuration is wrong is useless at the one moment it is needed. It now prints
the report, runs the diagnostics, and returns exit code 1. Note for the record:
the assertion in the old comment that `jarvis.py` called `enforce()` was wrong —
it never did. It has its own hard asserts for the money-losing cases
(`_assert_account_mode`, the arming file, the experimental contract, the AI
gate), which is why this is a reporting change and not a safety hole.

**A whitelisted symbol below its equity floor no longer blocks startup.**
XAUUSD at EUR 100 was a blocking error. But the equity floors exist so an
account can grow into instruments it cannot afford yet, so this is the expected
state of a small account, not a misconfiguration — and the floor already
prevents the trade. It is now a warning. If the floors rule out *everything*,
"no tradable symbol survived" still blocks, which is the case that matters.

**The autonomous path had no feasibility report.** `JarvisRunner.connect` now
logs the same startup report and alerts if nothing is expressible. The number
that governs this account — 1% risk buys ~11.5 pips of stop at the minimum lot,
measured, not estimated — was previously visible only if the operator happened
to run `main.py --status` first. A run that skips every setup for a legitimate
reason looked identical to one that was broken.

**The EUR 100 arithmetic, now measured rather than predicted:**

| symbol | pip value @ 0.01 lot | widest stop at 1% risk |
|---|---|---|
| EURUSD.i | EUR 0.087 | 11.5 pips |
| GBPUSD.i | EUR 0.087 | 11.5 pips |
| USDJPY.i | EUR 0.055 | 18.2 pips |
| AUDUSD.i | EUR 0.087 | 11.5 pips |
| USDCAD.i | EUR 0.062 | 16.2 pips |

Every structural stop wider than that is skipped as
`TRADE_SKIPPED_UNDERCAPITALIZED`. This is the constraint described in §1, now
confirmed against the actual broker. Nothing in the code can fix it.

## 6. Known issues and design debt

**`max_sl_pips` is an FX-shaped rule.** A "pip" on gold is one point ($0.01),
so a 30-pip ceiling there would mean a 30-cent stop. The ceiling is currently
applied to FX only (`spec.is_forex`); elsewhere the money-based
undercapitalised check is the binding guard, which is exact rather than a
proxy. When gold becomes genuinely tradable in phase 5, express this in ATR
multiples instead of pips.

**The knowledge base in the original brief is far too large for one system.**
Sections A–O add up to 60+ modules. At 100 trades minimum per module and 2–3
trades a week, that is decades of validation, and overfitting stops being a
risk and becomes a certainty. The narrowing in §4 (five to six core modules) is
deliberate and was accepted. Build the rest as research if you like; do not
give them weight without evidence.

**Three of the four live-enabled modules have no hypothesis document.**
`trend_momentum`, `liquidity_sweep` and `level_reaction` carry weight and are
promoted to live, but `docs/hypotheses/` only contains `market_structure.md`.
That breaks the pre-registration rule. `market_structure.md` itself is honest —
it states weight 0 and "not measured" — while `config.yaml` gives it 1.0. Write
the three missing hypotheses and reconcile the documented weight with the
configured one.

**The AI veto makes backtest and live diverge.** The adviser can only veto,
never bypass a gate, which is the right shape. But the backtester does not know
about it, so any edge measured offline is not the edge live will produce.
Consider running the veto in shadow (logged, not acted on) until
`runtime/ai_reviews.jsonl` shows whether it adds value.

**Free calendar feeds only cover this week and next.** Backtesting the news
filter over history needs the archive built by `verify_calendar.py --archive`,
which only starts accumulating from the day it first runs. Start it early.

---

## 7. Validation protocol (protection against overfitting)

With this many modules, overfitting is the primary threat. Mandatory:

1. **Pre-registration.** `docs/hypotheses/<name>.md` before the backtest.
2. **Data split.** Oldest 60% train/optimise, next 20% validation, final 20%
   **untouched** until the very end. Look at it once. Used twice, it is no
   longer out-of-sample.
3. **Deflated Sharpe.** Correct for the number of variants tested. Report how
   many configurations were tried — that number sets the threshold.
4. **Parameter stability.** An edge at EMA 21 that vanishes at 20 and 22 is
   noise. Plot performance across the whole range and require a broad plateau,
   not a spike.
5. **Cross-instrument.** Does it work on pairs it was not developed on?
6. **Minimum sample.** No conclusion under 100 trades per module.
7. **Economic rationale.** Why does this work — who is on the other side?
   Without that story it is data mining.

---

## 8. Open questions for the owner

Ask these early; they change what phases 6 and 8 look like. They do not block
phase 4.

1. **Which broker and account?** Exact symbol names (`EURUSD`, `EURUSD.pro`,
   `EURUSDm`?), hedging or netting, and whether they offer 0.001 lots.
   `python main.py --status` answers all of it in one go.
2. **The account-size decision from §1.** Stay at €100 and accept phase 8 will
   mostly produce skips, move to a broker with finer lots, or fund up. This is
   the most consequential open question in the project.
3. **PC 24/5 or a VPS?** A PC that sleeps overnight means the system can find
   positions it did not manage, which is different reconciliation logic.
4. **Telegram or Discord** for alerts? Needed for the circuit breaker.
5. **Netting or hedging account?** On a netting account "two simultaneous
   positions" works fundamentally differently — they net — which directly
   affects the risk manager.

---

## 9. What to do first in your session

1. Read `CLAUDE.md`, `PLAN.md`, and the docstrings of `core/instrument.py`,
   `risk/position_sizer.py`, `risk/risk_manager.py`, `filters/news_filter.py`.
2. Run the test suite. 282 should pass.
3. Ask the owner the §8 questions that matter to you, then start Phase 4 with
   `docs/hypotheses/market_structure.md` — the hypothesis before the code.
4. One module per delivery, with a visualisation. Do not batch them.
