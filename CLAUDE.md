# Project conventions — MT5 trading system

Read this before touching anything in this repository.

Current state: **Phase 1 complete.** `PLAN.md` has the roadmap, the open
questions, and the reasoning behind the design choices below.

---

## The rules that outrank everything

These are not style preferences. Code that breaks one of them is wrong even if
it passes tests and the user asked for it — raise the conflict instead of
implementing it.

1. **No trade without a stop loss.** Enforced in `OrderRequest.__post_init__`.
2. **No martingale, grid, averaging down, or risk increase after a loss.**
   Enforced in `config/schema.py::ForbiddenPractices`. Anti-martingale
   (halving risk after a losing streak) is allowed and configured.
3. **No calendar, no trade.** The news filter fails closed. `fail_closed` is
   typed `Literal[True]`; it cannot be turned off in YAML.
4. **Never round a lot size up.** If the computed size is below the broker's
   minimum, skip the trade with `TRADE_SKIPPED_UNDERCAPITALIZED`. Rounding up
   silently multiplies the risk the sizer computed.
5. **The forming bar is never visible to analysis.** `DataManager` drops it.
   Anything that reintroduces it reintroduces look-ahead bias.
6. **No trade is the default.** A trade is the exception that has to earn its
   way past every filter. Code that makes trading easier needs a much better
   argument than code that makes it harder.

## Design conventions

**Config, never constants.** Every number the system acts on lives in
`config/config.yaml` and is typed in `config/schema.py`. A literal threshold in
a strategy or risk module is a bug. `extra="forbid"` is set, so a typo'd key is
a startup failure rather than a silently ignored setting.

**Hard rules are validated, not documented.** If a rule matters, encode it as a
pydantic validator or a `__post_init__` check with a test that proves it fires.
`tests/test_config.py::TestHardRules` is the pattern.

**Ask the clock, never the wall.** Nothing calls `datetime.now()` outside
`core/clock.py`. All times are tz-aware UTC. Broker server time is a separate,
explicitly converted concept (`LiveClock.server_offset`) — conflating the two
shifts every session boundary by hours.

**Fail loudly, never degrade quietly.** Missing data, a stale feed, a rejected
order, a spec that will not parse: raise. The one acceptable fallback is
"do not trade". Returning a default and continuing is how a system trades on
assumptions nobody checked.

**Immutable domain objects.** `Signal`, `MarketContext`, `OrderRequest` and
friends are `frozen=True`. A module that can mutate the context makes the
journal unreliable: what was logged is no longer what was decided on.

**MT5 is Windows-only, the codebase is not.** The `MetaTrader5` package is
imported lazily in `core/mt5_connector.py::import_mt5`. Constants are mirrored
in `core/mt5_codes.py` and verified against the real package at connect time.
Never import `MetaTrader5` at module scope — it breaks the test suite and every
non-Windows contributor.

**Test through `FakeMT5`.** `tests/fakes/fake_mt5.py` scripts return codes,
fill offsets and connection failures. The failure paths are the ones that
matter, and they are the ones you cannot trigger on demand against a broker.

## Layout

```
config/    typed configuration; schema.py is where hard rules live
core/      types, errors, clock, instrument maths, connector, data, startup guard
infra/     logging (JSON), kill switch
tests/     pytest; runs on any platform, no terminal required
scripts/   operator tools (phase1_acceptance.py)
docs/      modules/<name>.md per analysis module, hypotheses/ pre-registrations
```

Planned, not yet built: `filters/`, `analysis/`, `strategy/`, `risk/`,
`backtest/`, `journal/`, `learning/`, `monitoring/`. See `PLAN.md`.

## Style

- Python 3.11+, `from __future__ import annotations` everywhere.
- Type hints on every function. `black` and `ruff` (line length 100) must pass.
- Docstrings explain **why**, not what. The what is in the code; the why is the
  thing that gets lost. Comments that restate the line below are noise.
- Log with `extra={"event": "...", ...}` so the JSON log stays queryable. The
  `event` key is what you grep for.
- Secrets come from `config/.env` via `config/loader.py` only. They never enter
  the `Settings` tree, a log line, or the journal.

## Before adding an analysis module (Phase 4+)

1. Write the hypothesis in `docs/hypotheses/<name>.md` **first**: what you
   expect, why, and who is on the losing side of the trade. Inventing the
   explanation after finding the pattern is not evidence.
2. Implement against the `AnalysisModule` protocol in `core/types.py`. Return
   `Signal.neutral()` when the module has no read — "no opinion" and "slightly
   bullish" are different statements and the confluence engine treats them
   differently.
3. Backtest out-of-sample. Document the measured edge in
   `docs/modules/<name>.md`.
4. Weight stays 0 until the module clears the validation protocol in `PLAN.md`.
   Modules that never clear it stay in the tree at weight 0 — deleting them
   loses the record that they were tried.

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python -m ruff check . && .venv/bin/python -m black --check .
python main.py --check-config          # offline validation
python main.py --status                # needs a terminal
```
