-- The account's long-term memory.
--
-- `runtime/trading_memory.json` was the whole brain until now: forty lessons,
-- two hundred symbols, a retention window that throws the rest away, and one
-- file on one VPS. It works, and it is the wrong shape for the question the
-- operator is actually asking, which is "what has this account learned in its
-- entire life, and can it still see it".
--
-- Four things this gives that a JSON file cannot:
--
--   1. It survives the machine. A rebuilt VPS starts with everything it knew.
--   2. It is queryable. "Every EURUSD short taken inside a news spike, and how
--      they ended" is a WHERE clause instead of a rewrite.
--   3. It is complete. Every decision is written, including the ones that were
--      refused -- and the refusals are where most of the evidence is, because
--      there are two thousand of them for every trade.
--   4. It does not forget. Aggregates stay honest over months.
--
-- DESIGN RULE, AND IT IS THE IMPORTANT ONE: nothing in this database may move
-- a risk limit, a threshold, a weight or a lot size. Realised trades may only
-- add a small, configured ordering modifier after the minimum sample; they
-- cannot make a rejected setup eligible. Everything else is evidence and
-- context for a prompt. The same rule `learning/memory.py` already states, for
-- the same reason -- a learning system that can rewrite its own risk controls
-- is how an account dies. Config changes by an edit, visible in a diff.

CREATE TABLE IF NOT EXISTS decisions (
    id              BIGSERIAL PRIMARY KEY,
    -- Idempotency. The runner may retry a write after a network blip, and a
    -- duplicated decision would double-count every aggregate built on top.
    fingerprint     TEXT        NOT NULL UNIQUE,
    decided_at      TIMESTAMPTZ NOT NULL,
    account         TEXT        NOT NULL,
    mode            TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    direction       TEXT,
    -- OK when it became a trade; otherwise the gate that stopped it. This one
    -- column is what makes "what is this system refusing, and was it right"
    -- answerable at all.
    reason          TEXT        NOT NULL,
    detail          TEXT        NOT NULL DEFAULT '',
    taken           BOOLEAN     NOT NULL DEFAULT FALSE,
    equity          NUMERIC(14, 2),
    conviction      NUMERIC(6, 2),
    playbook        TEXT,
    asset_class     TEXT,
    regime          TEXT,
    session         TEXT,
    horizon         TEXT,
    planning_timeframe TEXT,
    entry           NUMERIC(18, 8),
    stop_loss       NUMERIC(18, 8),
    take_profit     NUMERIC(18, 8),
    -- Everything the gates measured, as they measured it: spread, session,
    -- correlation, cost share, news pressure. JSONB rather than columns
    -- because the set of filters changes and a migration per filter would
    -- mean the ones added in a hurry never get recorded at all.
    filters         JSONB       NOT NULL DEFAULT '{}'::JSONB,
    -- What the reviewer said, and what it cost to ask.
    ai_verdict      TEXT,
    ai_confidence   NUMERIC(5, 3),
    ai_reasoning    TEXT,
    ai_tokens       INTEGER,
    -- The wire copy that existed at the moment of the decision. Written at
    -- decision time on purpose: reconstructing it afterwards is impossible,
    -- because a feed only carries the last few hours.
    headlines       JSONB       NOT NULL DEFAULT '[]'::JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Existing Neon databases predate the typed market context above. CREATE TABLE
-- does not add columns, so migrations remain idempotent and explicit here.
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS asset_class TEXT;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS regime TEXT;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS session TEXT;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS horizon TEXT;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS planning_timeframe TEXT;
-- WHICH detector actually found this setup, and how sure it was.
--
-- The largest hole in the whole record. `conviction` stored the blended score
-- and nothing stored what produced it, so the question that decides whether
-- this system can ever get better -- does `liquidity_sweep` make money while
-- `trend_momentum` loses it -- was not answerable from any table here.
--
-- Attribution is the difference between learning and merely accumulating.
-- Without it every trade is one undifferentiated data point and the only
-- lesson available is "we are down", which is not a lesson: it names nothing
-- to stop doing. With it, sixty-four trades become sixty-four votes on each
-- detector, and the ones that lose money can be found and switched off.
--
-- One row per decision holding every module that scored: name, signed score,
-- confidence, and whatever the module recorded about itself. JSONB for the
-- same reason `filters` is -- the set of modules changes, and a migration per
-- detector means the ones added in a hurry never get recorded at all.
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS signals JSONB NOT NULL DEFAULT '[]'::JSONB;

CREATE INDEX IF NOT EXISTS decisions_symbol_time  ON decisions (symbol, decided_at DESC);
CREATE INDEX IF NOT EXISTS decisions_reason_time  ON decisions (reason, decided_at DESC);
CREATE INDEX IF NOT EXISTS decisions_taken_time   ON decisions (decided_at DESC) WHERE taken;
CREATE INDEX IF NOT EXISTS decisions_segment_time ON decisions (
    asset_class, playbook, horizon, direction, regime, decided_at DESC
);

-- Passive outcomes for executable plans that a gate or Claude refused. They
-- grade the refusal; they never count as real trades and never calibrate live
-- selection. Keeping that distinction in the schema prevents a later report
-- from quietly mixing hypothetical fills with broker-confirmed positions.
CREATE TABLE IF NOT EXISTS counterfactuals (
    id              BIGSERIAL PRIMARY KEY,
    fingerprint     TEXT        NOT NULL UNIQUE,
    account         TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    direction       TEXT        NOT NULL,
    blocked_by      TEXT        NOT NULL,
    opened_at       TIMESTAMPTZ NOT NULL,
    entry           NUMERIC(18, 8) NOT NULL,
    stop_loss       NUMERIC(18, 8) NOT NULL,
    take_profit     NUMERIC(18, 8) NOT NULL,
    resolved_at     TIMESTAMPTZ NOT NULL,
    outcome         TEXT        NOT NULL,
    pnl_r           NUMERIC(10, 4) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS counterfactuals_gate_time
    ON counterfactuals (blocked_by, resolved_at DESC);
CREATE INDEX IF NOT EXISTS counterfactuals_symbol_time
    ON counterfactuals (symbol, resolved_at DESC);

-- One row per position that actually opened, closed out when it ends.
CREATE TABLE IF NOT EXISTS trades (
    id              BIGSERIAL PRIMARY KEY,
    account         TEXT        NOT NULL,
    ticket          BIGINT      NOT NULL,
    -- The decision that produced it, so the whole chain -- what the gates saw,
    -- what the news was, what Claude said, what happened -- is one join.
    decision_id     BIGINT      REFERENCES decisions (id) ON DELETE SET NULL,
    symbol          TEXT        NOT NULL,
    direction       TEXT        NOT NULL,
    volume          NUMERIC(10, 2) NOT NULL,
    opened_at       TIMESTAMPTZ NOT NULL,
    entry           NUMERIC(18, 8) NOT NULL,
    stop_loss       NUMERIC(18, 8) NOT NULL,
    take_profit     NUMERIC(18, 8),
    -- The denominator of every R this trade will ever report. Recorded because
    -- the lot rounds down to the broker step, so it is not 2% of equity and
    -- is not the same from trade to trade.
    risk_money      NUMERIC(14, 4) NOT NULL,
    closed_at       TIMESTAMPTZ,
    exit_price      NUMERIC(18, 8),
    exit_reason     TEXT,
    pnl_money       NUMERIC(14, 4),
    pnl_r           NUMERIC(10, 4),
    -- Best and worst it ever reached, ratcheted by the guard. `kept` is
    -- pnl_r / mfe_r and is the number the operator actually cares about.
    mfe_r           NUMERIC(10, 4),
    mae_r           NUMERIC(10, 4),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (account, ticket, opened_at)
);

CREATE INDEX IF NOT EXISTS trades_symbol_closed ON trades (symbol, closed_at DESC);
CREATE INDEX IF NOT EXISTS trades_open          ON trades (account) WHERE closed_at IS NULL;

-- Everything the guard did to a position while it was open, in order. This is
-- the "what happened, when, and why" the operator asked for: BREAK_EVEN at
-- 13:42 on +0.31R, PROFIT_BANKED at 13:58 because the move stopped running.
CREATE TABLE IF NOT EXISTS trade_events (
    id              BIGSERIAL PRIMARY KEY,
    trade_id        BIGINT      NOT NULL REFERENCES trades (id) ON DELETE CASCADE,
    happened_at     TIMESTAMPTZ NOT NULL,
    action          TEXT        NOT NULL,
    reason          TEXT        NOT NULL DEFAULT '',
    r_at_action     NUMERIC(10, 4),
    price           NUMERIC(18, 8),
    money           NUMERIC(14, 4)
);

CREATE INDEX IF NOT EXISTS trade_events_trade ON trade_events (trade_id, happened_at);

-- What was concluded after a trade ended, one row per lesson rather than one
-- blob per reflection. One row per lesson is what makes "this lesson has now
-- arrived from nine separate trades" a GROUP BY instead of a text search, and
-- a lesson's evidence count is the only thing separating a pattern from an
-- anecdote.
CREATE TABLE IF NOT EXISTS lessons (
    id              BIGSERIAL PRIMARY KEY,
    trade_id        BIGINT      REFERENCES trades (id) ON DELETE SET NULL,
    learned_at      TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL DEFAULT '',
    direction       TEXT        NOT NULL DEFAULT '',
    -- Normalised for grouping: lowercased, punctuation and digits stripped.
    -- The same observation phrased two ways is one lesson with two sightings,
    -- and counting it as two is how a memory talks itself into a conviction.
    lesson_key      TEXT        NOT NULL,
    lesson          TEXT        NOT NULL,
    pnl_r           NUMERIC(10, 4),
    source          TEXT        NOT NULL DEFAULT 'reflection'
);

CREATE INDEX IF NOT EXISTS lessons_key  ON lessons (lesson_key);
CREATE INDEX IF NOT EXISTS lessons_time ON lessons (learned_at DESC);

-- Wire copy, kept beyond the feed's own few hours so that "what was being
-- written when this trade was opened" stays answerable a year later.
CREATE TABLE IF NOT EXISTS headlines (
    id              BIGSERIAL PRIMARY KEY,
    -- The same de-duplication key the in-memory window uses, so a story
    -- carried by four wires is one row here too.
    fingerprint     TEXT        NOT NULL UNIQUE,
    published_at    TIMESTAMPTZ NOT NULL,
    source          TEXT        NOT NULL,
    title           TEXT        NOT NULL,
    link            TEXT        NOT NULL DEFAULT '',
    currencies      TEXT[]      NOT NULL DEFAULT '{}',
    systemic        BOOLEAN     NOT NULL DEFAULT FALSE,
    seen_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS headlines_published  ON headlines (published_at DESC);
CREATE INDEX IF NOT EXISTS headlines_currencies ON headlines USING GIN (currencies);

-- A closed trade beside the decision that produced it. Used by the briefing
-- and by the scorecard, and defined here so both read the same join rather
-- than each writing their own and drifting apart.
CREATE OR REPLACE VIEW trade_history AS
SELECT
    t.id,
    t.account,
    t.symbol,
    t.direction,
    t.opened_at,
    t.closed_at,
    t.exit_reason,
    t.pnl_r,
    t.pnl_money,
    t.mfe_r,
    CASE WHEN t.mfe_r > 0 THEN t.pnl_r / t.mfe_r END AS kept,
    d.reason        AS decision_reason,
    d.conviction,
    d.playbook,
    d.ai_verdict,
    d.ai_confidence,
    d.filters,
    d.headlines
FROM trades t
LEFT JOIN decisions d ON d.id = t.decision_id
WHERE t.closed_at IS NOT NULL;

-- What the reviewer said about a position that was already open, and what the
-- position went on to do. The loop this closes is the one the system could not
-- see: every other table records the account grading its own rules, and this
-- one records it grading its own adviser.
--
-- Held apart from `decisions` on purpose. A decision is asked once, before
-- anything exists; a supervision is asked repeatedly of a position that is
-- already running, and folding the two together would make "how often was the
-- reviewer right" a query nobody could write correctly.
CREATE TABLE IF NOT EXISTS supervisions (
    id              BIGSERIAL PRIMARY KEY,
    trade_id        BIGINT      REFERENCES trades (id) ON DELETE CASCADE,
    account         TEXT        NOT NULL,
    asked_at        TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    -- hold, tighten_stop, pull_target_in, partial_close, close.
    action          TEXT        NOT NULL,
    confidence      NUMERIC(5, 3),
    reasoning       TEXT,
    -- Where the trade stood when the question was asked, so the answer can be
    -- judged against what was knowable rather than against the ending.
    r_at_the_time   NUMERIC(10, 4),
    -- Whether the manager carried it out. A verdict the risk layer refused is
    -- still evidence about the adviser, and counting it as acted-upon would
    -- credit or blame it for something that never happened.
    applied         BOOLEAN     NOT NULL DEFAULT FALSE,
    latency_ms      INTEGER,
    model           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS supervisions_trade ON supervisions (trade_id, asked_at);
CREATE INDEX IF NOT EXISTS supervisions_action ON supervisions (action, asked_at DESC);

-- Every verdict beside what the trade finally did. "The reviewer said hold at
-- +0.4R and the trade ended at -1R" is one row here and an unwritable join
-- without it.
CREATE OR REPLACE VIEW supervision_outcomes AS
SELECT
    s.id,
    s.account,
    s.symbol,
    s.asked_at,
    s.action,
    s.confidence,
    s.r_at_the_time,
    s.applied,
    t.pnl_r        AS trade_ended_at_r,
    t.exit_reason,
    t.closed_at
FROM supervisions s
JOIN trades t ON t.id = s.trade_id
WHERE t.closed_at IS NOT NULL;

-- The whole life of a position, sampled at guard cadence.
--
-- Everything else in this file records the moments a decision was taken:
-- opened, banked, closed, reviewed. None of it records what the trade was
-- actually DOING in between, so every question about management has been
-- answered from its endpoints. "Should the stop have gone to entry sooner"
-- and "how long did it sit at its high before it gave up" are questions about
-- the path, and the path was never kept.
--
-- The live case that forced it: a CADCHF long showing EUR 2.82 on a EUR 130
-- account — over two percent of everything — with the broker stop still twelve
-- pips below entry. Whether holding that was right is answerable only against
-- what the price did next, second by second, and nothing had written it down.
--
-- One row per position per guard pass. A trade open three hours produces a few
-- thousand small rows; that is what Postgres is for, and `path_step` is
-- deliberately narrow so it stays cheap.
CREATE TABLE IF NOT EXISTS position_path (
    id              BIGSERIAL PRIMARY KEY,
    trade_id        BIGINT      NOT NULL REFERENCES trades (id) ON DELETE CASCADE,
    sampled_at      TIMESTAMPTZ NOT NULL,
    price           NUMERIC(18, 8) NOT NULL,
    r_now           NUMERIC(10, 4) NOT NULL,
    peak_r          NUMERIC(10, 4) NOT NULL,
    money           NUMERIC(14, 4) NOT NULL,
    -- Where the broker stop sat at this instant. The one field that turns the
    -- path into a management record rather than a price series: it says how
    -- much of the money on the table was actually safe.
    stop_price      NUMERIC(18, 8),
    stop_r          NUMERIC(10, 4),
    protected       BOOLEAN     NOT NULL DEFAULT FALSE,
    health          TEXT        NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS position_path_trade ON position_path (trade_id, sampled_at);

-- What stepping in was worth, one row per closed trade.
--
-- Every closed trade is replayed against its own untouched stop and target
-- until one of them is reached, and the difference is the whole question:
-- did the rule that closed this trade beat leaving it alone?
--
-- The comparison already existed and lived only in local SQLite, where the
-- part of the system that decides hold-versus-close every second could not
-- read it. So the judgement layer has been making that call on this account
-- for weeks with no idea what its own interventions have earned — while the
-- answer sat in a file on the VPS. `Brain.management_records` reads it back
-- into the briefing.
--
-- `exit_action` is the rule that actually closed it: AI_CLOSE, PEAK_STALL,
-- PROFIT_BANKED, BROKER_TP, BROKER_SL. Grouped by that, `lift_r` says which
-- interventions pay and which are expensive habits.
CREATE TABLE IF NOT EXISTS management_outcomes (
    id              BIGSERIAL PRIMARY KEY,
    account         TEXT        NOT NULL,
    trade_id        BIGINT      REFERENCES trades (id) ON DELETE SET NULL,
    local_trade_id  BIGINT      NOT NULL,
    resolved_at     TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL DEFAULT '',
    direction       TEXT        NOT NULL DEFAULT '',
    exit_action     TEXT        NOT NULL DEFAULT '',
    -- What the untouched original plan would have paid.
    baseline_pnl_r  NUMERIC(10, 4) NOT NULL,
    -- What the trade actually took home.
    actual_pnl_r    NUMERIC(10, 4) NOT NULL,
    -- Positive means intervening beat holding. This is the number.
    lift_r          NUMERIC(10, 4) NOT NULL,
    UNIQUE (account, local_trade_id)
);

CREATE INDEX IF NOT EXISTS management_outcomes_action
    ON management_outcomes (account, exit_action);
