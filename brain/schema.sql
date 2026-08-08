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
-- a risk limit, a threshold, a weight or a lot size. It is evidence and it is
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

CREATE INDEX IF NOT EXISTS decisions_symbol_time  ON decisions (symbol, decided_at DESC);
CREATE INDEX IF NOT EXISTS decisions_reason_time  ON decisions (reason, decided_at DESC);
CREATE INDEX IF NOT EXISTS decisions_taken_time   ON decisions (decided_at DESC) WHERE taken;

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
