import sqlite3

from dashboard.ai_exchange import execution_outcomes, pair_ai_reviews


def test_ai_exchange_pairs_exact_request_and_response() -> None:
    rows = [
        {
            "timestamp": "2026-08-02T13:00:00+00:00",
            "event": "pretrade_request",
            "cycle_id": "cycle-1",
            "symbol": "EURUSD.i",
            "direction": "LONG",
            "request": {"score": 72, "executable_proposal": {"risk_pct": 1.0}},
        },
        {
            "timestamp": "2026-08-02T13:00:01+00:00",
            "event": "pretrade_response",
            "cycle_id": "cycle-1",
            "symbol": "EURUSD.i",
            "direction": "LONG",
            "latency_ms": 842.3,
            "decision": {
                "approved": False,
                "confidence": 0.81,
                "thesis": "Event risk is too close",
                "risks": ["event risk"],
                "error": "",
            },
        },
    ]

    exchanges = pair_ai_reviews(rows)

    assert len(exchanges) == 1
    assert exchanges[0]["status"] == "VETO"
    assert exchanges[0]["request"]["score"] == 72
    assert exchanges[0]["decision"]["confidence"] == 0.81
    assert exchanges[0]["latency_ms"] == 842.3


def test_ai_exchange_shows_request_as_pending_before_response() -> None:
    exchanges = pair_ai_reviews(
        [
            {
                "timestamp": "2026-08-02T13:00:00+00:00",
                "event": "pretrade_request",
                "cycle_id": "cycle-pending",
                "symbol": "BTCUSD",
                "direction": "SHORT",
                "request": {"score": 80},
            }
        ]
    )

    assert exchanges[0]["status"] == "PENDING"
    assert exchanges[0]["decision"] == {}


def test_ai_exchange_marks_provider_failure_as_fail_closed() -> None:
    exchanges = pair_ai_reviews(
        [
            {
                "timestamp": "2026-08-02T13:00:00+00:00",
                "event": "pretrade_response",
                "cycle_id": "cycle-error",
                "symbol": "GBPUSD.i",
                "direction": "LONG",
                "decision": {
                    "approved": False,
                    "confidence": 0.0,
                    "thesis": "AI unavailable; trade vetoed",
                    "error": "TimeoutError",
                },
            }
        ]
    )

    assert exchanges[0]["status"] == "ERROR / FAIL CLOSED"


def _outcome_database(path) -> sqlite3.Connection:  # type: ignore[no-untyped-def]
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE analysis_cycles (
            id INTEGER PRIMARY KEY, cycle_id TEXT, mode TEXT, decision TEXT,
            reason TEXT, detail TEXT
        );
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY, cycle_pk INTEGER, ticket INTEGER,
            entry_state TEXT, exit_reason TEXT
        );
        """)
    return connection


def test_execution_outcome_explains_a_post_ai_block(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "journal.db"
    connection = _outcome_database(path)
    connection.execute(
        "INSERT INTO analysis_cycles VALUES (1, 'cycle-1', 'live', 'SKIP', ?, ?)",
        ("ENTRY_MOVED_DURING_REVIEW", "price moved 0.8 ATR during Claude review"),
    )
    connection.commit()
    connection.close()

    outcome = execution_outcomes(path, ["cycle-1"])["cycle-1"]

    assert outcome["status"] == "NA CLAUDE GEBLOKKEERD"
    assert "ENTRY_MOVED_DURING_REVIEW" in outcome["detail"]


def test_execution_outcome_shows_the_confirmed_mt5_ticket(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "journal.db"
    connection = _outcome_database(path)
    connection.execute(
        "INSERT INTO analysis_cycles VALUES (1, 'cycle-1', 'live', 'TRADE', 'OK', '')"
    )
    connection.execute("INSERT INTO trades VALUES (1, 1, 5049535, 'OPEN', NULL)")
    connection.commit()
    connection.close()

    outcome = execution_outcomes(path, ["cycle-1"])["cycle-1"]

    assert outcome["status"] == "GEOPEND IN MT5"
    assert outcome["ticket"] == 5049535
