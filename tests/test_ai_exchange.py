from dashboard.ai_exchange import pair_ai_reviews


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
