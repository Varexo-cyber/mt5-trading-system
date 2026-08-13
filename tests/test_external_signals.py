from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from core.types import Direction
from external_signals import ExternalSignalInbox, NotificationEnvelope, RioSignalParser
from external_signals.gold_follow import build_gold_follow_plan
from external_signals.models import ExternalSignalKind
from external_signals.server import SignalReceiver


def envelope(text: str, *, title: str = "Gold Intraday Signals") -> NotificationEnvelope:
    return NotificationEnvelope(
        event_id="event-1",
        received_at=datetime.now(UTC),
        app="Rio Traders",
        title=title,
        text=text,
        source="rio",
    )


def test_parses_complete_gold_zone_signal() -> None:
    event = RioSignalParser().parse(
        envelope("NEW SIGNAL: GOLD/XAUUSD\nSELL: 4385 - 4387\nSL: 4393\nTP1: 4378\nTP2: 4365")
    )

    assert event.kind is ExternalSignalKind.NEW
    assert event.symbol_alias == "XAUUSD"
    assert event.direction is Direction.SHORT
    assert event.entry_low == 4385
    assert event.entry_high == 4387
    assert event.stop_loss == 4393
    assert event.take_profits == (4378, 4365)
    assert event.complete_entry


def test_preliminary_gold_direction_is_a_follow_candidate() -> None:
    event = RioSignalParser().parse(envelope("GOLD\nSELL 4385"))

    assert event.kind is ExternalSignalKind.NEW
    assert not event.complete_entry
    assert "stop loss" in event.reason


def test_incomplete_non_gold_direction_remains_invalid() -> None:
    event = RioSignalParser().parse(envelope("EURUSD BUY 1.1000", title="Rio Traders"))

    assert event.kind is ExternalSignalKind.INVALID
    assert not event.complete_entry


def gold_bars() -> pd.DataFrame:
    closes = [4382.0 + index * 0.05 for index in range(40)]
    return pd.DataFrame(
        {
            "open": [value - 0.1 for value in closes],
            "high": [value + 0.4 for value in closes],
            "low": [value - 0.4 for value in closes],
            "close": closes,
        }
    )


def test_gold_follow_builds_missing_short_protection_from_market_structure() -> None:
    event = RioSignalParser().parse(envelope("GOLD SELL 4385"))

    plan = build_gold_follow_plan(
        event,
        live_entry=4385.5,
        bars=gold_bars(),
        spread=0.2,
        minimum_stop_distance=0.1,
        atr_period=14,
        structure_bars=24,
        stop_atr_multiple=1.5,
        structure_buffer_atr=0.25,
        target_reward_risk=1.2,
        max_entry_deviation_atr=0.75,
        max_entry_deviation_bps=8.0,
    )

    assert plan.used_fallback_stop
    assert plan.used_fallback_target
    assert plan.stop_loss > 4385.5
    assert plan.take_profit < 4385.5


def test_gold_follow_refuses_a_signal_after_price_has_run_too_far() -> None:
    event = RioSignalParser().parse(envelope("GOLD SELL 4385"))

    with pytest.raises(ValueError, match="moved"):
        build_gold_follow_plan(
            event,
            live_entry=4400.0,
            bars=gold_bars(),
            spread=0.2,
            minimum_stop_distance=0.1,
            atr_period=14,
            structure_bars=24,
            stop_atr_multiple=1.5,
            structure_buffer_atr=0.25,
            target_reward_risk=1.2,
            max_entry_deviation_atr=0.75,
            max_entry_deviation_bps=8.0,
        )


def test_gold_follow_keeps_valid_provider_protection() -> None:
    event = RioSignalParser().parse(envelope("GOLD SELL 4385-4386 SL 4393 TP1 4378"))

    plan = build_gold_follow_plan(
        event,
        live_entry=4385.5,
        bars=gold_bars(),
        spread=0.2,
        minimum_stop_distance=0.1,
        atr_period=14,
        structure_bars=24,
        stop_atr_multiple=1.5,
        structure_buffer_atr=0.25,
        target_reward_risk=1.2,
        max_entry_deviation_atr=0.75,
        max_entry_deviation_bps=8.0,
    )

    assert not plan.used_fallback_stop
    assert not plan.used_fallback_target
    assert plan.stop_loss == 4393.0
    assert plan.take_profit == 4378.0


def test_gold_follow_replaces_wrong_side_provider_protection() -> None:
    event = RioSignalParser().parse(envelope("GOLD BUY 4385 SL 4390 TP1 4380"))

    plan = build_gold_follow_plan(
        event,
        live_entry=4385.0,
        bars=gold_bars(),
        spread=0.2,
        minimum_stop_distance=0.1,
        atr_period=14,
        structure_bars=24,
        stop_atr_multiple=1.5,
        structure_buffer_atr=0.25,
        target_reward_risk=1.2,
        max_entry_deviation_atr=0.75,
        max_entry_deviation_bps=8.0,
    )

    assert plan.used_fallback_stop
    assert plan.used_fallback_target
    assert plan.stop_loss < 4385.0
    assert plan.take_profit > 4385.0


@pytest.mark.parametrize(
    ("text", "kind", "value"),
    [
        ("GOLD Move SL at 4390", ExternalSignalKind.MOVE_STOP, 4390),
        ("GOLD Book Some Profits", ExternalSignalKind.BOOK_PROFIT, None),
        ("GOLD TP2 HIT", ExternalSignalKind.TP_HIT, None),
        ("GOLD CLOSE NOW", ExternalSignalKind.CLOSE, None),
    ],
)
def test_parses_campaign_updates(text: str, kind: ExternalSignalKind, value: float | None) -> None:
    event = RioSignalParser().parse(envelope(text))

    assert event.kind is kind
    assert event.move_stop_to == value


def test_marketing_summary_is_information_only() -> None:
    event = RioSignalParser().parse(envelope("Today 4 signals sent, all TP2. Share feedback"))

    assert event.kind is ExternalSignalKind.INFO


def test_inbox_deduplicates_same_notification_for_fifteen_minutes(tmp_path) -> None:
    inbox = ExternalSignalInbox(tmp_path)
    first = inbox.enqueue(
        {"app": "Rio Traders", "text": "GOLD SELL 4385", "source": "rio"},
        now=datetime.now(UTC),
    )
    second = inbox.enqueue(
        {"app": "Rio Traders", "text": "GOLD SELL 4385", "source": "rio"},
        now=datetime.now(UTC) + timedelta(seconds=10),
    )

    assert first.event_id != second.event_id
    assert len(inbox.pending()) == 1
    assert inbox.recent()[-1]["status"] == "DUPLICATE"


def test_campaign_requires_an_unambiguous_match(tmp_path) -> None:
    inbox = ExternalSignalInbox(tmp_path)
    inbox.open_campaign(
        source="rio",
        symbol_alias="GOLD",
        broker_symbol="XAUUSD",
        ticket=123,
        direction="SHORT",
        event_id="one",
    )

    key, campaign = inbox.active_campaign("rio", "GOLD") or ("", {})
    assert campaign["ticket"] == 123
    inbox.close_campaign(key, "CLOSE")
    assert inbox.active_campaign("rio", "GOLD") is None


def test_http_receiver_requires_token_and_accepts_form(tmp_path) -> None:
    inbox = ExternalSignalInbox(tmp_path)
    receiver = SignalReceiver(inbox, "127.0.0.1", 0, "x" * 32)
    receiver.start()
    assert receiver._server is not None
    port = receiver._server.server_address[1]
    url = f"http://127.0.0.1:{port}/v1/rio"
    body = urllib.parse.urlencode(
        {"app": "Rio Traders", "text": "GOLD SELL 4385", "source": "rio"}
    ).encode()
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=2)
        assert error.value.code == 401

        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "X-Jarvis-Token": "x" * 32,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
        assert response.status == 202
        assert payload["status"] == "queued"
        assert len(inbox.pending()) == 1
    finally:
        receiver.stop()
