from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta

import pytest

from core.types import Direction
from external_signals import ExternalSignalInbox, NotificationEnvelope, RioSignalParser
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


def test_preliminary_signal_without_stop_never_becomes_entry() -> None:
    event = RioSignalParser().parse(envelope("GOLD\nSELL 4385"))

    assert event.kind is ExternalSignalKind.INVALID
    assert not event.complete_entry
    assert "stop loss" in event.reason


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
