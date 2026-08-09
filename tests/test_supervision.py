"""The supervisor may only ever reduce risk. That is the whole safety argument."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from advisory.providers import (
    SUPERVISION_ACTIONS,
    ConsensusAdvisor,
    Supervision,
    _parse_supervision,
    build_supervision_payload,
)
from core.types import Direction, MarketContext, Position, Tick

LONG = 1
SHORT = -1


def _position():  # type: ignore[no-untyped-def]
    """A EURUSD long 6 pips up on a 10-pip stop, 20 pips from its target."""
    from datetime import UTC, datetime

    from core.types import Direction, Position

    return Position(
        ticket=1,
        symbol="EURUSD",
        direction=Direction.LONG,
        volume=0.01,
        price_open=1.10000,
        sl=1.09900,
        tp=1.10200,
        profit=0.60,
        swap=0.0,
        opened_at=datetime(2026, 8, 7, 10, 0, tzinfo=UTC),
    )


def _context():  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime, timedelta

    import numpy as np
    import pandas as pd

    from core.types import MarketContext, Series, Tick, Timeframe

    now = datetime(2026, 8, 7, 11, 0, tzinfo=UTC)
    bars = 60
    closes = 1.10000 + np.linspace(0, 0.0006, bars)
    index = pd.date_range(end=now, periods=bars, freq=Timeframe.H1.duration, tz=UTC)
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.0002,
            "low": closes - 0.0002,
            "close": closes,
            "tick_volume": np.full(bars, 500),
        },
        index=index,
    )
    del timedelta
    return MarketContext(
        symbol="EURUSD",
        now=now,
        series={Timeframe.H1: Series("EURUSD", Timeframe.H1, frame, now)},
        tick=Tick("EURUSD", now, bid=1.10060, ask=1.10072),
    )


# --------------------------------------------------------------- the guard ---


def test_hold_and_close_are_always_permitted() -> None:
    for action in ("hold", "close", "partial_close"):
        verdict = Supervision(action, "x", 0.8, close_fraction=0.5)
        assert verdict.is_risk_reducing(
            direction_sign=LONG, current_sl=90.0, current_tp=120.0, price_now=105.0
        )


def test_a_tighter_stop_is_permitted() -> None:
    verdict = Supervision("tighten_stop", "x", 0.8, stop_loss=100.0)
    assert verdict.is_risk_reducing(
        direction_sign=LONG, current_sl=90.0, current_tp=120.0, price_now=105.0
    )


def test_a_wider_stop_is_refused() -> None:
    """The one move that increases loss. Refused regardless of the argument."""
    verdict = Supervision("tighten_stop", "give it room", 0.9, stop_loss=80.0)
    assert not verdict.is_risk_reducing(
        direction_sign=LONG, current_sl=90.0, current_tp=120.0, price_now=105.0
    )


def test_a_stop_already_through_price_is_refused() -> None:
    """A stop on the far side of price is a market exit wearing a stop's name."""
    verdict = Supervision("tighten_stop", "x", 0.9, stop_loss=110.0)
    assert not verdict.is_risk_reducing(
        direction_sign=LONG, current_sl=90.0, current_tp=120.0, price_now=105.0
    )


def test_the_short_side_is_mirrored() -> None:
    tighter = Supervision("tighten_stop", "x", 0.8, stop_loss=100.0)
    assert tighter.is_risk_reducing(
        direction_sign=SHORT, current_sl=110.0, current_tp=80.0, price_now=95.0
    )
    wider = Supervision("tighten_stop", "x", 0.8, stop_loss=120.0)
    assert not wider.is_risk_reducing(
        direction_sign=SHORT, current_sl=110.0, current_tp=80.0, price_now=95.0
    )


def test_pulling_a_target_in_is_permitted_and_pushing_it_out_is_not() -> None:
    nearer = Supervision("pull_target_in", "x", 0.8, take_profit=115.0)
    assert nearer.is_risk_reducing(
        direction_sign=LONG, current_sl=90.0, current_tp=120.0, price_now=105.0
    )
    further = Supervision("pull_target_in", "let it run", 0.9, take_profit=140.0)
    assert not further.is_risk_reducing(
        direction_sign=LONG, current_sl=90.0, current_tp=120.0, price_now=105.0
    )


def test_a_target_behind_price_is_refused() -> None:
    verdict = Supervision("pull_target_in", "x", 0.8, take_profit=100.0)
    assert not verdict.is_risk_reducing(
        direction_sign=LONG, current_sl=90.0, current_tp=120.0, price_now=105.0
    )


def test_a_price_bearing_action_without_a_price_is_refused() -> None:
    for action in ("tighten_stop", "pull_target_in"):
        assert not Supervision(action, "x", 0.9).is_risk_reducing(
            direction_sign=LONG, current_sl=90.0, current_tp=120.0, price_now=105.0
        )


def test_an_unknown_action_is_refused() -> None:
    assert not Supervision("add_to_position", "x", 1.0).is_risk_reducing(
        direction_sign=LONG, current_sl=90.0, current_tp=120.0, price_now=105.0
    )


# -------------------------------------------------------------- the parser ---


def parse(payload: dict[str, object]) -> Supervision:
    return _parse_supervision(json.dumps(payload), "test", "m", "req")


def test_a_well_formed_verdict_parses() -> None:
    verdict = parse(
        {
            "action": "close",
            "reason": "M15 lost the trendline it held for 14 bars",
            "confidence": 0.71,
            "stop_loss": None,
            "take_profit": None,
            "close_fraction": None,
        }
    )
    assert verdict.action == "close"
    assert verdict.confidence == pytest.approx(0.71)
    assert not verdict.error


def test_an_action_outside_the_vocabulary_becomes_hold() -> None:
    verdict = parse(
        {
            "action": "reverse",
            "reason": "flip it",
            "confidence": 0.9,
            "stop_loss": None,
            "take_profit": None,
            "close_fraction": None,
        }
    )
    assert verdict.action == "hold"
    assert verdict.error == "invalid_response"


def test_a_partial_without_a_usable_fraction_becomes_hold() -> None:
    """It must not silently become a full exit."""
    verdict = parse(
        {
            "action": "partial_close",
            "reason": "bank some",
            "confidence": 0.8,
            "stop_loss": None,
            "take_profit": None,
            "close_fraction": None,
        }
    )
    assert verdict.action == "hold"


def test_a_fraction_outside_zero_to_one_becomes_hold() -> None:
    verdict = parse(
        {
            "action": "partial_close",
            "reason": "bank some",
            "confidence": 0.8,
            "stop_loss": None,
            "take_profit": None,
            "close_fraction": 1.5,
        }
    )
    assert verdict.action == "hold"


def test_a_zero_price_is_read_as_absent() -> None:
    """MT5 uses 0.0 for "no stop"; it must never be sent as a price."""
    verdict = parse(
        {
            "action": "tighten_stop",
            "reason": "x",
            "confidence": 0.8,
            "stop_loss": 0.0,
            "take_profit": None,
            "close_fraction": None,
        }
    )
    assert verdict.stop_loss is None
    assert not verdict.is_risk_reducing(
        direction_sign=LONG, current_sl=90.0, current_tp=120.0, price_now=105.0
    )


def test_unparseable_text_becomes_hold_not_close() -> None:
    """Failing to `close` would let a bad response liquidate the book."""
    verdict = _parse_supervision("not json at all", "test", "m", "")
    assert verdict.action == "hold"
    assert verdict.error == "invalid_response"


def test_the_structured_judgement_is_preserved_for_audit() -> None:
    verdict = parse(
        {
            "action": "hold",
            "reason": "H1 structure remains intact",
            "confidence": 0.78,
            "stop_loss": None,
            "take_profit": None,
            "close_fraction": None,
            "thesis_state": "intact",
            "urgency": "next_close",
            "evidence": ["M15 held the prior higher low"],
            "review_after_minutes": 5,
        }
    )

    assert verdict.thesis_state == "intact"
    assert verdict.urgency == "next_close"
    assert verdict.evidence == ("M15 held the prior higher low",)
    assert verdict.review_after_minutes == pytest.approx(5.0)


def test_supervision_r_stays_anchored_to_the_original_stop() -> None:
    now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    position = Position(
        ticket=7,
        symbol="TEST",
        direction=Direction.LONG,
        volume=0.01,
        price_open=100.0,
        sl=100.0,
        tp=120.0,
        profit=1.0,
        swap=0.0,
        opened_at=now - timedelta(hours=1),
    )
    context = MarketContext(
        "TEST",
        now,
        {},
        tick=Tick("TEST", now, bid=105.0, ask=105.2),
    )

    payload = build_supervision_payload(
        position,
        context,
        {
            "peak_r": 0.8,
            "trade_record": {"original_plan": {"stop_loss": 90.0}},
        },
    )

    assert payload["initial_risk_distance"] == pytest.approx(10.0)
    assert payload["unrealised_r"] == pytest.approx(0.5)
    assert payload["profit_given_back_r"] == pytest.approx(0.3)


# ------------------------------------------------------------- consensus ----


class _Fixed:
    def __init__(self, verdict: Supervision) -> None:
        self.verdict = verdict

    def review(self, *_a: object, **_k: object) -> object:  # pragma: no cover - unused
        raise NotImplementedError

    def reflect(self, *_a: object, **_k: object) -> object:  # pragma: no cover - unused
        raise NotImplementedError

    def supervise(self, _state: object) -> Supervision:
        return self.verdict


def test_consensus_takes_the_most_protective_answer() -> None:
    """Unanimity would let the least protective adviser decide."""
    advisor = ConsensusAdvisor(
        [
            _Fixed(Supervision("hold", "looks fine", 0.6)),
            _Fixed(Supervision("close", "thesis broken", 0.8)),
        ]
    )
    assert advisor.supervise({}).action == "close"


def test_the_action_ladder_is_ordered_by_protection() -> None:
    order = list(SUPERVISION_ACTIONS)
    assert order.index("hold") < order.index("tighten_stop") < order.index("close")
    assert order.index("tighten_stop") < order.index("partial_close")


class TestTheReviewerIsToldWhatTheMoneyMeans:
    """It was given `unrealised_money: 0.76` and the account currency, and
    nothing else. No account size, so no way to know whether that is most of a
    good day or a rounding error — on a hundred euro it is the first, on ten
    thousand the second, and the payload could not tell them apart.

    The operator's framing, which is the correct instinct: on a hundred euro,
    fifty to ninety cents is worth banking; on a thousand, five to twenty.
    """

    @staticmethod
    def payload(**extra):  # type: ignore[no-untyped-def]
        from advisory.providers import build_supervision_payload

        return build_supervision_payload(_position(), _context(), extra)

    def test_the_account_size_reaches_the_reviewer(self) -> None:
        state = self.payload(account_equity=123.43)

        assert state["account_equity"] == pytest.approx(123.43)

    def test_the_profit_is_expressed_as_a_share_of_the_account(self) -> None:
        """The number the judgement actually turns on."""
        state = self.payload(account_equity=100.0)

        money = state["unrealised_money"]
        assert state["unrealised_pct_of_account"] == pytest.approx(money, abs=0.02)

    def test_without_an_equity_the_shares_are_absent_rather_than_wrong(self) -> None:
        """A percentage of an unknown account would be a made-up number, and a
        made-up number in a prompt is worse than a missing one."""
        state = self.payload()

        assert state["account_equity"] is None
        assert state["unrealised_pct_of_account"] is None

    def test_what_is_still_to_win_is_stated_in_money(self) -> None:
        """ "Keep waiting or take it" is a comparison, and it cannot be made
        against a target expressed only as a price."""
        state = self.payload(account_equity=100.0)

        assert state["money_still_to_win_if_target_hit"] is not None
        assert state["money_if_the_current_stop_is_hit"] is not None

    def test_the_instructions_carry_the_operators_own_framing(self) -> None:
        from advisory.providers import _SUPERVISION_INSTRUCTIONS

        assert "JUDGE THE MONEY AGAINST THE ACCOUNT" in _SUPERVISION_INSTRUCTIONS
        assert "fifty to ninety cents" in _SUPERVISION_INSTRUCTIONS

    def test_the_reviewer_is_asked_to_look_forward_not_only_back(self) -> None:
        """Every mechanical rule fires on damage already done. This is the only
        part that can say where price is going, and it was never asked to."""
        from advisory.providers import _SUPERVISION_INSTRUCTIONS

        assert "ANTICIPATE, DO NOT ONLY REACT" in _SUPERVISION_INSTRUCTIONS
