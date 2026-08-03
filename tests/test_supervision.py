"""The supervisor may only ever reduce risk. That is the whole safety argument."""

from __future__ import annotations

import json

import pytest

from advisory.providers import (
    SUPERVISION_ACTIONS,
    ConsensusAdvisor,
    Supervision,
    _parse_supervision,
)

LONG = 1
SHORT = -1


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
