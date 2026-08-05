"""Do the independent techniques agree before real money is committed?

`veto_on_conflict` only ever caught the playbooks contradicting *each other*.
The swing engine reading H1 structure as LONG while an M5 impulse theory read
the same chart as SHORT went straight through — which is the one case where two
genuinely different techniques disagree, and the clearest evidence available
that the read is ambiguous.

Standing aside there costs a trade that was a coin flip. Taking it costs the
spread plus, more often than not, the stop.
"""

from __future__ import annotations

from types import SimpleNamespace

from analysis.confluence import TradeIdea
from analysis.playbooks import Play, PlaybookVerdict
from core.types import Direction
from runner.service import JarvisRunner

FLOOR = 60.0


def gate(*, require: bool = True, floor: float = FLOOR):  # type: ignore[no-untyped-def]
    instance = JarvisRunner.__new__(JarvisRunner)
    instance.playbook_config = SimpleNamespace(  # type: ignore[assignment]
        require_method_agreement=require, min_conviction=floor
    )
    return instance._method_disagreement


def idea(direction: Direction = Direction.LONG) -> TradeIdea:
    return TradeIdea(
        symbol="EURUSD",
        approved=True,
        direction=direction,
        score=62.0,
        confidence=0.8,
        entry=1.10,
        stop_loss=1.09,
        take_profit=1.12,
        reason="H1 structure",
        signals=(),
    )


def play(direction: Direction, conviction: float = 75.0, name: str = "momentum_scalp") -> Play:
    return Play(
        playbook=name,
        direction=direction,
        entry=1.10,
        stop_loss=1.0988,
        take_profit=1.1024,
        conviction=conviction,
        horizon_minutes=60,
        thesis="M5 impulse",
    )


def verdict(*plays: Play) -> PlaybookVerdict:
    return PlaybookVerdict(plays=tuple(plays), conflict=False, note="")


def test_an_opposing_theory_stands_the_trade_down() -> None:
    said = gate()(idea(Direction.LONG), verdict(play(Direction.SHORT)))
    assert said is not None
    assert "momentum_scalp says SHORT" in said
    assert "LONG" in said


def test_agreement_passes() -> None:
    assert gate()(idea(Direction.LONG), verdict(play(Direction.LONG))) is None


def test_silence_is_not_disagreement() -> None:
    """A theory with no setup has no opinion. Most markets have no short-horizon
    setup at any moment, and treating that as a veto would refuse nearly
    everything."""
    assert gate()(idea(Direction.LONG), verdict()) is None


def test_no_playbooks_at_all_passes() -> None:
    assert gate()(idea(Direction.LONG), None) is None


def test_a_weak_opposing_play_does_not_count() -> None:
    """It was not good enough to trade in its own right, so it is not good
    enough to cancel somebody else's trade either."""
    assert gate()(idea(Direction.LONG), verdict(play(Direction.SHORT, conviction=40.0))) is None


def test_a_play_exactly_at_the_floor_counts() -> None:
    """The floor is what a play must reach to be tradeable, so reaching it is
    also what makes it worth listening to."""
    assert (
        gate()(idea(Direction.LONG), verdict(play(Direction.SHORT, conviction=FLOOR))) is not None
    )


def test_every_opposing_theory_is_named() -> None:
    """The sentence lands in the journal, and "something disagreed" is not
    answerable six months later."""
    said = gate()(
        idea(Direction.LONG),
        verdict(
            play(Direction.SHORT, name="momentum_scalp"),
            play(Direction.SHORT, name="range_fade"),
        ),
    )
    assert said is not None
    assert "momentum_scalp" in said and "range_fade" in said


def test_an_agreeing_play_alongside_an_opposing_one_still_stands_down() -> None:
    """Two of three techniques agreeing is not consensus, it is a split — and a
    split read is the thing worth avoiding."""
    said = gate()(
        idea(Direction.LONG),
        verdict(play(Direction.LONG, name="range_fade"), play(Direction.SHORT)),
    )
    assert said is not None


def test_the_short_side_is_mirrored() -> None:
    assert gate()(idea(Direction.SHORT), verdict(play(Direction.LONG))) is not None
    assert gate()(idea(Direction.SHORT), verdict(play(Direction.SHORT))) is None


def test_it_can_be_switched_off() -> None:
    assert gate(require=False)(idea(Direction.LONG), verdict(play(Direction.SHORT))) is None
