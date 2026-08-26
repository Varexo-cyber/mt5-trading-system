"""Section six's own trades were recorded with the diagnostic columns empty.

FOUND BY READING A POSTMORTEM, NOT THE CODE. A XAUAUD short opened at 16:40:44
and closed at 16:40:56 -- twelve seconds -- never showed a cent of profit, and
returned -1.13R on a -1.00R plan. `why.cmd` was asked why, and answered:

    score         33.0 against a 35.0 threshold
    session       ?
    regime        ?

Three failures in five lines. The threshold is a false fact: section six has
its own lane precisely BECAUSE its module tops out at 33.75 against a bar of
45, so that comparison describes a vote the trade never faced, and any later
query filtering `total_score >= score_threshold` drops every section six trade
the account has ever taken. The other two are empty columns:
`_journal_cycle_context` promotes `session`, `spread_pips` and
`volatility_regime` out of the JSON, the lane passed `extra` straight through
from `_record_skip`, and on the NO_SIGNAL path that dict holds none of them.
`market_context` was never passed either, so ATR was NULL as well.

WHY IT MATTERS MOST ON THIS LANE. A scalp is over in seconds, which makes the
cost of entry -- not the quality of the read -- the likeliest explanation for
any single loser. Section six was the only lane recording none of it.

THE ARITHMETIC THESE COLUMNS EXIST FOR. A plan is drawn from the entry side of
the book and both exits happen on the other side: a short is sold at the bid
and bought back at the ask. So the market travels one spread LESS than the
stop looks to lose, and one spread MORE than the target looks to win. At the
configured gate that is four spreads against you versus eight for you, and a
coin flip lands at 33% where the 1.4:1 payoff needs 41.7%. Whether the module
beats that is a measurement nobody can make while the columns are NULL.
"""

from __future__ import annotations

from core.instrument import InstrumentSpec
from runner.service import JarvisRunner


class _Spec:
    """Only the one method the evidence builder asks for."""

    def __init__(self, pip: float) -> None:
        self.pip = pip

    def price_to_pips(self, price: float) -> float:
        return price / self.pip


class _SessionFilter:
    def session_label(self, moment: object) -> str:
        return "london"


class _Filters:
    def find(self, kind: type) -> object:
        return _SessionFilter()


class _Clock:
    def now(self) -> str:
        return "now"


def _runner() -> JarvisRunner:
    service = object.__new__(JarvisRunner)
    service.filters = _Filters()  # type: ignore[assignment]
    service.clock = _Clock()  # type: ignore[assignment]
    return service


def _evidence(**kwargs: object) -> dict:
    """The XAUAUD short that prompted this, unless told otherwise.

    entry 6398.94 bid, stop 6402.49, target 6393.97 -- a 3.55 stop and a 4.97
    target, the 1.0 / 1.4 candle spans the config asks for.
    """
    fields: dict = {
        "spec": _Spec(0.01),
        "extra": None,
        "entry": 6398.94,
        "stop": 6402.49,
        "target": 6393.97,
        "spread": 0.71,
    }
    fields.update(kwargs)
    return _runner()._scalp_cycle_evidence(**fields)  # type: ignore[arg-type]


def test_the_three_columns_that_were_empty_are_filled() -> None:
    evidence = _evidence()
    assert evidence["session"] == "london"
    assert evidence["spread_pips"] == 71.0
    assert evidence["section"] == "six"


def test_the_cost_of_entry_is_written_down_as_a_share_of_r() -> None:
    """0.71 of spread behind a 3.55 stop is a fifth of the risk, and that is
    the number that decides a trade measured in seconds."""
    evidence = _evidence()
    assert evidence["cost_share"] == 0.2


def test_both_distances_are_recorded_not_just_the_one_the_gate_uses() -> None:
    """The module gates on target/spread. R is the STOP, so the gate's own
    number cannot say what a loss costs -- both have to be there."""
    evidence = _evidence()
    assert evidence["stop_in_spreads"] == 5.0
    assert evidence["target_in_spreads"] == 7.0
    # And that pair is the whole finding: 5.0 - 1 = 4 spreads to lose against
    # 7.0 + 1 = 8 to win, so a coin flip hits the stop two times in three.
    adverse = evidence["stop_in_spreads"] - 1.0
    favourable = evidence["target_in_spreads"] + 1.0
    assert adverse / (adverse + favourable) < 1 / 2.4


def test_a_caller_that_already_measured_something_is_not_overwritten() -> None:
    """`extra` carries the filter chain's own readings when section one ran
    first. Its spread is measured at the gate; ours is measured at the order.
    The earlier one wins, because the filters are what actually judged it."""
    evidence = _evidence(extra={"spread_pips": 12.5, "session": "tokyo", "minutes_to_news": 40.0})
    assert evidence["spread_pips"] == 12.5
    assert evidence["session"] == "tokyo"
    assert evidence["minutes_to_news"] == 40.0


def test_a_market_with_no_quoted_spread_records_no_invented_ratio() -> None:
    """Zero spread is 'not supplied', not 'free to trade'. Dividing by it
    would write an infinite target-in-spreads into the journal and every
    later report would read it as the best trade the account ever took."""
    evidence = _evidence(spread=0.0)
    assert "cost_share" not in evidence
    assert "stop_in_spreads" not in evidence
    assert "spread_pips" not in evidence
    assert evidence["section"] == "six"


def test_the_columns_survive_the_promotion_into_the_journal() -> None:
    """Filling the dict is only half of it: `_journal_cycle_context` is what
    lifts these into their own columns, and it reads specific keys. A rename
    on either side would leave the report saying `?` again."""
    context = JarvisRunner._journal_cycle_context("XAUAUD", 224.25, _evidence())
    assert context.session == "london"
    assert context.spread_pips == 71.0
    assert context.extra["cost_share"] == 0.2


def test_the_spec_contract_is_the_real_one() -> None:
    """`_Spec` above stands in for an InstrumentSpec. If the method it fakes
    is ever renamed, this test fails instead of the stub silently drifting."""
    assert hasattr(InstrumentSpec, "price_to_pips")
