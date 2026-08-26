"""Fitting a trade to the margin must mean the same thing on both sides of the
adviser, and for months it did not.

WHAT THE DECK SHOWED. The AI exchange panel's own header says only proposals
that have already passed "analyse, risico, filters, sizing en marge" reach the
decision layer -- and then most of what reached it died there on
INSUFFICIENT_MARGIN. A proposal cannot pass the margin on the way in and fail
it on the way out unless the two checks are different checks.

They were. Before the adviser, a position too large for the account is SHRUNK
until it fits and only refused when even `volume_min` will not go
(`reduce_volume_to_fit_margin`). After the adviser, the revalidation re-sizes
from scratch -- which recreates the original oversized volume -- and then
refused outright with no fitting at all:

    sizer wants 0.32 lots        -> margin refuses
    fitted down to 0.05          -> margin passes
    adviser approves
    revalidation re-sizes        -> 0.32 lots again
    margin refuses               -> INSUFFICIENT_MARGIN

The trade was made to fit, then unmade, then refused for not fitting. On a 176
EUR account against single-name stocks wanting hundreds of euro of margin,
that was most of the funnel.

One helper, both call sites. Two copies of a rule that must agree is how this
started.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from runner.service import JarvisRunner


@dataclass
class _Sizing:
    volume: float
    entry: float
    actual_risk_money: float
    actual_risk_pct: float


@dataclass
class _Margin:
    approved: bool
    reason: str = "INSUFFICIENT_MARGIN"
    detail: str = "not enough margin"


class _Risk:
    """Refuses anything above `affordable`, and reports the largest that fits."""

    def __init__(self, affordable: float) -> None:
        self.affordable = affordable
        self.checks: list[float] = []

    def check_margin(self, state, symbol, direction, volume, entry):  # type: ignore[no-untyped-def]
        self.checks.append(volume)
        return _Margin(approved=volume <= self.affordable)

    def largest_volume_within_margin(  # type: ignore[no-untyped-def]
        self, state, symbol, direction, volume, entry, *, volume_min, volume_step
    ):
        if self.affordable < volume_min:
            return 0.0
        return min(volume, self.affordable)


def _runner(affordable: float) -> JarvisRunner:
    service = object.__new__(JarvisRunner)
    service.risk = _Risk(affordable)  # type: ignore[attr-defined]
    service.settings = SimpleNamespace(  # type: ignore[attr-defined]
        risk=SimpleNamespace(reduce_volume_to_fit_margin=True)
    )
    return service


SPEC = SimpleNamespace(volume_min=0.01, volume_step=0.01)


def _fit(service, sizing):  # type: ignore[no-untyped-def]
    first = service.risk.check_margin(None, "ADS", None, sizing.volume, sizing.entry)
    return service._fit_to_margin(None, "ADS", None, sizing, SPEC, first)


def test_an_oversized_position_is_shrunk_rather_than_refused() -> None:
    """The live shape: the risk budget asks for 0.32 lots, the account holds
    enough for 0.05."""
    service = _runner(affordable=0.05)
    sizing = _Sizing(volume=0.32, entry=100.0, actual_risk_money=8.80, actual_risk_pct=5.0)

    fitted, margin = _fit(service, sizing)

    assert margin.approved
    assert fitted.volume == 0.05


def test_the_risk_shrinks_with_the_volume() -> None:
    """Entry, stop and target do not move, so the expectancy that approved the
    trade is the expectancy of the smaller one -- but the money at risk is not
    the money the sizer originally worked out."""
    service = _runner(affordable=0.05)
    sizing = _Sizing(volume=0.10, entry=100.0, actual_risk_money=8.80, actual_risk_pct=5.0)

    fitted, _ = _fit(service, sizing)

    assert fitted.actual_risk_money == 4.40
    assert fitted.actual_risk_pct == 2.5


def test_it_never_rounds_up_to_reach_the_broker_minimum() -> None:
    """The owner's standing rule. When even `volume_min` will not fit, nothing
    is changed and the caller's refusal stands."""
    service = _runner(affordable=0.001)
    sizing = _Sizing(volume=0.32, entry=100.0, actual_risk_money=8.80, actual_risk_pct=5.0)

    unchanged, margin = _fit(service, sizing)

    assert not margin.approved
    assert unchanged.volume == 0.32


def test_a_position_that_already_fits_is_left_exactly_alone() -> None:
    service = _runner(affordable=1.0)
    sizing = _Sizing(volume=0.05, entry=100.0, actual_risk_money=8.80, actual_risk_pct=5.0)

    same, margin = _fit(service, sizing)

    assert margin.approved
    assert same is sizing


def test_both_paths_call_the_same_helper() -> None:
    """The actual defect was two copies of this rule, one of which had no
    fitting in it. Read the source rather than trusting that it stays true:
    every margin refusal in the candidate flow must be preceded by a fit.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(JarvisRunner))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_fit_to_margin"
    ]

    assert len(calls) >= 2, "the post-adviser revalidation is not fitting again"
