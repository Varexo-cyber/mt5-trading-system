"""Filter protocol and the chain that runs them.

Filters are the layer between "the analysis likes this setup" and "the risk
layer may size it". They answer one question — may we enter this instrument
right now — and they answer it without knowing anything about the setup.

That separation matters: a filter that could see the score would eventually be
tempted to let a really good setup through a blackout window, and the value of
a hard rule is precisely that it does not negotiate.

Order matters. The chain runs cheapest-and-most-absolute first, so the reason
recorded in the journal is the most fundamental one: news beats session beats
spread beats correlation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.instrument import InstrumentSpec
from core.types import Direction, Position, Tick
from infra.logging import get_logger
from risk.reasons import Reason

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FilterContext:
    """Everything a filter is allowed to look at.

    Notably absent: the confluence score and the setup's levels. Filters decide
    on market conditions alone.
    """

    symbol: str
    spec: InstrumentSpec
    now: datetime
    #: The direction being considered. Only the correlation filter needs it —
    #: doubled exposure depends on which way both positions point.
    direction: Direction | None = None
    tick: Tick | None = None
    open_positions: tuple[Position, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FilterVerdict:
    """One filter's answer, with the numbers behind it.

    `data` is merged into the journal's cycle context, so a filter reports what
    it measured even when it passes: "spread 0.9 pips against a 2.1 baseline"
    is what makes it possible to tell later whether a threshold was sensible.
    """

    filter_name: str
    passed: bool
    reason: Reason = Reason.OK
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls, name: str, detail: str = "", **data: Any) -> FilterVerdict:
        return cls(filter_name=name, passed=True, detail=detail, data=data)

    @classmethod
    def block(cls, name: str, reason: Reason, detail: str, **data: Any) -> FilterVerdict:
        return cls(filter_name=name, passed=False, reason=reason, detail=detail, data=data)

    def __bool__(self) -> bool:
        return self.passed


class Filter(ABC):
    """A pre-trade gate."""

    name: str = "filter"

    @abstractmethod
    def check(self, ctx: FilterContext) -> FilterVerdict:
        """Decide whether entry is permitted right now.

        Must never raise for an ordinary "conditions are wrong" outcome — that
        is a blocking verdict. Raise only when the filter cannot tell, and even
        then prefer returning a block: a filter that cannot evaluate is a
        filter that has not cleared the trade.
        """


class FilterChain:
    """Runs filters in order and stops at the first block.

    Short-circuiting is intentional. Evaluating a correlation filter after the
    news filter has already said no wastes a data fetch, and a journal row
    listing four simultaneous reasons is harder to aggregate than one naming
    the first and most fundamental.
    """

    def __init__(self, filters: list[Filter]) -> None:
        self.filters = filters

    def check(self, ctx: FilterContext) -> tuple[FilterVerdict, dict[str, Any]]:
        """Return the deciding verdict and the merged data from every filter run.

        The data dict accumulates across filters that passed, so a blocked
        cycle still records the spread and session that were measured before
        the block.
        """
        collected: dict[str, Any] = {}
        for filter_ in self.filters:
            verdict = filter_.check(ctx)
            collected.update(verdict.data)
            if not verdict.passed:
                log.info(
                    "filter blocked entry",
                    extra={
                        "event": "filter_block",
                        "filter": verdict.filter_name,
                        "symbol": ctx.symbol,
                        "reason": str(verdict.reason),
                        "detail": verdict.detail,
                    },
                )
                return verdict, collected

        return (
            FilterVerdict.allow("chain", f"{len(self.filters)} filters clear"),
            collected,
        )

    def __len__(self) -> int:
        return len(self.filters)
