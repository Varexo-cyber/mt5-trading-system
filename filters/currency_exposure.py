"""One bet, placed once. Not the same bet wearing two symbols.

A live account held GBPAUD short and GBPJPY short at the same time, both
losing, and the operator reasonably read that as two bad trades. It was one
bad trade sized twice: a short on any GBP cross is a short on GBP, and when
GBP rallied both moved together because they were never independent.

The correlation filter is next door and did not catch it, by design rather
than by accident. It measures the statistical correlation of returns over 200
hourly bars, deliberately preferring a measurement to a static table because
currency relationships shift. That is right, and it is blind to this: GBPAUD
and GBPJPY can measure well under the 0.7 threshold — the AUD and JPY legs
pull them apart — while the GBP leg they share is an identity, not a
correlation. It does not weaken when the measurement says so.

So this filter does not measure anything. It decomposes each open position
into the two currencies it is long and short, adds up what the account is
already carrying, and refuses a candidate that would stack more of the same
direction onto a currency it is already exposed to.

On a two-slot account the default of one is the whole point: two positions
that share a directional leg are one position with a second lot attached, and
the account believed it was diversified.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from config.schema import CurrencyExposureConfig
from core.instrument import InstrumentSpec
from core.types import Direction
from filters.base import Filter, FilterContext, FilterVerdict
from infra.logging import get_logger
from risk.reasons import Reason

log = get_logger(__name__)

#: symbol -> spec. Injected for the same reason the correlation filter injects
#: its series provider: this filter never touches the connector, so it replays.
SpecProvider = Callable[[str], InstrumentSpec]


def legs(spec: InstrumentSpec, direction: Direction) -> dict[str, int]:
    """What one position is long and short, by currency.

    A long on GBPAUD is long GBP and short AUD; a short is the reverse. That
    is the definition of a cross, not a heuristic, which is why this needs no
    threshold and no lookback.

    Instruments whose base is not a currency — XAU, BTC, an index — return only
    the leg that is one. The pseudo-currency is left out rather than tracked,
    because two gold positions are not a currency concentration and pretending
    otherwise would block a legitimate second trade.
    """
    sign = int(direction)
    base, quote = spec.currency_base.upper(), spec.currency_profit.upper()
    exposure: dict[str, int] = {}
    if _is_currency(base):
        exposure[base] = sign
    if _is_currency(quote):
        exposure[quote] = -sign
    return exposure


def _is_currency(code: str) -> bool:
    """Three letters that name money, rather than a metal or a coin.

    A deliberately short list of what to *exclude*. Treating an unknown code as
    a currency errs toward blocking a second position, which is the safe
    direction for an exposure limit.
    """
    return len(code) == 3 and code not in _NOT_CURRENCIES


#: Bases that look like currency codes and are not.
_NOT_CURRENCIES = frozenset({"XAU", "XAG", "XPT", "XPD", "BTC", "ETH", "LTC", "XRP", "BCH"})


class CurrencyExposureFilter(Filter):
    """Refuses a second position that doubles a currency bet already on."""

    name = "currency_exposure"

    def __init__(self, config: CurrencyExposureConfig, spec_provider: SpecProvider) -> None:
        self.config = config
        self.spec_provider = spec_provider

    def standing(self, ctx: FilterContext) -> Counter[str]:
        """Net directional exposure per currency across the open book.

        Counted in positions rather than weighted by risk. On an account that
        sizes every trade to the same percentage the two are the same number,
        and the position count is the one an operator can check against their
        own terminal.
        """
        totals: Counter[str] = Counter()
        for position in ctx.open_positions:
            try:
                spec = self.spec_provider(position.symbol)
            except Exception as exc:  # noqa: BLE001 - an unreadable spec is not fatal
                log.warning(
                    "cannot read spec for exposure accounting",
                    extra={"symbol": position.symbol, "reason": str(exc)},
                )
                continue
            for code, sign in legs(spec, position.direction).items():
                totals[code] += sign
        return totals

    def check(self, ctx: FilterContext) -> FilterVerdict:
        if not self.config.enabled or not ctx.open_positions:
            return FilterVerdict.allow(
                self.name,
                (
                    "no open positions to stack against"
                    if self.config.enabled
                    else "currency exposure filter disabled"
                ),
            )
        if ctx.direction is None:
            raise ValueError(
                "currency exposure needs the intended direction; which currency a "
                "cross is long is not defined without it"
            )

        standing = self.standing(ctx)
        adding = legs(ctx.spec, ctx.direction)
        limit = self.config.max_positions_per_currency

        for code, sign in adding.items():
            already = standing.get(code, 0)
            # Only stacking counts. Taking the other side reduces the bet and
            # is exactly what a book should be allowed to do.
            if sign * already < limit:
                continue
            side = "long" if sign > 0 else "short"
            return FilterVerdict.block(
                self.name,
                Reason.CURRENCY_CONCENTRATION,
                f"already {side} {code} on {abs(already)} open position(s); this would "
                f"make it {abs(already) + 1}. GBPAUD short and GBPJPY short are not two "
                f"trades, they are one GBP short with a second lot on it",
                currency=code,
                standing=already,
                would_be=already + sign,
            )

        return FilterVerdict.allow(
            self.name,
            "adds no currency the book is already leaning on: "
            + ", ".join(f"{code}{sign:+d}" for code, sign in sorted(adding.items())),
            exposure={code: count for code, count in standing.items() if count},
        )
