"""One index stepped out of line with the others. That gap closes.

SECTION FIVE, and the first reader on this account that is MEANINGLESS on a
single chart. Every one of the nine in section one reads one price series;
drift burst runs a test on one price series. This one does not exist without a
second instrument, which is exactly why it can be the corroboration the others
structurally cannot give each other.

THE IDEA. Equity indices do not move independently. SPX500, NDX100, US30,
FRA40, UK100 share macro drivers, risk appetite and a session, and intraday
they run 80-95% together. So when the basket moves 40 basis points and one
member moves 10, the thing that moved four indices is the same thing that has
not yet moved the fifth.

WHY THE HIT RATE IS HIGH BY CONSTRUCTION. This does not forecast the market.
The basket and the laggard may both go up, both go down, or neither — the trade
pays when the GAP closes. Being wrong about direction and right about the trade
is possible here and impossible for a trend follower, and that is the whole
reason it is worth having.

WHY IT CAN GO LIVE WHERE SECTIONS TWO AND FOUR CANNOT. Those rest on a
statistic the research measured on tick data, and whether it survives M1 bars
is genuinely unknown. There is no equivalent question here: a move measured
between two M1 closes is exactly as well measured on M1 as on ticks. Nothing to
wait out.

THE TRAP, AND IT IS THE ONE THAT KILLS NAIVE VERSIONS. FRA40 and UK100 do not
keep the same hours as SPX500. Comparing a closed market to an open one is
comparing a STALE price to a live one, and it manufactures divergences that do
not exist — every one of them a "gap" that will never close because one side
stopped printing hours ago. This does not consult a session table to avoid it.
It requires every peer's last bar to be recent, which is the same fact measured
directly rather than inferred from a calendar that can be wrong about a holiday.

WHAT IT IS NOT. On this account there is no margin for two legs, so this trades
ONE side — the laggard, toward the basket. That is a directional trade with a
relative trigger, not arbitrage, and calling it market-neutral would be a lie
about where the risk sits. When an index decouples for a real reason the gap
widens instead of closing, and that is where the losses are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config.schema import BasketDivergenceConfig
from core.types import MarketContext, Signal, Timeframe

#: Key the runner writes the peer readings under. One name, imported by both
#: sides, because a string typed twice is a string that eventually differs.
BASKET_META_KEY = "basket_peers"


@dataclass(frozen=True, slots=True)
class PeerMove:
    """What one other instrument in the basket just did."""

    symbol: str
    #: Move over the same window, in basis points, so a 7,600 index and a
    #: 40,000 index are comparable without either dominating.
    move_bp: float
    #: Seconds since that peer's last closed bar. The session guard.
    age_seconds: float


def divergence(
    own_move_bp: float, peers: list[PeerMove], *, max_age_seconds: float, minimum_peers: int
) -> tuple[float, float, int] | None:
    """The gap between this instrument and its basket, or None.

    Returns (gap_bp, basket_bp, peers_used). The basket is a MEDIAN rather than
    a mean: one index halted, gapping on its own news, or simply mispriced for a
    minute would drag a mean far enough to invent a divergence in every other
    member of the group at once. The median ignores it.
    """
    fresh = [peer for peer in peers if peer.age_seconds <= max_age_seconds]
    if len(fresh) < minimum_peers:
        return None
    moves = sorted(peer.move_bp for peer in fresh)
    middle = len(moves) // 2
    basket = moves[middle] if len(moves) % 2 else (moves[middle - 1] + moves[middle]) / 2.0
    if not math.isfinite(basket) or not math.isfinite(own_move_bp):
        return None
    return basket - own_move_bp, basket, len(fresh)


class BasketDivergence:
    """Trades the laggard toward its basket."""

    name = "basket_divergence"

    def __init__(self, config: BasketDivergenceConfig | None = None) -> None:
        self.config = config or BasketDivergenceConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        config = self.config
        if not config.enabled:
            return Signal.neutral(self.name, "basket divergence disabled")
        raw = (ctx.meta or {}).get(BASKET_META_KEY)
        if not raw:
            return Signal.neutral(self.name, "no basket peers this cycle")
        peers = [item for item in raw if isinstance(item, PeerMove)]
        if not peers:
            return Signal.neutral(self.name, "no usable basket peers")

        timeframe = Timeframe.parse(config.timeframe)
        series = ctx.series.get(timeframe)
        if series is None or len(series.df) < config.move_bars + 1:
            return Signal.neutral(
                self.name, f"needs {config.move_bars + 1} closed {timeframe.value} bars"
            )
        closes = series.df["close"]
        start = float(closes.iloc[-(config.move_bars + 1)])
        now = float(closes.iloc[-1])
        if start <= 0 or now <= 0:
            return Signal.neutral(self.name, "unusable prices")
        own_bp = (now / start - 1.0) * 10_000.0

        found = divergence(
            own_bp,
            peers,
            max_age_seconds=config.peer_max_age_seconds,
            minimum_peers=config.minimum_peers,
        )
        if found is None:
            return Signal.neutral(
                self.name,
                f"fewer than {config.minimum_peers} peers with a bar inside "
                f"{config.peer_max_age_seconds:.0f}s — a stale peer is a closed market",
            )
        gap_bp, basket_bp, used = found
        details = {
            "gap_bp": gap_bp,
            "basket_bp": basket_bp,
            "own_bp": own_bp,
            "peers": used,
            "minimum_gap_bp": config.minimum_gap_bp,
        }

        if abs(gap_bp) < config.minimum_gap_bp:
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning=(
                    f"in line with its basket: {own_bp:+.0f}bp against {basket_bp:+.0f}bp "
                    f"over {used} peers, a {gap_bp:+.0f}bp gap"
                ),
                details=details,
            )

        # THE BASKET HAS TO HAVE ACTUALLY MOVED.
        #
        # The gap is a difference, so it can be large because the basket ran or
        # because THIS instrument ran while the basket sat still. Those are
        # opposite trades: the first is a laggard catching up, the second is a
        # single market doing its own thing — which is the decoupling this
        # module is most wrong about. Only the first is the setup.
        if abs(basket_bp) < config.minimum_basket_bp:
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning=(
                    f"the basket barely moved ({basket_bp:+.0f}bp) while this went "
                    f"{own_bp:+.0f}bp — that is this market decoupling, not a laggard"
                ),
                details=details,
            )
        if (gap_bp > 0) != (basket_bp > 0):
            return Signal(
                module=self.name,
                score=0.0,
                confidence=0.0,
                reasoning=(
                    f"gap {gap_bp:+.0f}bp points against a basket of {basket_bp:+.0f}bp — "
                    f"this market has overshot its peers rather than lagged them"
                ),
                details=details,
            )

        span = max(1e-9, config.gap_saturation_bp - config.minimum_gap_bp)
        strength = min(1.0, (abs(gap_bp) - config.minimum_gap_bp) / span)
        direction = 1.0 if gap_bp > 0 else -1.0
        score = direction * (config.base_score + strength * (100.0 - config.base_score))
        room = config.maximum_confidence - config.base_confidence
        confidence = min(config.maximum_confidence, config.base_confidence + strength * room)
        way = "behind" if gap_bp > 0 else "ahead of"
        return Signal(
            module=self.name,
            score=score,
            confidence=confidence,
            reasoning=(
                f"{abs(gap_bp):.0f}bp {way} a basket of {used} peers that moved "
                f"{basket_bp:+.0f}bp while this moved {own_bp:+.0f}bp"
            ),
            details=details,
        )
