"""US30 M1 and M5, on the two mechanisms this account already measured.

WHAT THIS IS AND, MORE IMPORTANTLY, WHAT IT IS NOT.

Four experimental sections -- impulse-retest and order-block, each on M1 and
M5, all four on US30 only. They are SHADOW: enabled so a replay can judge
them, weighted zero, and absent from `live_enabled_modules`. Nothing here has
been measured on US30 and nothing in this file claims otherwise.

THE DETECTORS ARE NOT COPIED, THEY ARE CALLED. `ImpulseRetest` and
`OrderBlock` are constructed here with US30's own config and asked the same
question the live sections ask them. Duplicating 245 lines of detection logic
would have produced a second implementation that drifts from the measured one
the first time either is touched -- which is the single most repeated defect
in this repository, and the reason `analysis/mechanisms.py` and
`analysis/legs_gap.py` exist at all.

So this file contains what is genuinely NEW and nothing else:

    the US30 symbol gate       the shared detectors have none; they read
                               whatever context they are handed
    the one-setup rule         no second entry while the same zone is still
                               the zone that fired
    a named refusal per path   a section that is quiet has to say which quiet

THE PARAMETERS ARE THE STOCK ONES, and that is worth saying out loud because
it looks like a coincidence and is not. Every number in the owner's brief --
ATR 14, impulse 1.0/1.5 ATR, span 1.5, tolerance 0.15/0.25, stop 0.85/1.0,
lookback 96, five bars back -- is already the default on `ImpulseRetestConfig`
and `OrderBlockConfig`.

Which means: THESE NUMBERS WERE MEASURED ON M15 AND M30, on FX and index CFDs,
over eleven years. They have never been measured on US30 M1 or M5. A one-ATR
impulse on M15 and a one-ATR impulse on M1 are not the same event, and the
cost of a trade against a one-ATR stop grows as the clock shrinks -- that is
exactly what turned section eleven from +0.047 R a trade into -0.18. Nobody
should read the parameter table as evidence. The replay is the evidence, and
it has not been run.
"""

from __future__ import annotations

from analysis.impulse_retest import ImpulseRetest
from analysis.order_block import OrderBlock
from core.types import MarketContext, Signal, Timeframe

#: The two mechanisms a US30 section may be built on. A name outside this set
#: is refused at config load rather than becoming a section that silently
#: never fires.
US30_MECHANISMS: tuple[str, ...] = ("impulse_retest", "order_block")


class SectionUs30:
    """One existing detector, restricted to US30 and to one clock."""

    def __init__(self, name: str, config, broker_symbol: str | None = None) -> None:
        self.name = name
        self.config = config
        # THE NAME THIS BROKER USES, resolved by the caller that has the
        # instruments table. `ctx.symbol` carries the broker's spelling and
        # Eightcap suffixes almost everything; comparing it raw against the
        # config's `US30` is a section that is silent on every bar with
        # nothing anywhere saying why. That exact mismatch has now silently
        # disabled four separate things in this repository.
        self.broker_symbol = broker_symbol or config.symbol
        # BUILT ONCE, FROM THIS SECTION'S OWN CONFIG. The detector never sees
        # the live section's settings and the live section never sees these,
        # so tuning US30 cannot reach section two or three.
        self.detector = (
            ImpulseRetest(config.as_impulse_config(), name=name)
            if config.mechanism == "impulse_retest"
            else OrderBlock(config.as_order_block_config(), name=name)
        )
        #: Per symbol, the zone this section last signalled on. The one-setup
        #: rule, and the reason this module is not stateless.
        self._last_zone: dict[str, tuple[int, float]] = {}

    # -- the read -----------------------------------------------------------

    def analyze(self, ctx: MarketContext) -> Signal:
        cfg = self.config

        def quiet(why: str) -> Signal:
            # EVERY SILENCE NAMES ITSELF. Section eleven produced 204,575
            # decisions reading "no weighted directional evidence" and there
            # was no way to tell a broken build from a rare mechanism.
            return Signal.neutral(self.name, f"{self.name}: {why}")

        if not cfg.enabled:
            return quiet("section disabled")
        if ctx.symbol not in (self.broker_symbol, cfg.symbol):
            return quiet("not this section's market")

        clock = Timeframe.parse(cfg.timeframe)
        if clock not in ctx.series:
            # ITS OWN CLOCK AND NO OTHER. An M1 section handed only M5 bars
            # would fall through to the detector, which reads
            # `config.timeframe` itself and would return a bare "needs N
            # closed bars" -- true, and it hides that the clock is absent.
            return quiet(f"{cfg.timeframe} bars were not loaded for this context")

        signal = self.detector.analyze(ctx)
        if not signal.score:
            # The detector's own words. It has more to say about why than this
            # wrapper does, and replacing them with a generic sentence is how
            # a diagnosis becomes a shrug.
            return Signal.neutral(self.name, signal.reasoning or f"{self.name}: no setup")

        zone = self.zone_of(signal)
        if zone is not None and self._last_zone.get(ctx.symbol) == zone:
            # ONE ENTRY PER SETUP. These detectors re-fire on every bar for as
            # long as price stays inside the zone, which is correct for a
            # detector and wrong for a section: it would take the same idea
            # five times and the duplicates are drawn from the losers, because
            # a zone that works is left in one bar and a zone that fails is
            # sat on.
            return quiet(f"already signalled on this zone at {zone[1]:.2f}")
        if zone is not None:
            self._last_zone[ctx.symbol] = zone
        return signal

    # -- the pieces, each drivable on its own by a test ----------------------

    def zone_of(self, signal: Signal) -> tuple[int, float] | None:
        """`(direction, level)` identifying the setup this signal came from.

        Read from the detector's own `details`/`key_levels` rather than
        recomputed, so the thing being de-duplicated is the thing that fired.
        A signal with no identifiable level is NOT de-duplicated -- refusing
        on an identity nothing produced would silence the section on a
        detail of its own reporting.
        """

        level = (signal.details or {}).get("level")
        if level is None and signal.key_levels:
            level = signal.key_levels[0]
        if level is None:
            return None
        return (1 if signal.score > 0 else -1, round(float(level), 6))
