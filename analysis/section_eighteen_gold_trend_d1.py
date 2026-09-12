"""SECTIE ACHTTIEN: meedoen met een goudtrend in plaats van eromheen scalpen.

WAAROM DEZE BESTAAT. Op 12 september 2026 mat `domtest.cmd holdout` over het
onaangeraakte jaar 1 sep 2024 - 31 aug 2025:

    de hele dag long goud, 256 dagen        +1250,93 R
    sectie zes over dezelfde periode           +4,21 R

Goud maakte een van zijn grootste bewegingen ooit. Sectie zes is LONG-ONLY
goud. Hij stond de goede kant op en ving er drie tiende procent van.

Dat is geen kostenprobleem en geen modelprobleem. Het is een DEELNAMEprobleem:
alle 44 bestaande modules zijn intraday, dus per definitie afwezig terwijl een
beweging van maanden zich voltrekt. Ze zoeken richting die hier gratis was.

Dit is de eerste sectie in dit project op de dagklok.

TWEE PARAMETERS, EN DAT IS HET ONTWERP. `trend_bars` en `stop_atr`. Zes
combinaties staan vooraf vastgelegd in de hypothese, en de Bonferroni-lat voor
zes is t>2,64. Sectie zes had veertig knoppen en heeft nooit een eerlijke lat
gehaald omdat niemand ooit telde hoeveel er geprobeerd was.

LONG ONLY, uit ontwerp en niet uit voorkeur. Dit is een deelnamevoertuig, geen
richtingvoorspeller. Short gaan is een tweede hypothese en verdient eigen
bewijs.

DE LAT DIE ERTOE DOET IS NIET DE WINST. Kopen-en-vasthouden gaf +1250,93 R bij
een terugval van 295,08 R: een verhouding van 4,24. Onbruikbaar op deze
rekening, want 295 R is een veelvoud van het kapitaal. Deze sectie hoeft niet
meer te VERDIENEN dan buy-and-hold -- hij moet meer verdienen PER EENHEID
TERUGVAL, en de hypothese legt die grens op 2,12. Haalt hij dat niet, dan is
hij een dure manier om goud te kopen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import SectionEighteenGoldTrendConfig
from core.types import MarketContext, Signal, Timeframe


def _atr(frame: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's true range, voortschrijdend gemiddeld.

    IDENTIEK AAN `analysis.sections_eight_nine._atr` op de periode na, en dat is
    hier geen toeval maar een reparatie: de nulhypothese in
    `scripts/dumb_session_baseline.py` rekende ooit zijn EIGEN ATR en zat er
    veertien keer naast. Twee definities van dezelfde grootheid is de fout die
    dit project achtervolgt.
    """

    previous = frame["close"].shift(1)
    spans = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return spans.rolling(period).mean()


class SectionEighteenGoldTrendD1:
    """Long XAUUSD zolang de dagslotkoers boven zijn trage gemiddelde staat."""

    name = "section_eighteen_gold_trend_d1"

    def __init__(self, config: SectionEighteenGoldTrendConfig | None = None) -> None:
        self.config = config or SectionEighteenGoldTrendConfig()

    def analyze(self, ctx: MarketContext) -> Signal:
        cfg = self.config
        if not cfg.enabled:
            return Signal.neutral(self.name, "sectie achttien staat uit")
        # ELKE STILTE ZEGT WELKE, en dat is de meest herhaalde fout in dit
        # project: sectie zes had drie storingen die dezelfde zin printten en
        # dat kostte drie dagen zoeken.
        if ctx.symbol not in cfg.allowed_symbols:
            return Signal.neutral(
                self.name, f"sectie achttien handelt {'/'.join(cfg.allowed_symbols)}, niet dit"
            )
        clock = Timeframe.parse(cfg.timeframe)
        series = ctx.series.get(clock)
        if series is None:
            return Signal.neutral(self.name, f"{cfg.timeframe} zit niet in deze context")
        frame = series.df
        needed = cfg.trend_bars + cfg.atr_period + 1
        if len(frame) < needed:
            return Signal.neutral(
                self.name,
                f"sectie achttien heeft {len(frame)} gesloten {cfg.timeframe}-bars en "
                f"er zijn er {needed} nodig",
            )

        close = frame["close"].astype(float)
        trend = close.rolling(cfg.trend_bars).mean()
        unit = _atr(frame, cfg.atr_period)
        last, average, atr = float(close.iloc[-1]), float(trend.iloc[-1]), float(unit.iloc[-1])
        if not np.isfinite([last, average, atr]).all() or atr <= 0.0:
            return Signal.neutral(
                self.name, "een dagwaarde is niet eindig (meestal een gat in de historie)"
            )

        if last <= average:
            return Signal.neutral(
                self.name,
                f"slot {last:.2f} staat onder het {cfg.trend_bars}-daags gemiddelde "
                f"{average:.2f}; deze sectie gaat niet short",
            )

        # DE AFSTAND TOT HET GEMIDDELDE IS EEN POORT, want instappen op een
        # koers die al ver boven zijn gemiddelde staat is dezelfde trade met een
        # veel slechter instappunt. De stop staat vast op ATR, dus hoe verder
        # boven het gemiddelde, hoe meer er terug te geven valt voordat de
        # these zelf breekt.
        stretch = (last - average) / atr
        if stretch > cfg.maximum_stretch_atr:
            return Signal.neutral(
                self.name,
                f"slot staat {stretch:.2f} ATR boven het gemiddelde, boven de "
                f"{cfg.maximum_stretch_atr:.2f} die deze sectie nog instapt",
            )

        entry = ctx.tick.mid if ctx.tick is not None else last
        return Signal(
            module=self.name,
            score=cfg.score,
            confidence=cfg.confidence,
            reasoning=(
                f"goud sluit {stretch:.2f} ATR boven zijn {cfg.trend_bars}-daags "
                f"gemiddelde; long zolang dat zo blijft"
            ),
            # DE STOP IS BREED EN DAT IS HET HELE PUNT. Een trend van weken
            # overleeft geen stop van een halve ATR; die wordt uitgeschud op
            # ruis en dan betaal je de spread opnieuw voor dezelfde these.
            invalidation_price=entry - cfg.stop_atr * atr,
            details={
                "timeframe": cfg.timeframe,
                "trend_bars": cfg.trend_bars,
                "stretch_atr": round(stretch, 4),
                "daily_atr": round(atr, 4),
                "trend_average": round(average, 4),
            },
        )
