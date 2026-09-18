"""Sectie 21: vooraf vastgelegde meting van de straddle-afwikkeling op XAUUSD.

WAT DIT IS. De eigenaar vroeg om "hedging op mijn balans": 0,01 buy en 0,01
sell tegelijk, het been dat meeloopt met de trend aanhouden, het been dat in
verlies staat sluiten.

DAT IS GEEN HEDGE MAAR EEN STRADDLE. Een hedge dekt bestaand risico af; hier
wordt risico in twee richtingen tegelijk gecreeerd en daarna een kant
weggegooid. Je dekt niets af -- je betaalt om te ontdekken welke kant het op
gaat. `risk_manager.assert_not_forbidden` weigert het vandaag met zoveel
woorden, en die regel blijft staan.

DE BREAKEVEN LIGT VAST EN IS NIET TE ONTLOPEN:

    netto = winst winnaar - verlies verliezer - 2 x spread x 2 benen

DE CONTROLE IS HET HELE PUNT VAN DEZE MODULE. Naast elke straddle draait
dezelfde cyclus met EEN vooraf gekozen richting, op hetzelfde moment en met
dezelfde X en Y. Verslaat de straddle die niet, dan is het tweede been zuivere
kostenpost en is de vraag beantwoord zonder dat er iets live hoeft. Daarnaast
loopt `altijd long` mee als domme nulhypothese, want goud steeg -- en dan
verdient long in elk venster geld.

EN DE VAL BIJ "WELK UUR WERKT HET BEST". Het beste van vierentwintig uren is
het maximum van vierentwintig ruizige getallen; dat ziet er altijd goed uit.
`uur_permutatie` gooit dezelfde cycli duizend keer willekeurig over de uren en
rapporteert hoe goed het beste uur er dan uitziet. Is het echte beste uur niet
beter dan die verdeling, dan is het een selectie en geen vondst.

DRAAIEN: straddle.cmd, op Windows -- MT5 is Windows-only. De rekenkern
hieronder draait in de tests op verzonnen bars, zonder MT5.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC

import numpy as np
import pandas as pd

from scripts.section_twenty_pullback_ladder import CONTRACT, _lees_csv, _sessie_van


@dataclass(frozen=True)
class Instelling:
    """Een configuratie. Er worden er 48 gedraaid."""

    #: Verlies waarbij het tegendraadse been eruit gaat, in punten.
    kap: float = 5.0
    #: Doel voor het overgebleven been, in punten. Alleen gebruikt als
    #: `compenseer` uit staat.
    doel: float = 20.0
    #: DE REGEL ZOALS HIJ BEDOELD IS. De winnaar loopt door tot de hele mand
    #: het verlies van het gesloten been heeft goedgemaakt PLUS deze marge in
    #: punten. Niet tot een vast doel -- dat was mijn fout.
    compenseer: bool = True
    marge: float = 2.0
    #: Stop voor het overgebleven been. None = laten lopen, zoals gevraagd.
    winnaar_stop: float | None = None
    lot: float = 0.01
    spread: float = 0.16
    #: Hoeveel bars een cyclus maximaal open blijft voordat hij plat gaat.
    max_bars: int = 240

    @property
    def naam(self) -> str:
        stop = "geen" if self.winnaar_stop is None else f"{self.winnaar_stop:g}"
        if self.compenseer:
            return f"kap{self.kap:g}·compenseer+{self.marge:g}·stop{stop}"
        return f"kap{self.kap:g}·doel{self.doel:g}·stop{stop}"


@dataclass
class Cyclus:
    """Een straddle van openen tot plat, of een controle-trade."""

    geopend: pd.Timestamp
    soort: str                      # "straddle" | "een_richting" | "altijd_long"
    netto_euro: float = 0.0
    diepste_euro: float = 0.0
    #: +1 / -1 / 0 -- welk been de cyclus uitliep. 0 = allebei afgekapt.
    gehouden: int = 0
    bars: int = 0
    gesloten: pd.Timestamp | None = None


def _kosten(instelling: Instelling, benen: int) -> float:
    """Spread twee keer per been: in en uit."""

    return instelling.spread * 2 * instelling.lot * CONTRACT * benen


def _loop_een_been(
    m1: pd.DataFrame, start: int, entry: float, richting: int, instelling: Instelling,
    nodig: float | None = None,
) -> tuple[float, float, int, pd.Timestamp | None]:
    """Het overgebleven been tot het doel, de stop of de tijd.

    `nodig` is hoeveel punten dit been moet maken. Bij de compenseer-regel is
    dat het verlies van het afgekapte been plus de marge plus de kosten; bij
    een vast doel is het gewoon `instelling.doel`.
    """

    if nodig is None:
        nodig = instelling.doel

    diepste = 0.0
    for offset in range(instelling.max_bars):
        pos = start + offset
        if pos >= len(m1):
            break
        bar = m1.iloc[pos]
        hoog, laag = float(bar["high"]), float(bar["low"])
        # Expliciet per richting en zonder tekentrucjes. Een long staat er het
        # slechtst voor op de LOW en het best op de HIGH; bij een short precies
        # andersom. Dat met een `* richting` proberen op te lossen is precies
        # hoe je hier een omgedraaid teken in stopt dat niemand meer ziet.
        slechtst = (laag - entry) if richting > 0 else (entry - hoog)
        best = (hoog - entry) if richting > 0 else (entry - laag)
        diepste = min(diepste, slechtst)

        # PESSIMISTISCH: raakt een bar zowel de stop als het doel, dan wint de
        # STOP. Wat er binnen die bar eerst gebeurde weet niemand, en de andere
        # aanname flatteert elke uitslag.
        if instelling.winnaar_stop is not None and slechtst <= -instelling.winnaar_stop:
            return -instelling.winnaar_stop, diepste, offset + 1, m1.index[pos]
        # HET DOEL IS NIET VAST MAAR AFGELEID.
        #
        # Gevraagd is: de winnaar loopt door tot hij het verlies van het
        # afgekapte been heeft GOEDGEMAAKT plus wat winst. Ik had er een vast
        # doel van gemaakt, en dat is een andere strategie -- een vast doel
        # sluit te vroeg als het verlies groot was en te laat als het klein was.
        #
        # `nodig` wordt door de aanroeper meegegeven: de kap plus de marge plus
        # de vier spreads. Precies genoeg om de mand op winst te zetten.
        if best >= nodig:
            return nodig, diepste, offset + 1, m1.index[pos]

    pos = min(start + instelling.max_bars, len(m1)) - 1
    slot = float(m1.iloc[pos]["close"])
    punten = (slot - entry) if richting > 0 else (entry - slot)
    return punten, diepste, instelling.max_bars, m1.index[pos]


def straddle_cyclus(
    m1: pd.DataFrame, start: int, *, instelling: Instelling
) -> Cyclus | None:
    """Beide benen open, de verliezer bij -kap eruit, de winnaar laten lopen."""

    if start >= len(m1):
        return None
    entry = float(m1.iloc[start]["open"])
    cyclus = Cyclus(geopend=m1.index[start], soort="straddle")

    for offset in range(instelling.max_bars):
        pos = start + offset
        if pos >= len(m1):
            break
        bar = m1.iloc[pos]
        hoog, laag = float(bar["high"]), float(bar["low"])
        long_stuk = laag - entry          # slechtste stand van het LONG been
        short_stuk = entry - hoog         # slechtste stand van het SHORT been

        long_af = long_stuk <= -instelling.kap
        short_af = short_stuk <= -instelling.kap

        if long_af and short_af:
            # ALLEBEI IN EEN BAR. De volgorde binnen die bar is onbekend, en de
            # pessimistische lezing is dat je ze allebei kwijt bent. Dit is de
            # whipsaw die dit mechanisme duur maakt, en hem wegdefinieren zou
            # de hele meting waardeloos maken.
            cyclus.netto_euro = -2 * instelling.kap * instelling.lot * CONTRACT
            cyclus.netto_euro -= _kosten(instelling, 2)
            cyclus.diepste_euro = cyclus.netto_euro
            cyclus.gehouden = 0
            cyclus.bars = offset + 1
            cyclus.gesloten = m1.index[pos]
            return cyclus

        if long_af or short_af:
            houd = -1 if long_af else 1
            # WAT DE WINNAAR MOET GOEDMAKEN: het verlies van het afgekapte
            # been, de marge die je wil overhouden, en de vier spreads.
            kosten_punten = instelling.spread * 2 * 2
            nodig = (instelling.kap + instelling.marge + kosten_punten
                     if instelling.compenseer else instelling.doel)
            rest_punten, rest_diepste, rest_bars, slot = _loop_een_been(
                m1, pos, entry, houd, instelling, nodig=nodig
            )
            punten = rest_punten - instelling.kap
            cyclus.netto_euro = punten * instelling.lot * CONTRACT - _kosten(instelling, 2)
            cyclus.diepste_euro = (
                (rest_diepste - instelling.kap) * instelling.lot * CONTRACT
                - _kosten(instelling, 2)
            )
            cyclus.gehouden = houd
            cyclus.bars = offset + 1 + rest_bars
            cyclus.gesloten = slot
            return cyclus

    # Geen van beide benen geraakt binnen het venster: plat op de slotkoers.
    pos = min(start + instelling.max_bars, len(m1)) - 1
    slot = float(m1.iloc[pos]["close"])
    # Twee tegengestelde benen salderen tot nul; alleen de kosten blijven over.
    cyclus.netto_euro = -_kosten(instelling, 2)
    cyclus.gehouden = 0
    cyclus.bars = instelling.max_bars
    cyclus.gesloten = m1.index[pos]
    return cyclus


def een_richting_cyclus(
    m1: pd.DataFrame, start: int, *, instelling: Instelling, richting: int
) -> Cyclus | None:
    """DE CONTROLE. Zelfde moment, zelfde doel en stop, maar EEN been.

    Zonder deze vergelijking zegt de straddle-uitslag niets: een positief
    totaal kan net zo goed betekenen dat goud steeg als dat het mechanisme
    werkt. Hier gaat maar EEN spread-paar af in plaats van twee.
    """

    if start >= len(m1):
        return None
    entry = float(m1.iloc[start]["open"])
    stop = instelling.winnaar_stop if instelling.winnaar_stop is not None else instelling.kap
    eigen = Instelling(
        kap=instelling.kap, doel=instelling.doel, winnaar_stop=stop,
        lot=instelling.lot, spread=instelling.spread, max_bars=instelling.max_bars,
    )
    punten, diepste, bars, slot = _loop_een_been(m1, start, entry, richting, eigen)
    return Cyclus(
        geopend=m1.index[start],
        soort="altijd_long" if richting > 0 else "een_richting",
        netto_euro=punten * instelling.lot * CONTRACT - _kosten(instelling, 1),
        diepste_euro=diepste * instelling.lot * CONTRACT - _kosten(instelling, 1),
        gehouden=richting,
        bars=bars,
        gesloten=slot,
    )


def draai(
    m1: pd.DataFrame, *, instelling: Instelling, om_de: int = 60, zaad: int = 7
) -> dict[str, list[Cyclus]]:
    """Straddle en beide controles op exact dezelfde instapmomenten."""

    rng = np.random.default_rng(zaad)
    uit: dict[str, list[Cyclus]] = {"straddle": [], "een_richting": [], "altijd_long": []}
    for start in range(0, len(m1) - 1, om_de):
        s = straddle_cyclus(m1, start, instelling=instelling)
        if s is None:
            continue
        uit["straddle"].append(s)
        # DEZELFDE MOMENTEN, anders vergelijk je twee steekproeven.
        munt = 1 if rng.random() < 0.5 else -1
        eenzijdig = een_richting_cyclus(m1, start, instelling=instelling, richting=munt)
        if eenzijdig is not None:
            eenzijdig.soort = "een_richting"
            uit["een_richting"].append(eenzijdig)
        lang = een_richting_cyclus(m1, start, instelling=instelling, richting=1)
        if lang is not None:
            lang.soort = "altijd_long"
            uit["altijd_long"].append(lang)
    return uit


def rapport(cycli: list[Cyclus]) -> dict[str, object]:
    """Diepste stand en whipsaws voor trefkans, net als bij sectie 20."""

    if not cycli:
        return {"cycli": 0}
    netto = pd.Series([c.netto_euro for c in cycli])
    equity = netto.cumsum()
    return {
        "diepste_cyclus_euro": float(min(c.diepste_euro for c in cycli)),
        "terugval_euro": float((equity.cummax() - equity).max()),
        "beide_afgekapt": int(sum(1 for c in cycli if c.gehouden == 0)),
        "cycli": len(cycli),
        "trefkans": float((netto > 0).mean()),
        "netto_euro": float(netto.sum()),
        "per_cyclus_euro": float(netto.mean()),
        "t": float(netto.mean() / netto.std(ddof=1) * np.sqrt(len(netto)))
        if len(netto) > 1 and netto.std(ddof=1) > 0 else float("nan"),
    }


def uur_permutatie(
    cycli: list[Cyclus], *, rondes: int = 1000, zaad: int = 11
) -> dict[str, float]:
    """HET BESTE UUR, GETOETST TEGEN ZUIVER TOEVAL.

    Het beste van vierentwintig uren is het maximum van vierentwintig ruizige
    getallen -- dat ziet er altijd goed uit, ook als er niets is. Deze toets
    gooit dezelfde uitslagen willekeurig over de uren en kijkt hoe goed het
    beste uur er dan uitziet. Het echte beste uur moet dat verslaan, anders is
    het een selectie en geen vondst.
    """

    if len(cycli) < 24:
        return {"echt": float("nan"), "p": float("nan")}
    waarden = np.array([c.netto_euro for c in cycli])
    uren = np.array([c.geopend.hour for c in cycli])
    echt = max(waarden[uren == u].mean() for u in np.unique(uren))

    rng = np.random.default_rng(zaad)
    beter = 0
    for _ in range(rondes):
        door = rng.permutation(waarden)
        toevallig = max(door[uren == u].mean() for u in np.unique(uren))
        if toevallig >= echt:
            beter += 1
    return {"echt": float(echt), "p": (beter + 1) / (rondes + 1)}


def per_groep(cycli: list[Cyclus], sleutel) -> pd.DataFrame:
    if not cycli:
        return pd.DataFrame()
    frame = pd.DataFrame(
        [{"groep": sleutel(c), "netto": c.netto_euro} for c in cycli]
    )
    return frame.groupby("groep").agg(
        cycli=("netto", "size"),
        netto=("netto", "sum"),
        per_cyclus=("netto", "mean"),
        trefkans=("netto", lambda s: float((s > 0).mean())),
    ).sort_values("netto", ascending=False)


def rooster() -> list[Instelling]:
    """De 48 configuraties uit de vooraf vastgelegde hypothese."""

    # DE MARGE IS DE KNOP, NIET HET DOEL. Bij de compenseer-regel bepaalt de
    # kap hoeveel de winnaar moet goedmaken; wat je zelf kiest is hoeveel winst
    # je daarbovenop wil voordat je plat gaat.
    return [
        Instelling(kap=kap, marge=marge, winnaar_stop=stop, compenseer=True)
        for kap in (2.0, 5.0, 10.0, 20.0)
        for marge in (1.0, 2.0, 5.0, 10.0)
        for stop in (None, 10.0, 20.0)
    ]


def _haal_uit_mt5(args) -> pd.DataFrame:
    """Dezelfde aansluiting als sectie 20, en om dezelfde reden via broker_symbol."""

    from datetime import datetime, timedelta

    from backtesting.replay import fetch_mt5_history
    from config.loader import load_credentials, load_settings, terminal_path_from_env
    from core.mt5_connector import MT5Connector
    from core.types import Timeframe

    settings = load_settings(overlay=args.config, env_overrides=False)
    symbool = settings.instruments.broker_symbol(args.symbol)
    eind = datetime.now(UTC)
    start = eind - timedelta(days=args.days)
    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=False),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    connector.connect()
    try:
        return fetch_mt5_history(connector, symbool, Timeframe.M1, start, eind)
    finally:
        connector.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--om-de", type=int, default=60,
                        help="minuten tussen instapmomenten")
    parser.add_argument("--alle-configs", action="store_true")
    parser.add_argument("--config", default="config/eightcap.yaml")
    parser.add_argument("--csv", help="uitgevoerde M1-bars in plaats van MT5")
    args = parser.parse_args()

    if args.csv:
        m1 = _lees_csv(args.csv)
    else:
        m1 = _haal_uit_mt5(args)

    configs = rooster() if args.alle_configs else [Instelling()]
    print(f"\n  SECTIE 21 -- straddle op {args.symbol}, {args.days} dagen")
    print(f"  configuraties: {len(configs)}   instap elke {args.om_de} minuten")
    print("  " + "-" * 78)
    for instelling in configs:
        uit = draai(m1, instelling=instelling, om_de=args.om_de)
        regels = []
        for soort in ("straddle", "een_richting", "altijd_long"):
            r = rapport(uit[soort])
            regels.append(f"{soort[:12]:>12s} {r.get('per_cyclus_euro', 0):+7.2f}")
        s, e = rapport(uit["straddle"]), rapport(uit["een_richting"])
        beter = s.get("per_cyclus_euro", 0) > e.get("per_cyclus_euro", 0)
        print(f"  {instelling.naam:26s} " + "  ".join(regels) +
              ("   STRADDLE WINT" if beter else ""))

    if len(configs) == 1:
        s = rapport(uit["straddle"])
        print(f"\n  STRADDLE: {s['cycli']} cycli, trefkans {s['trefkans']:.1%}, "
              f"netto EUR {s['netto_euro']:.2f}, t {s['t']:+.2f}")
        print(f"  beide benen afgekapt: {s['beide_afgekapt']}  "
              f"terugval EUR {s['terugval_euro']:.2f}")
        for naam, sleutel in (
            ("JAAR", lambda c: c.geopend.year),
            ("SESSIE", lambda c: _sessie_van(c.geopend)),
            ("UUR (UTC)", lambda c: c.geopend.hour),
        ):
            print(f"\n  PER {naam}")
            print(per_groep(uit["straddle"], sleutel).to_string())
        toets = uur_permutatie(uit["straddle"])
        print(f"\n  BESTE UUR: EUR {toets['echt']:+.2f} per cyclus, "
              f"p = {toets['p']:.3f}")
        print("  p boven 0,05 betekent: dit uur is het maximum van 24 ruizige")
        print("  getallen en geen vondst.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
