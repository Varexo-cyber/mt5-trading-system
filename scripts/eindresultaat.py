"""Wat er van je balans over is na 730 dagen -- per sectie en alle drie samen.

DE VRAAG DIE HIER BEANTWOORD WORDT, letterlijk zoals hij gesteld is:

    "730 dagen geleden, als ik sectie 20 runde -- hoeveel zou mijn balans nu
     zijn? En als ik sectie 21? En sectie 22? En dan alles tegelijk?"

Dat is iets anders dan wat de losse secties rapporteren. Die geven statistiek
per mand of per cyclus: trefkans, verwachting, diepste stand. Nuttig om een
regel te BEOORDELEN, maar het is geen bedrag.

WAAROM "ALLE DRIE SAMEN" NIET DE SOM IS. Drie regels op een rekening delen die
rekening. Blaast sectie 20 hem in maand vier op, dan bestaan sectie 21 en 22
vanaf dat moment niet meer -- hoe goed ze op zichzelf ook waren. De gezamenlijke
run is daarom een CHRONOLOGISCHE samenvoeging tegen een balans, met de ruine
als harde stop. Optellen van drie eindbedragen zou een rekening beschrijven die
drie keer bestaat.

EN HIJ PRINT TERWIJL HIJ WERKT. Een scherm dat twintig minuten stilstaat is
niet te onderscheiden van een vastgelopen programma.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scripts.section_twenty_pullback_ladder import (
    KLOKKEN,
    Instelling as LadderInstelling,
    _hersample,
    _lees_csv,
    simuleer as ladder_simuleer,
)
from scripts.section_twentyone_straddle import (
    Instelling as StraddleInstelling,
    draai as straddle_draai,
)


def melding(tekst: str) -> None:
    """Meteen op het scherm, niet pas als de buffer vol is."""

    print(tekst, flush=True)


@dataclass
class Handeling:
    """Een afgesloten resultaat op een moment, in euro.

    `zwevend` is hoe diep deze trade ONDERWEG onder water stond. Dat getal
    hoort erbij en niet als bijzaak -- zie `_loop_balans`.
    """

    moment: pd.Timestamp
    sectie: str
    euro: float
    zwevend: float = 0.0


@dataclass
class Uitkomst:
    naam: str
    start: float
    eind: float
    handelingen: int
    diepste_terugval: float
    diepste_zwevend: float
    ruine: bool
    ruine_op: pd.Timestamp | None = None
    curve: list[float] = field(default_factory=list)

    @property
    def rendement(self) -> float:
        return self.eind / self.start - 1.0


def _loop_balans(
    handelingen: list[Handeling], start: float, naam: str
) -> Uitkomst:
    """Een balans door de tijd, met de ruine als harde stop.

    DE RUINE STOPT ALLES EN DAT IS GEEN DETAIL. Een rekening die op nul komt
    handelt niet verder; de broker sluit de posities. Doorrekenen alsof de
    volgende trade nog kon, is precies hoe een backtest een systeem overleeft
    dat in werkelijkheid weg was.
    """

    balans = start
    piek = start
    diepste = 0.0
    laagst = start
    curve = [start]
    ruine_op = None
    gedaan = 0

    for h in sorted(handelingen, key=lambda x: x.moment):
        # DE ZWEVENDE STAND EERST, EN DIT IS DE BELANGRIJKSTE REGEL HIER.
        #
        # De eerste versie rapporteerde alleen de terugval van de GESLOTEN
        # equity. Bij sectie 20 gaf dat EUR 0,00 over elfduizend manden -- want
        # een grid sluit alleen winnaars, dus die curve loopt kaarsrecht
        # omhoog. Precies de leugen waar sectie 20 tegen gebouwd is, en ik
        # reproduceerde hem een laag hoger.
        #
        # Wat telt is hoe laag je rekening ONDERWEG stond, met alles wat open
        # hing meegerekend. Dat is waar een margin call valt.
        laagst = min(laagst, balans + h.zwevend)
        balans += h.euro
        gedaan += 1
        curve.append(balans)
        if balans <= 0:
            balans = 0.0
            ruine_op = h.moment
            break
        piek = max(piek, balans)
        diepste = max(diepste, piek - balans)

    return Uitkomst(
        naam=naam, start=start, eind=balans, handelingen=gedaan,
        diepste_terugval=diepste, diepste_zwevend=start - laagst,
        ruine=ruine_op is not None, ruine_op=ruine_op, curve=curve,
    )


# ---------------------------------------------------------------- sectie 20

def sectie20(m1: pd.DataFrame, stapels, *, balans: float,
             instelling: LadderInstelling) -> list[Handeling]:
    melding("  sectie 20  de terugval-ladder ...")
    manden, _ = ladder_simuleer(m1, stapels, instelling=instelling, balans=balans)
    melding(f"             {len(manden):,} manden")
    return [
        Handeling(m.gesloten or m1.index[-1], "20", m.resultaat_euro,
                  zwevend=m.diepste_euro)
        for m in manden
    ]


# ---------------------------------------------------------------- sectie 21

def sectie21(m1: pd.DataFrame, *, instelling: StraddleInstelling,
             om_de: int) -> list[Handeling]:
    melding("  sectie 21  de straddle ...")
    uit = straddle_draai(m1, instelling=instelling, om_de=om_de)
    cycli = uit["straddle"]
    melding(f"             {len(cycli):,} cycli")
    return [
        Handeling(c.gesloten or c.geopend, "21", c.netto_euro,
                  zwevend=c.diepste_euro)
        for c in cycli
    ]


# ---------------------------------------------------------------- sectie 22

@dataclass(frozen=True)
class ElioInstelling:
    """Elio's bot als verhandelbare regel, afgeleid uit zijn eigen cijfers.

    UIT ZIJN STATISTIEKEN TERUGGEREKEND, niet verzonnen:

      gemiddelde winst   98,52 pips / $1.502,78
      gemiddeld verlies  491,22 pips / $7.178,30
      -> ongeveer $15 per pip, dus rond 1,5 lot
      -> doel   ongeveer  9,85 punten koers
      -> stop   ongeveer 49,10 punten koers
      gemiddelde tradeduur 21 minuten

    Dat is een KLEIN DOEL MET EEN WIJDE STOP: quitte bij 83,3% trefkans. Precies
    de vorm die eerder in dit project gemeten en goedgekeurd is, dus hier is op
    zichzelf niets mis mee. De vraag is alleen of die 95% er op goud ook uitkomt.
    """

    doel: float = 9.85
    stop: float = 49.10
    lot: float = 0.01
    spread: float = 0.16
    max_bars: int = 60          # zijn gemiddelde duur is 21 minuten
    om_de: int = 30


def sectie22(m1: pd.DataFrame, stapels, *, instelling: ElioInstelling) -> list[Handeling]:
    """Klein doel, wijde stop, in de richting van de stapel.

    DE STOP WINT DE BAR. Raakt een bar zowel het doel als de stop, dan telt de
    stop -- wat er binnen die bar eerst gebeurde weet niemand, en de andere
    aanname tilt precies dit soort regels op naar een trefkans die er niet is.
    """

    from scripts.section_twenty_pullback_ladder import stapel_omhoog

    melding("  sectie 22  Elio: klein doel, wijde stop ...")
    kosten = instelling.spread * 2 * instelling.lot * 100.0
    uit: list[Handeling] = []
    i = 0
    while i < len(m1) - 1:
        stamp = m1.index[i]
        if not stapel_omhoog(stapels, stamp):
            i += instelling.om_de
            continue
        entry = float(m1.iloc[i]["open"])
        punten = None
        for offset in range(instelling.max_bars):
            pos = i + offset
            if pos >= len(m1):
                break
            bar = m1.iloc[pos]
            if float(bar["low"]) - entry <= -instelling.stop:
                punten = -instelling.stop
                break
            if float(bar["high"]) - entry >= instelling.doel:
                punten = instelling.doel
                break
        if punten is None:
            pos = min(i + instelling.max_bars, len(m1)) - 1
            punten = float(m1.iloc[pos]["close"]) - entry
        else:
            pos = min(pos, len(m1) - 1)
        uit.append(Handeling(m1.index[pos], "22",
                             punten * instelling.lot * 100.0 - kosten,
                             zwevend=-instelling.stop * instelling.lot * 100.0
                             if punten < 0 else 0.0))
        i = pos + instelling.om_de
    melding(f"             {len(uit):,} trades")
    return uit


# ---------------------------------------------------------------------------

def regel(u: Uitkomst) -> str:
    vlag = "  RUINE" if u.ruine else "       "
    return (f"  {u.naam:22s}{vlag}  EUR {u.eind:>12,.2f}  {u.rendement:+9.1%}"
            f"  {u.handelingen:>7,} trades"
            f"   laagst onderweg EUR {u.start - u.diepste_zwevend:>10,.2f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--balans", type=float, required=True)
    p.add_argument("--om-de", type=int, default=60)
    p.add_argument("--rapport", help="ook naar dit bestand schrijven")
    args = p.parse_args()

    melding(f"\n  Bars laden uit {args.csv} ...")
    m1 = _lees_csv(args.csv)
    melding(f"  {len(m1):,} M1-bars  van {m1.index[0]}  tot {m1.index[-1]}")
    dagen = (m1.index[-1] - m1.index[0]).days
    melding(f"  dat is {dagen} kalenderdagen\n")

    melding("  Hogere klokken opbouwen uit dezelfde M1 ...")
    stapels = {naam: _hersample(m1, regel_) for naam, regel_ in KLOKKEN}
    melding(f"  {', '.join(stapels)}\n")

    h20 = sectie20(m1, stapels, balans=args.balans, instelling=LadderInstelling())
    h21 = sectie21(m1, instelling=StraddleInstelling(), om_de=args.om_de)
    h22 = sectie22(m1, stapels, instelling=ElioInstelling())

    u20 = _loop_balans(h20, args.balans, "sectie 20 alleen")
    u21 = _loop_balans(h21, args.balans, "sectie 21 alleen")
    u22 = _loop_balans(h22, args.balans, "sectie 22 alleen")
    samen = _loop_balans(h20 + h21 + h22, args.balans, "alle drie samen")

    kop = [
        "",
        "  " + "=" * 74,
        f"   EINDRESULTAAT  --  start EUR {args.balans:,.2f}, {dagen} dagen",
        "  " + "=" * 74,
        "",
    ]
    lijf = [regel(u) for u in (u20, u21, u22, samen)]
    staart = [
        "",
        "  LEES DE LAATSTE KOLOM EERST. Dat is hoe laag je rekening ONDERWEG",
        "  stond met alles wat openhing meegerekend -- daar valt een margin",
        "  call, niet op het eindbedrag. Een grid sluit alleen winnaars, dus",
        "  zijn gesloten curve loopt altijd mooi omhoog.",
        "",
        "  ALLE DRIE SAMEN IS NIET DE SOM VAN DE DRIE, en dat is met opzet.",
        "  Drie regels op een rekening delen die rekening. Blaast er een de",
        "  boel op in maand vier, dan bestaan de andere twee daarna niet meer.",
        "  Daarom lopen ze hier chronologisch door elkaar tegen EEN balans,",
        "  met de ruine als harde stop.",
        "",
    ]
    if samen.ruine:
        staart += [f"  DE REKENING GING OP {samen.ruine_op}.", ""]

    tekst = "\n".join(kop + lijf + staart)
    print(tekst, flush=True)
    if args.rapport:
        with open(args.rapport, "a", encoding="utf-8") as f:
            f.write(tekst + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
