"""Het handelsdagboek: elke trade apart, met bar, reden, verloop en balans.

WAT HIER ANDERS IS DAN IN ALLE ANDERE SECTIES. Die geven TOTALEN -- trefkans,
verwachting, eindbedrag. Nuttig om te beoordelen, maar je ziet er niets van.

Gevraagd is het omgekeerde: "hier zou ik in hebben gestaan, hey jammer man,
verloren." Dus per trade: op welke bar, waarom daar, wat er daarna gebeurde,
wat het kostte of opleverde, en wat je balans toen was.

DRIE DINGEN DIE DIT DAGBOEK MOET KUNNEN, anders is het decoratie:

  * DE REDEN ERBIJ. Niet "koop op 4287,31" maar welke klokken het eens waren
    en hoe diep de terugval was. Zonder de reden kun je achteraf niet zien
    waarom een reeks misging.

  * HET VERLOOP, NIET ALLEEN DE UITKOMST. Hoe diep stond hij onderweg, hoe
    lang duurde het, hoeveel benen kwamen erbij. Een trade die +1 EUR opleverde
    nadat hij -40 EUR onder water stond, is iets heel anders dan een trade die
    rechtstreeks naar +1 liep.

  * DE BALANS ERNAAST. Dezelfde trade is op EUR 59 dodelijk en op EUR 5.000
    een schrammetje, en dat zie je alleen als het bedrag ernaast staat.

BIJ 79.000 MANDEN KUN JE NIET ALLES LEZEN. Alles gaat naar het bestand; op het
scherm komen de eerste twintig, de tien ergste, de tien beste en de laatste
twintig. De ergste tien zijn met opzet de eerste die je te zien krijgt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.section_twenty_pullback_ladder import (
    CONTRACT, KLOKKEN, Instelling as Ladder, _hersample, _lees_csv, _sessie_van,
    simuleer as ladder_simuleer, stapel_omhoog,
)
from scripts.section_twentyone_straddle import (
    Instelling as Straddle, straddle_cyclus,
)
from scripts.zoektocht import _lot_voor


@dataclass
class Regel:
    """Een trade, zoals hij in het dagboek komt te staan."""

    moment: pd.Timestamp
    sectie: str
    reden: str
    verloop: str
    euro: float
    zwevend: float
    balans_na: float
    lot: float

    def __str__(self) -> str:
        # DRIE DECIMALEN, want bij 0,01 lot zijn de bedragen centen en met twee
        # decimalen staat er overal 0,00 -- dan lijkt elke trade hetzelfde.
        teken = "+" if self.euro >= 0 else ""
        oordeel = ("WINST " if self.euro > 0.005 else
                   "JAMMER" if self.euro < -0.005 else "vlak  ")
        return (
            f"{self.moment:%Y-%m-%d %H:%M}  {self.sectie}  {oordeel}  "
            f"{teken}{self.euro:>9.3f} EUR   balans {self.balans_na:>10.2f}   "
            f"lot {self.lot:.2f}\n"
            f"{'':>18}  {self.reden}\n"
            f"{'':>18}  {self.verloop}"
        )


def _klokken_eens(stapels, stamp) -> str:
    omhoog = []
    for naam, frame in stapels.items():
        pos = frame.index.searchsorted(stamp, side="right") - 1
        if pos < 1:
            continue
        rij, vorige = frame.iloc[pos], frame.iloc[pos - 1]
        if rij["close"] > rij["open"] and rij["close"] > vorige["close"]:
            omhoog.append(naam)
    return "+".join(omhoog) if omhoog else "geen"


def dagboek_ladder(m1, stapels, *, balans: float, inst: Ladder,
                   risico: float) -> list[Regel]:
    manden, _ = ladder_simuleer(m1, stapels, instelling=inst, balans=balans)
    uit: list[Regel] = []
    huidig = balans
    for m in manden:
        lot = _lot_voor(huidig, inst.stap * 20, deel=risico, euro_per_punt=1.0)
        factor = lot / 0.01
        euro = m.resultaat_euro * factor
        zwevend = m.diepste_euro * factor
        huidig = max(0.0, huidig + euro)
        duur = m.bars_onder_water
        uit.append(Regel(
            moment=m.geopend, sectie="S20",
            reden=(f"KOOP {m.benen[0]:.2f}   {_klokken_eens(stapels, m.geopend)} "
                   f"omhoog   sessie {_sessie_van(m.geopend)}"),
            verloop=(f"{m.aantal_benen} benen tot {m.benen[-1]:.2f}   "
                     f"diepst {zwevend:+.2f} EUR   {duur} min"
                     + ("   AFGEKAPT door de mandstop" if m.afgekapt else "")
                     + ("   NOG OPEN aan het eind" if m.gesloten is None else "")),
            euro=euro, zwevend=zwevend, balans_na=huidig, lot=lot))
        if huidig <= 0:
            uit[-1].verloop += "   <<< REKENING OP"
            break
    return uit


def dagboek_straddle(m1, *, balans: float, inst: Straddle, risico: float,
                     om_de: int = 60) -> list[Regel]:
    uit: list[Regel] = []
    huidig = balans
    i = 0
    while i < len(m1) - 1 and huidig > 0:
        c = straddle_cyclus(m1, i, instelling=inst)
        if c is None:
            break
        lot = _lot_voor(huidig, inst.kap * 2, deel=risico, euro_per_punt=1.0)
        factor = lot / 0.01
        euro, zwevend = c.netto_euro * factor, c.diepste_euro * factor
        huidig = max(0.0, huidig + euro)
        kant = {1: "LONG bleef staan", -1: "SHORT bleef staan",
                0: "ALLEBEI afgekapt (whipsaw)"}[c.gehouden]
        uit.append(Regel(
            moment=c.geopend, sectie="S21",
            reden=(f"KOOP+VERKOOP op {float(m1.iloc[i]['open']):.2f}   "
                   f"sessie {_sessie_van(c.geopend)}"),
            verloop=(f"{kant}   moest {inst.kap + inst.marge:.1f} pt goedmaken   "
                     f"diepst {zwevend:+.2f} EUR   {c.bars} min"),
            euro=euro, zwevend=zwevend, balans_na=huidig, lot=lot))
        i += max(om_de, c.bars)
    return uit


def dagboek_elio(m1, stapels, *, balans: float, doel: float, stop: float,
                 risico: float, max_bars: int = 60, om_de: int = 30,
                 spread: float = 0.16) -> list[Regel]:
    uit: list[Regel] = []
    huidig = balans
    i = 20
    while i < len(m1) - 1 and huidig > 0:
        stamp = m1.index[i]
        if not stapel_omhoog(stapels, stamp):
            i += om_de
            continue
        entry = float(m1.iloc[i]["open"])
        punten, pos, reden = None, i, "tijd"
        diepst = 0.0
        for offset in range(max_bars):
            pos = min(i + offset, len(m1) - 1)
            bar = m1.iloc[pos]
            diepst = min(diepst, float(bar["low"]) - entry)
            if float(bar["low"]) - entry <= -stop:
                punten, reden = -stop, "STOP"
                break
            if float(bar["high"]) - entry >= doel:
                punten, reden = doel, "doel"
                break
        if punten is None:
            punten = float(m1.iloc[pos]["close"]) - entry
        lot = _lot_voor(huidig, stop, deel=risico, euro_per_punt=1.0)
        factor = lot / 0.01
        euro = (punten - spread * 2) * factor
        huidig = max(0.0, huidig + euro)
        uit.append(Regel(
            moment=stamp, sectie="S22",
            reden=(f"KOOP {entry:.2f}   doel {entry + doel:.2f}   "
                   f"stop {entry - stop:.2f}   sessie {_sessie_van(stamp)}"),
            verloop=(f"uit op {reden}   diepst {diepst * factor:+.2f} EUR   "
                     f"{pos - i + 1} min"),
            euro=euro, zwevend=diepst * factor, balans_na=huidig, lot=lot))
        i = pos + om_de
    return uit


def toon(regels: list[Regel], *, naam: str, balans: float) -> list[str]:
    """DE ERGSTE TIEN STAAN BOVENAAN, en dat is met opzet.

    Een dagboek dat met de mooiste trades begint leest als reclame. Waar je van
    leert is de reeks die misging, dus die komt eerst.
    """

    if not regels:
        return [f"\n  {naam}: geen enkele trade"]
    uit = [f"\n  {'=' * 76}", f"  {naam}  --  {len(regels):,} trades",
           f"  {'=' * 76}"]

    op_euro = sorted(regels, key=lambda r: r.euro)
    uit.append(f"\n  DE TIEN ERGSTE")
    uit += [str(r) for r in op_euro[:10]]
    uit.append(f"\n  DE TIEN BESTE")
    uit += [str(r) for r in reversed(op_euro[-10:])]
    uit.append(f"\n  DE EERSTE TWINTIG (hoe het begon op EUR {balans:,.2f})")
    uit += [str(r) for r in regels[:20]]
    uit.append("\n  DE LAATSTE TWINTIG")
    uit += [str(r) for r in regels[-20:]]

    winst = [r for r in regels if r.euro > 0]
    diepst = min(regels, key=lambda r: r.zwevend)
    uit += [
        f"\n  SAMEN: {len(regels):,} trades, {len(winst):,} in winst "
        f"({len(winst) / len(regels):.1%})",
        f"  balans {balans:,.2f} -> {regels[-1].balans_na:,.2f}",
        f"  diepste zwevende stand: {diepst.zwevend:+,.2f} EUR op "
        f"{diepst.moment:%Y-%m-%d %H:%M}",
    ]
    return uit


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--balans", type=float, required=True)
    p.add_argument("--risico", type=float, default=0.02)
    p.add_argument("--uit", default="runtime/dagboek.txt")
    args = p.parse_args()

    m1 = _lees_csv(args.csv)
    stapels = {n: _hersample(m1, r) for n, r in KLOKKEN}
    print(f"\n  {len(m1):,} bars   {m1.index[0]:%Y-%m-%d} t/m {m1.index[-1]:%Y-%m-%d}",
          flush=True)
    print(f"  startbalans EUR {args.balans:,.2f}   risico {args.risico:.1%}\n",
          flush=True)

    alles: list[str] = []
    for naam, maak in (
        ("SECTIE 20  terugval-ladder",
         lambda: dagboek_ladder(m1, stapels, balans=args.balans,
                                inst=Ladder(), risico=args.risico)),
        ("SECTIE 21  straddle",
         lambda: dagboek_straddle(m1, balans=args.balans, inst=Straddle(),
                                  risico=args.risico)),
        ("SECTIE 22  Elio",
         lambda: dagboek_elio(m1, stapels, balans=args.balans, doel=9.85,
                              stop=49.1, risico=args.risico)),
    ):
        print(f"  {naam} ...", flush=True)
        regels = maak()
        blok = toon(regels, naam=naam, balans=args.balans)
        alles += blok
        for r in blok:
            print(r, flush=True)
        # Het VOLLEDIGE dagboek gaat naar het bestand, ook de 79.000 rijen.
        alles += ["", f"  --- alle {len(regels):,} trades van {naam} ---"]
        alles += [str(r) for r in regels]

    with open(args.uit, "w", encoding="utf-8") as f:
        f.write("\n".join(alles) + "\n")
    print(f"\n  Het volledige dagboek staat in {args.uit}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
