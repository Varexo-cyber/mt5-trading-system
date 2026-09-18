"""De zoeklus: meten, bijstellen, opnieuw meten -- tot er niets beters komt.

WAT DE EIGENAAR VROEG, letterlijk: "meten, waar ging het mis, laat me dat
aanpassen, opnieuw meten, nog niet tevreden, weer aanpassen, honderdduizend
keer." Dat is geen gesprek maar een programma, en dit is dat programma.

HOE HIJ ZOEKT. Coordinaatsgewijs klimmen: hij verandert steeds EEN instelling,
meet, en houdt de verandering alleen als ze de vorige stand verslaat. Als geen
enkele losse verandering nog helpt, is hij klaar. Dat is bewust niet slimmer --
een slimmere zoeker vindt sneller iets dat alleen op DEZE data werkt.

EN NU HET BELANGRIJKSTE, WANT ZONDER DIT IS DE HELE LUS SCHADELIJK.

Een zoeker die honderd configuraties probeert en de beste kiest, VINDT ALTIJD
IETS. Ook in zuivere ruis. Dat is geen ontdekking maar het maximum van honderd
ruizige getallen, en hoe langer hij zoekt hoe mooier het wordt en hoe minder
het betekent.

Daarom is de data in tweeen geknipt:

    OEFENDEEL (oudste 70%)   hier mag hij zoeken, kiezen, bijstellen
    TOETSDEEL (nieuwste 30%) hier komt hij pas EEN KEER, aan het eind

Het getal dat telt is dat van het TOETSDEEL. Staat daar iets heel anders dan
op het oefendeel, dan heeft de zoeker de oefendata uit zijn hoofd geleerd en
is de winnaar niets waard. Dat is geen theorie: het is de enige manier waarop
je het verschil ziet.

HET AANTAL GEPROBEERDE CONFIGURATIES STAAT IN DE UITSLAG. De beste van
tweehonderd is iets anders dan een ontdekking, en dat getal hoort ernaast.

EN DE DOMME REGEL LOOPT MEE. Goud steeg 71% in dit venster; een long-regel die
dat niet verslaat heeft de stijging gemeten en niet zichzelf.
"""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from scripts.uitvoering import RAW_GOUD

from scripts.section_twenty_pullback_ladder import (
    CONTRACT, KLOKKEN, Instelling as Ladder, _hersample, _lees_csv, _loopt,
    _sessie_van, simuleer as ladder_simuleer, stapel_omhoog,
)
from scripts.section_twentyone_straddle import (
    Instelling as Straddle, straddle_cyclus,
)

#: Uren waarin doorgaans de zware Amerikaanse cijfers vallen (UTC). Grof, maar
#: het punt is dat de zoeker MAG kiezen om ze te mijden en dat we dan zien of
#: dat helpt -- niet dat wij vooraf beslissen dat het helpt.
NIEUWSUREN = (12, 13, 14)

#: DE ZOEKER MAG NIET BUITEN DE EIGEN RISICOGRENS ZOEKEN.
#:
#: Op de eerste run koos hij 5% per trade en kwam op EUR 2,9 miljoen uit. Dat
#: was deels de blinde vlek hierboven, maar ook een fout in de zoekruimte:
#: `config/eightcap.yaml` zet `risk_per_trade_pct: 2.0`, en de risk manager
#: gooit een ForbiddenStrategyError zodra een trade daar overheen gaat.
#:
#: Een zoeker die 5% aanbeveelt, beveelt iets aan dat de live-kant weigert uit
#: te voeren. Dan optimaliseer je een systeem dat niet bestaat. De grens van de
#: rekening is dus ook de grens van de zoektocht.
RISICO = [0.02, 0.005, 0.01, 0.015]


@dataclass
class Uitslag:
    """Wat een configuratie deed. In euro's op de echte balans."""

    trades: int
    eind: float
    start: float
    diepste_terugval: float      # als deel van de piek
    ruine: bool
    per_trade: float

    @property
    def rendement(self) -> float:
        return self.eind / self.start - 1.0 if self.start else 0.0


def _lot_voor(balans: float, risico_punten: float, *, deel: float,
              euro_per_punt: float, stap: float = 0.01) -> float:
    """DE LOTGROOTTE GROEIT MEE, en het minimumlot is een harde vloer.

    Dit is wat er in de eerste meting ontbrak en waarom die EUR 132.847 nep
    was: met een vast lot verdampt je risico terwijl je balans groeit. Jarvis
    doet het andersom -- de positie schaalt met de rekening.

    De vloer blijft: onder 0,01 lot bestaat niet. Op een kleine rekening kun je
    dus NIET binnen je grens blijven, en dat is geen detail maar de reden dat
    kleine rekeningen anders werken dan grote.
    """

    gewenst = deel * balans / max(risico_punten * euro_per_punt / stap, 1e-9)
    return max(stap, int(gewenst / stap) * stap)


def _toegestaan(stamp: pd.Timestamp, *, sessies: tuple[str, ...],
                mijd_nieuws: bool) -> bool:
    if mijd_nieuws and stamp.hour in NIEUWSUREN:
        return False
    return not sessies or _sessie_van(stamp) in sessies


# --------------------------------------------------------------------------
#  De drie secties, elk met dezelfde vorm: instellingen in, Uitslag uit.
# --------------------------------------------------------------------------

def draai_ladder(m1, stapels, cfg: dict, *, balans: float,
                 euro_per_punt: float) -> Uitslag:
    inst = Ladder(stap=cfg["stap"], max_benen=cfg["max_benen"],
                  mandstop_deel=cfg["mandstop"], lot=0.01, spread=cfg["spread"])
    manden, _ = ladder_simuleer(m1, stapels, instelling=inst, balans=balans)
    manden = [m for m in manden
              if _toegestaan(m.geopend, sessies=cfg["sessies"],
                             mijd_nieuws=cfg["mijd_nieuws"])]
    return _naar_uitslag(
        [(m.gesloten or m1.index[-1], m.resultaat_euro, m.diepste_euro) for m in manden],
        balans=balans, risico_punten=cfg["stap"] * 20, deel=cfg["risico"],
        euro_per_punt=euro_per_punt)


def draai_straddle(m1, cfg: dict, *, balans: float, euro_per_punt: float) -> Uitslag:
    inst = Straddle(kap=cfg["kap"], marge=cfg["marge"],
                    winnaar_stop=cfg["winnaar_stop"], compenseer=True,
                    lot=0.01, spread=cfg["spread"])
    rijen = []
    i = 0
    while i < len(m1) - 1:
        if not _toegestaan(m1.index[i], sessies=cfg["sessies"],
                           mijd_nieuws=cfg["mijd_nieuws"]):
            i += cfg["om_de"]
            continue
        c = straddle_cyclus(m1, i, instelling=inst)
        if c is None:
            break
        rijen.append((c.gesloten or c.geopend, c.netto_euro, c.diepste_euro))
        i += max(cfg["om_de"], c.bars)
    return _naar_uitslag(rijen, balans=balans, risico_punten=cfg["kap"] * 2,
                         deel=cfg["risico"], euro_per_punt=euro_per_punt)


def draai_elio(m1, stapels, cfg: dict, *, balans: float,
               euro_per_punt: float) -> Uitslag:
    # In PUNTEN, want deze lus rekent in punten. Spread een keer per rondje;
    # slippage alleen op de stop, want het doel is een limietorder.
    kosten = cfg["spread"]
    slip = RAW_GOUD.slippage
    rijen = []
    i = 20
    while i < len(m1) - 1:
        stamp = m1.index[i]
        if not (_toegestaan(stamp, sessies=cfg["sessies"], mijd_nieuws=cfg["mijd_nieuws"])
                and stapel_omhoog(stapels, stamp)):
            i += cfg["om_de"]
            continue
        entry = float(m1.iloc[i]["open"])
        punten = None
        pos = i
        for offset in range(cfg["max_bars"]):
            pos = min(i + offset, len(m1) - 1)
            bar = m1.iloc[pos]
            if float(bar["low"]) - entry <= -cfg["stop"]:
                punten = -cfg["stop"]
                break
            if float(bar["high"]) - entry >= cfg["doel"]:
                punten = cfg["doel"]
                break
        if punten is None:
            punten = float(m1.iloc[pos]["close"]) - entry
        # Bij een vaste stop is de diepste stand hoogstens die stop.
        geraakt_stop = punten is not None and punten < 0
        rijen.append((m1.index[pos], punten - kosten
                      - (slip if geraakt_stop else 0.0),
                      -cfg["stop"] if punten < 0 else 0.0))
        i = pos + cfg["om_de"]
    return _naar_uitslag(rijen, balans=balans, risico_punten=cfg["stop"],
                         deel=cfg["risico"], euro_per_punt=euro_per_punt,
                         in_punten=True)


def _naar_uitslag(rijen, *, balans, risico_punten, deel, euro_per_punt,
                  in_punten=False) -> Uitslag:
    """Een balans door de tijd, met meegroeiend lot en de ruine als harde stop.

    DE ZWEVENDE STAND MOET MEE, EN DAT VERGAT IK HIER.

    Dit is de duurste fout van de hele zoeklus en een achtergrondrun vond hem:
    de zoeker kwam op EUR 2,9 MILJOEN uit met zogenaamd 4,8% terugval, en koos
    daarbij het hoogste risico dat hij mocht.

    De oorzaak: deze functie keek alleen naar GESLOTEN resultaten. Een grid
    sluit per definitie alleen winnaars -- de verliezers blijven openstaan --
    dus zijn gesloten curve daalt nooit. De zoeker zag geen enkel gevaar,
    koos dus 5% per trade, en vermenigvuldigde 1,05 achtduizend keer met
    zichzelf. Dat is 10^170, en het is geen resultaat maar een blinde vlek.

    Ik had precies dit al gerepareerd in `eindresultaat.py` en vervolgens in
    de zoeklus opnieuw ingebouwd. Vandaar dat elke rij nu haar diepste
    ZWEVENDE stand meeneemt, en dat die meeschaalt met de lotgrootte.
    """

    if not rijen:
        return Uitslag(0, balans, balans, 0.0, False, 0.0)
    huidig = balans
    piek = balans
    diepste = 0.0
    gedaan = 0
    som = 0.0
    for moment, waarde, zwevend in sorted(rijen, key=lambda r: r[0]):
        lot = _lot_voor(huidig, risico_punten, deel=deel, euro_per_punt=euro_per_punt)
        factor = lot / 0.01
        winst = (waarde * euro_per_punt * factor) if in_punten else (waarde * factor)
        onder = (zwevend * euro_per_punt * factor) if in_punten else (zwevend * factor)

        # EERST HOE DIEP HIJ ONDERWEG STOND, en pas daarna de afrekening.
        # Daar valt een margin call, niet op het eindbedrag.
        laagste = huidig + min(0.0, onder)
        if laagste <= 0:
            return Uitslag(gedaan + 1, 0.0, balans, 1.0, True,
                           som / max(gedaan + 1, 1))
        diepste = max(diepste, (piek - laagste) / piek)

        huidig += winst
        som += winst
        gedaan += 1
        if huidig <= 0:
            return Uitslag(gedaan, 0.0, balans, 1.0, True, som / gedaan)
        piek = max(piek, huidig)
        diepste = max(diepste, (piek - huidig) / piek)
    return Uitslag(gedaan, huidig, balans, diepste, False, som / gedaan)


# --------------------------------------------------------------------------
#  De zoeklus
# --------------------------------------------------------------------------

def punten(u: Uitslag, *, min_trades: int) -> float:
    """WAAROP GEKOZEN WORDT, en dat is niet de winst.

    Kiezen op eindbedrag levert de configuratie op die op DEZE data het meest
    gokte. Daarom drie dempers:

      * een ruine is altijd de slechtste uitkomst, ongeacht wat ervoor kwam
      * te weinig trades telt niet mee -- vier goede trades is geen bewijs
      * de terugval gaat van het rendement af, want een rekening die 80%
        onderwater ging heb je in het echt gesloten
    """

    if u.ruine or u.trades < min_trades:
        return -1e9
    return u.rendement - 3.0 * u.diepste_terugval


def zoek(meet, ruimte: dict[str, list], *, min_trades: int, rondes: int,
         etiket: str) -> tuple[dict, Uitslag, int]:
    """Coordinaatsgewijs klimmen: per ronde EEN instelling tegelijk."""

    huidig = {k: v[0] for k, v in ruimte.items()}
    beste_u = meet(huidig)
    beste_p = punten(beste_u, min_trades=min_trades)
    geprobeerd = 1
    print(f"  [{etiket}] start: {beste_u.eind:,.2f} EUR  ({beste_p:+.3f} punten)",
          flush=True)

    for ronde in range(1, rondes + 1):
        verbeterd = False
        for sleutel, opties in ruimte.items():
            for waarde in opties:
                if waarde == huidig[sleutel]:
                    continue
                kandidaat = dict(huidig, **{sleutel: waarde})
                u = meet(kandidaat)
                geprobeerd += 1
                p = punten(u, min_trades=min_trades)
                if p > beste_p + 1e-9:
                    huidig, beste_u, beste_p = kandidaat, u, p
                    verbeterd = True
                    print(f"  [{etiket}] ronde {ronde}: {sleutel} -> {waarde}   "
                          f"{u.eind:,.2f} EUR  ({p:+.3f})", flush=True)
        if not verbeterd:
            print(f"  [{etiket}] ronde {ronde}: niets verbetert meer, klaar",
                  flush=True)
            break
    return huidig, beste_u, geprobeerd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--balans", type=float, required=True)
    p.add_argument("--rondes", type=int, default=6)
    p.add_argument("--min-trades", type=int, default=100)
    p.add_argument("--rapport")
    args = p.parse_args()

    m1 = _lees_csv(args.csv)
    knip = int(len(m1) * 0.70)
    oefen, toets = m1.iloc[:knip], m1.iloc[knip:]
    st_oefen = {n: _hersample(oefen, r) for n, r in KLOKKEN}
    st_toets = {n: _hersample(toets, r) for n, r in KLOKKEN}
    euro_per_punt = 0.01 * CONTRACT

    print(f"\n  {len(m1):,} bars   oefendeel {len(oefen):,} "
          f"({oefen.index[0].date()} t/m {oefen.index[-1].date()})", flush=True)
    print(f"                 toetsdeel {len(toets):,} "
          f"({toets.index[0].date()} t/m {toets.index[-1].date()})", flush=True)
    print(f"  startbalans EUR {args.balans:,.2f}\n", flush=True)

    SESSIES = [(), ("londen",), ("newyork",), ("overlap",),
               ("londen", "newyork", "overlap")]
    families = {
        "sectie 20 ladder": (
            {"stap": [1.0, 0.5, 2.0, 4.0], "max_benen": [10, 5, 20, None],
             "mandstop": [0.05, 0.02, 0.10, None], "spread": [0.16],
             "sessies": SESSIES, "mijd_nieuws": [False, True],
             "risico": RISICO},
            lambda d, s, c, b: draai_ladder(d, s, c, balans=b, euro_per_punt=euro_per_punt),
        ),
        "sectie 21 straddle": (
            {"kap": [5.0, 2.0, 10.0, 20.0], "marge": [2.0, 1.0, 5.0, 10.0],
             "winnaar_stop": [None, 10.0, 20.0], "spread": [0.16],
             "om_de": [60, 30, 120], "sessies": SESSIES,
             "mijd_nieuws": [False, True], "risico": RISICO},
            lambda d, s, c, b: draai_straddle(d, c, balans=b, euro_per_punt=euro_per_punt),
        ),
        "sectie 22 Elio": (
            {"doel": [9.85, 5.0, 15.0, 25.0], "stop": [49.1, 20.0, 35.0, 70.0],
             "max_bars": [60, 30, 120, 240], "spread": [0.16],
             "om_de": [30, 15, 60], "sessies": SESSIES,
             "mijd_nieuws": [False, True], "risico": RISICO},
            lambda d, s, c, b: draai_elio(d, s, c, balans=b, euro_per_punt=euro_per_punt),
        ),
    }

    uit = []
    for naam, (ruimte, fn) in families.items():
        t0 = time.time()
        cfg, u_oefen, n = zoek(
            lambda c: fn(oefen, st_oefen, c, args.balans),
            ruimte, min_trades=args.min_trades, rondes=args.rondes, etiket=naam)
        u_toets = fn(toets, st_toets, cfg, args.balans)
        uit.append((naam, cfg, u_oefen, u_toets, n, time.time() - t0))
        print(f"  [{naam}] {n} configuraties in {time.time() - t0:.0f}s\n", flush=True)

    # De domme regel op het TOETSDEEL, want daar wordt alles mee vergeleken.
    dom_punten = float(toets.iloc[-1]["close"]) - float(toets.iloc[0]["open"])
    dom_eind = args.balans + dom_punten * euro_per_punt

    tekst = ["", "  " + "=" * 78,
             f"  ZOEKTOCHT  --  start EUR {args.balans:,.2f}", "  " + "=" * 78, ""]
    for naam, cfg, uo, ut, n, secs in uit:
        tekst.append(f"  {naam}   ({n} configuraties geprobeerd)")
        tekst.append(f"    gekozen: " + ", ".join(
            f"{k}={v}" for k, v in cfg.items() if k not in ("spread",)))
        tekst.append(f"    OEFENDEEL  EUR {uo.eind:>12,.2f}  {uo.rendement:+9.1%}  "
                     f"{uo.trades:>6,} trades  terugval {uo.diepste_terugval:>5.1%}"
                     + ("  RUINE" if uo.ruine else ""))
        tekst.append(f"    TOETSDEEL  EUR {ut.eind:>12,.2f}  {ut.rendement:+9.1%}  "
                     f"{ut.trades:>6,} trades  terugval {ut.diepste_terugval:>5.1%}"
                     + ("  RUINE" if ut.ruine else ""))
        tekst.append("")
    tekst += [
        "  " + "-" * 78,
        f"  goud kopen en niks doen (toetsdeel)   EUR {dom_eind:>12,.2f}  "
        f"{dom_eind / args.balans - 1:+9.1%}",
        "",
        "  LEES ALLEEN DE TOETSDEEL-REGEL.",
        "",
        "  Op het oefendeel MOCHT de zoeker kiezen, bijstellen en opnieuw",
        "  proberen. Dat getal is per definitie mooi -- hij heeft net honderden",
        "  configuraties bekeken en de beste eruit gehaald. Dat is het maximum",
        "  van honderden ruizige getallen en geen voorspelling.",
        "",
        "  Het toetsdeel heeft hij EEN KEER gezien, aan het eind. Staat daar",
        "  iets heel anders, dan heeft hij de oefendata uit zijn hoofd geleerd.",
        "",
        "  En verslaat een sectie de domme regel niet, dan heeft ze de",
        "  goudstijging gemeten en niet zichzelf.",
        "",
    ]
    for r in tekst:
        print(r, flush=True)
    if args.rapport:
        with open(args.rapport, "a", encoding="utf-8") as f:
            f.write("\n".join(tekst) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
