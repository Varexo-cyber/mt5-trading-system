"""Koop en verkoop tegelijk, sluit er een, hou de ander vast. HOE LANG?

DE VRAAG, letterlijk zoals hij gesteld is: je zet tegelijk een koop en een
verkoop neer. Eentje gaat eruit. De ander hou je vast "totdat je verlies
compenseert en nog meer winst overhoudt dan je verloren hebt". Hoe lang duurt
dat?

Dat is een andere vraag dan sectie 21 beantwoordde. Die gaf trefkans en
verwachting; dit geeft de WACHTTIJD, en bij dit mechanisme is de wachttijd het
hele verhaal. Een regel die gemiddeld veertig minuten wacht is verhandelbaar;
een die gemiddeld drie weken wacht is dat niet, ook al klopt de rekensom -- in
drie weken beweegt goud honderden punten en dan bepaalt je marge of je er nog
bent.

TWEE LEZINGEN, EN ZE WORDEN ALLEBEI GEMETEN, want de omschrijving liet allebei
toe en ernaar raden is deze week al te vaak fout gegaan:

  A. SLUIT DE VERLIEZER, HOU DE WINNAAR. Het been dat `kap` punten onder water
     staat gaat eruit; de winnaar loopt door tot de mand samen `kap + marge`
     punten in de plus staat.

  B. SLUIT DE WINNAAR, HOU DE VERLIEZER. Het been dat `kap` punten in de plus
     staat wordt gecasht; de verliezer blijft staan tot hij terugkomt en de mand
     alsnog `marge` punten overhoudt.

WAT ER GERAPPORTEERD WORDT, en in deze volgorde:

  * hoe vaak het doel NOOIT gehaald wordt binnen de data -- dat getal eerst,
    want een gemiddelde over alleen de geslaagde wachttijden is overlevings-
    selectie in zuivere vorm
  * de wachttijd: mediaan, 90e percentiel en langste
  * hoe diep je onderweg onder water stond, in euro op 0,01 lot
  * wat dat betekent voor een rekening van EUR 59,16: de marge groeit niet,
    maar het zwevende verlies wel, en de broker gooit je eruit op 50%

    python scripts/hoelang_holden.py --csv runtime/xauusd_m1.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.robot import CONTRACT, EURUSD
from scripts.uitvoering import Uitvoering, laad as laad_uitvoering
from scripts.voorsprong import lees_csv

#: Hoeveel punten een been moet bewegen voordat er een wordt gesloten.
KAPPEN: tuple[float, ...] = (2.0, 5.0, 10.0)

#: Hoeveel punten winst de mand uiteindelijk moet overhouden.
MARGE = 2.0

#: Om de hoeveel minuten er een nieuwe cyclus wordt gestart. Ruim genoeg dat
#: cycli elkaar niet overlappen in de telling.
OM_DE = 240


@dataclass
class Cyclus:
    geopend: pd.Timestamp
    #: +1 = de long bleef staan, -1 = de short bleef staan.
    gehouden: int
    #: Minuten vanaf openen tot het doel gehaald werd. None = nooit gehaald.
    minuten: int | None
    #: Diepste stand van de HELE mand onderweg, in punten.
    diepste_punten: float


def _cyclus(opens: np.ndarray, hoog: np.ndarray, laag: np.ndarray,
            start: int, *, kap: float,
            marge: float, sluit_verliezer: bool,
            max_bars: int) -> Cyclus | None:
    """Eén koop-en-verkoop tegelijk, tot het doel of tot de tijd op is.

    BEIDE BENEN OP DEZELFDE PRIJS. Long en short op de open van dezelfde bar,
    dus de mand staat op precies nul minus de kosten. Vanaf daar is het verschil
    tussen de twee benen altijd gelijk aan de beweging van de prijs: wat de een
    verliest, wint de ander. Pas als er EEN wordt gesloten ontstaat er een
    positie met richting, en pas dan kan er iets verdiend of verloren worden.
    """

    # DE OPEN VAN DE BAR, want daarop stap je in. Hier stond `laag[start]` en
    # dat is de gunstigste prijs van die bar -- gratis voordeel op elke cyclus.
    entry = opens[start]
    diepste = 0.0
    for offset in range(max_bars):
        i = start + offset
        if i >= len(hoog):
            return None
        # De long staat er het slechtst voor op de low, de short op de high.
        long_stand = laag[i] - entry
        short_stand = entry - hoog[i]
        diepste = min(diepste, long_stand, short_stand)

        if sluit_verliezer:
            # LEZING A: het been dat `kap` onder water staat gaat eruit.
            if long_stand <= -kap:
                houd, verlies = -1, kap
            elif short_stand <= -kap:
                houd, verlies = +1, kap
            else:
                continue
        else:
            # LEZING B: het been dat `kap` in de PLUS staat wordt gecasht.
            long_winst = hoog[i] - entry
            short_winst = entry - laag[i]
            if long_winst >= kap:
                houd, verlies = -1, -kap      # gecashte winst, dus negatief
            elif short_winst >= kap:
                houd, verlies = +1, -kap
            else:
                continue

        # En nu wachten tot de mand samen `marge` punten overhoudt, NA aftrek
        # van wat het gesloten been kostte.
        for na in range(offset, max_bars):
            j = start + na
            if j >= len(hoog):
                return Cyclus(pd.NaT, houd, None, diepste)
            best = (hoog[j] - entry) if houd > 0 else (entry - laag[j])
            slechtst = (laag[j] - entry) if houd > 0 else (entry - hoog[j])
            diepste = min(diepste, slechtst - max(verlies, 0.0))
            # De mand = het overgebleven been MIN wat het gesloten been kostte.
            if best - verlies >= marge:
                return Cyclus(pd.NaT, houd, na + 1, diepste)
        return Cyclus(pd.NaT, houd, None, diepste)
    return None


def meet(m1: pd.DataFrame, *, kap: float, marge: float, sluit_verliezer: bool,
         om_de: int, max_bars: int) -> list[Cyclus]:
    opens = m1["open"].to_numpy(dtype=float)
    hoog = m1["high"].to_numpy(dtype=float)
    laag = m1["low"].to_numpy(dtype=float)
    index = m1.index
    uit: list[Cyclus] = []
    for start in range(0, len(m1) - 1, om_de):
        c = _cyclus(opens, hoog, laag, start, kap=kap, marge=marge,
                    sluit_verliezer=sluit_verliezer, max_bars=max_bars)
        if c is not None:
            uit.append(Cyclus(index[start], c.gehouden, c.minuten,
                              c.diepste_punten))
    return uit


def _uren(minuten: float) -> str:
    if minuten < 60:
        return f"{minuten:,.0f} min"
    if minuten < 1440:
        return f"{minuten / 60:,.1f} uur"
    return f"{minuten / 1440:,.1f} dagen"


def rapport(m1: pd.DataFrame, *, balans: float, uitvoering: Uitvoering,
            eurusd: float, max_bars: int) -> list[str]:
    regels = [
        "",
        "  " + "=" * 84,
        "   HOE LANG MOET JE HOLDEN?",
        "  " + "=" * 84,
        "",
        f"   {len(m1):,} M1-bars   {m1.index[0]:%Y-%m-%d} t/m {m1.index[-1]:%Y-%m-%d}",
        f"   0,01 lot per been, een cyclus per {OM_DE} minuten, "
        f"maximaal {_uren(max_bars)} wachten",
        f"   doel: de mand houdt {MARGE:g} punten over na het gesloten been",
        "",
    ]

    for naam, sluit_verliezer in (("A. sluit de VERLIEZER, hou de winnaar", True),
                                  ("B. sluit de WINNAAR, hou de verliezer", False)):
        regels += ["", f"  {naam}", ""]
        regels.append(
            f"  {'kap':>5}  {'cycli':>7}  {'nooit gehaald':>14}  {'mediaan':>10}  "
            f"{'90%':>10}  {'langste':>10}  {'diepst EUR':>11}  {'EUR 59 dood':>12}")
        regels.append("  " + "-" * 92)
        for kap in KAPPEN:
            cycli = meet(m1, kap=kap, marge=MARGE, sluit_verliezer=sluit_verliezer,
                         om_de=OM_DE, max_bars=max_bars)
            if not cycli:
                continue
            gehaald = [c.minuten for c in cycli if c.minuten is not None]
            nooit = len(cycli) - len(gehaald)
            diepste_punten = min(c.diepste_punten for c in cycli)
            # Op 0,01 lot is een punt EUR 1 / eurusd.
            diepste_eur = diepste_punten * 0.01 * CONTRACT / eurusd
            # Wanneer gaat EUR 59 eraan? Marge voor twee benen, en het zwevende
            # verlies van de mand erbij.
            marge_eur = uitvoering.marge_voor(0.01, float(m1.iloc[0]["open"]),
                                              benen=2) / eurusd
            dood = sum(
                1 for c in cycli
                if uitvoering.vliegt_eruit(
                    balans + c.diepste_punten * 0.01 * CONTRACT / eurusd, marge_eur))
            regels.append(
                f"  {kap:>5.0f}  {len(cycli):>7,}  "
                f"{nooit:>6,} ({nooit / len(cycli):>4.0%})  "
                f"{_uren(float(np.median(gehaald))) if gehaald else '-':>10}  "
                f"{_uren(float(np.percentile(gehaald, 90))) if gehaald else '-':>10}  "
                f"{_uren(float(max(gehaald))) if gehaald else '-':>10}  "
                f"{diepste_eur:>11,.2f}  {dood:>5,} ({dood / len(cycli):>4.0%})")

    regels += [
        "",
        "  " + "-" * 84,
        "   'NOOIT GEHAALD' STAAT VOORAAN EN DAT IS MET OPZET. Een gemiddelde over",
        "   alleen de cycli die het doel HAALDEN is overlevingsselectie: precies de",
        "   fout die sectie 20 twee jaar lang winstgevend liet lijken. De cycli die",
        "   nooit terugkwamen zijn juist degene waar het antwoord in zit.",
        "",
        "   'EUR 59 DOOD' is hoeveel cycli je rekening onderweg zouden hebben",
        f"   opgeblazen bij een startbalans van EUR {balans:,.2f} -- gemeten op de",
        "   diepste stand van de mand, met de stop-out op "
        f"{uitvoering.stop_out_niveau:.0%}.",
        "",
    ]
    return regels


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--balans", type=float, default=59.16)
    p.add_argument("--eurusd", type=float, default=EURUSD)
    p.add_argument("--max-dagen", type=float, default=30.0,
                   help="hoe lang er maximaal gewacht wordt voordat 'nooit' geldt")
    p.add_argument("--db", default="runtime/journal.db")
    p.add_argument("--uit", default="runtime/holden.txt")
    args = p.parse_args()

    m1 = lees_csv(args.csv)
    uitv = laad_uitvoering(args.db)
    regels = rapport(m1, balans=args.balans, uitvoering=uitv,
                     eurusd=args.eurusd,
                     max_bars=int(args.max_dagen * 1440))
    for r in regels:
        print(r, flush=True)
    if args.uit:
        Path(args.uit).parent.mkdir(parents=True, exist_ok=True)
        Path(args.uit).write_text("\n".join(regels) + "\n", encoding="utf-8")
        print(f"  Ook opgeslagen in {args.uit}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
