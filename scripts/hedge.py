"""De hedge, als complete rekening: koop + verkoop, sluit er een, hou de ander.

DE REGEL, zoals de eigenaar hem beschrijft:

    Je zet tegelijk een koop en een verkoop neer. Het been dat in verlies loopt
    sluit je. Het been dat goed loopt hou je vast tot de mand het verlies van
    dat gesloten been heeft goedgemaakt plus wat winst. Dan plat.

EN DE TOEVOEGING DIE HIJ ZELF AANDROEG, en die is de kern van dit bestand:
"zorg dat hij wat eerder uitstapt bij verliezen, en wanneer hij denkt hé, deze
gaat toch niet meer door -- dan eruit." Dat is `opgeven_na`.

WAAROM DAT ZOVEEL UITMAAKT. Zonder opgeeftijd wachtte de vorige meting tot het
doel gehaald werd, hoe lang dat ook duurde. Mediaan 49 minuten, maar de langste
27,6 dagen -- en juist die staart blies de rekening op. De mediaan van de
diepste stand was EUR 5,27 en het allerdiepste EUR 1.003,59. Tweehonderd keer
de mediaan. Opgeven kost je een klein verlies; niet opgeven kost je de rekening.

WAT HIER ANDERS IS DAN `hoelang_holden.py`. Dat bestand telde WACHTTIJD en
sterfte. Dit telt GELD: een rekening die loopt, met kosten, met marge, met de
stop-out, en met de opgeeftijd die een verlies ook echt realiseert. Zodra je
mag opgeven is de vraag niet meer "hoe vaak haalt hij het doel" maar "wat blijft
er onder de streep over".

EN HET AANTAL CYCLI IS EEN UITKOMST, GEEN INSTELLING. De vorige versie startte
elke 240 minuten een cyclus -- zes per dag, en dat getal was van mij. Hier
begint er een nieuwe zodra de vorige plat is, zoals een bot het zou doen.

    python scripts/hedge.py --csv runtime/xauusd_m1.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.robot import CONTRACT, EURUSD
from scripts.uitvoering import RAW_GOUD, Uitvoering, laad as laad_uitvoering
from scripts.voorsprong import TOETSDEEL, lees_csv

#: Startbalansen die de eigenaar noemde als "wat ik kwijt kan".
BALANSEN: tuple[float, ...] = (59.16, 150.0, 200.0, 250.0, 300.0)

#: Opgeeftijden in minuten. `None` = wachten tot het doel, hoe lang ook -- dat
#: is de variant zonder opgeefregel, en die staat erbij om te laten zien wat het
#: opgeven oplevert.
OPGEVEN: tuple[int | None, ...] = (60, 240, 720, 1440, 4320, None)

#: De instapdeuren die de eigenaar noemde: altijd, elke nieuwe uurcandle,
#: of alleen bij hoog volume.
DEUREN: tuple[str, ...] = ("altijd", "uur", "volume")


@dataclass(frozen=True)
class Regels:
    #: Punten verlies waarbij het tegendraadse been eruit gaat.
    kap: float = 5.0
    #: Punten winst die de mand uiteindelijk moet overhouden.
    marge: float = 2.0
    #: Minuten. Daarna gaat de mand plat, wat de stand ook is. None = nooit.
    opgeven_na: int | None = 240
    #: WANNEER HIJ MAG INSTAPPEN. "altijd" = zodra hij plat is. "uur" = alleen
    #: op de eerste minuut van een nieuw uur. "volume" = alleen als de VORIGE
    #: bar meer tikken had dan de drempel.
    #:
    #: DE VORIGE BAR, EN DAT IS GEEN DETAIL. Het volume van de bar waarop je
    #: instapt ken je pas als die bar voorbij is. Daarop filteren is dezelfde
    #: vooruitkijkfout die de klokkenstapel EUR 398.572 liet "verdienen".
    instap: str = "altijd"
    #: Percentiel van het tikvolume waarboven "hoog volume" begint.
    volume_percentiel: float = 80.0
    lot: float = 0.01
    uitvoering: Uitvoering = RAW_GOUD
    eurusd: float = EURUSD


@dataclass
class Uitkomst:
    balans: float
    opgeven_na: int | None
    start: float
    eind: float
    cycli: int = 0
    raak: int = 0
    opgegeven: int = 0
    dagen: float = 0.0
    diepste_terugval: float = 0.0
    ruine: bool = False
    ruine_op: pd.Timestamp | None = None
    #: Wat elke cyclus opleverde, met reden. Om te kunnen ZIEN waar het
    #: geld heen ging in plaats van het uit het eindbedrag te raden.
    logboek: list[tuple] = field(default_factory=list)

    @property
    def per_dag(self) -> float:
        return self.cycli / self.dagen if self.dagen else 0.0

    @property
    def trefkans(self) -> float:
        return self.raak / self.cycli if self.cycli else 0.0


def draai(m1: pd.DataFrame, *, regels: Regels, balans: float) -> Uitkomst:
    """Eén rekening, cyclus na cyclus, tot het eind of tot de broker ingrijpt."""

    opens = m1["open"].to_numpy(dtype=float)
    hoog = m1["high"].to_numpy(dtype=float)
    laag = m1["low"].to_numpy(dtype=float)
    index = m1.index
    n = len(m1)

    # DE INSTAPDEUR, een keer vooraf uitgerekend.
    if regels.instap == "uur":
        mag_in = np.asarray(index.minute == 0)
    elif regels.instap == "volume":
        if "tick_volume" not in m1.columns:
            raise SystemExit("  Deze CSV heeft geen tick_volume-kolom.")
        vorige = m1["tick_volume"].shift(1)
        drempel = float(np.nanpercentile(vorige.dropna(), regels.volume_percentiel))
        mag_in = (vorige >= drempel).to_numpy()
    else:
        mag_in = np.ones(n, dtype=bool)

    uitv = regels.uitvoering
    lot = regels.lot
    per_punt = lot * CONTRACT / regels.eurusd            # euro per punt, per been
    # Twee benen open, elk hun eigen spread en slippage bij het sluiten.
    kosten_cyclus = (uitv.kosten_per_been(lot) + uitv.slippage_kosten(lot)) \
        * 2 / regels.eurusd
    marge_twee = uitv.marge_voor(lot, float(opens[0]), benen=2) / regels.eurusd

    uit = Uitkomst(balans=balans, opgeven_na=regels.opgeven_na,
                   start=balans, eind=balans,
                   dagen=(index[-1] - index[0]).total_seconds() / 86400.0)
    stand = balans
    piek = balans
    i = 0
    grens = regels.opgeven_na if regels.opgeven_na is not None else n

    while i < n - 1:
        # WACHTEN TOT DE DEUR OPENGAAT. Bij "altijd" staat hij altijd open.
        while i < n - 1 and not mag_in[i]:
            i += 1
        if i >= n - 1:
            break
        entry = opens[i]
        houd = 0
        verlies = 0.0
        klaar = False

        for offset in range(grens):
            j = i + offset
            if j >= n:
                break

            if houd == 0:
                # BEIDE BENEN STAAN NOG. Wat de een verliest wint de ander, dus
                # de mand staat vlak; alleen de kosten lopen. Sluiten gebeurt
                # zodra een been `kap` punten onder water staat.
                long_stand = laag[j] - entry
                short_stand = entry - hoog[j]
                if long_stand <= -regels.kap:
                    houd, verlies = -1, regels.kap
                elif short_stand <= -regels.kap:
                    houd, verlies = +1, regels.kap
                # DE MARGE VAN TWEE BENEN TELT WEL, ook al is de mand vlak.
                # Dat is de reden dat een kleine rekening dit niet kan dragen.
                if uitv.vliegt_eruit(stand, marge_twee):
                    stand = 0.0
                    uit.ruine, uit.ruine_op = True, index[j]
                    break
                continue

            # ER STAAT NOG EEN BEEN. Nu pas kan de mand bewegen.
            best = (hoog[j] - entry) if houd > 0 else (entry - laag[j])
            slechtst = (laag[j] - entry) if houd > 0 else (entry - hoog[j])

            # 1. DE BROKER EERST. Gemeten op de slechtste stand van deze bar.
            zwevend = (slechtst - verlies) * per_punt
            marge_een = uitv.marge_voor(lot, entry) / regels.eurusd
            if uitv.vliegt_eruit(stand + zwevend, marge_een):
                was = stand
                stand = max(0.0, stand + zwevend - kosten_cyclus)
                uit.logboek.append((index[j], "broker", stand - was))
                uit.cycli += 1
                if stand <= 0:
                    uit.ruine, uit.ruine_op = True, index[j]
                klaar = True
                i = j
                break

            # 2. EN PAS DAARNA: is het doel gehaald?
            if best - verlies >= regels.marge:
                stand += (regels.marge) * per_punt - kosten_cyclus
                uit.logboek.append(
                    (index[j], "doel", regels.marge * per_punt - kosten_cyclus))
                uit.cycli += 1
                uit.raak += 1
                klaar = True
                i = j
                break

        if klaar:
            if uit.ruine:
                break
            piek = max(piek, stand)
            uit.diepste_terugval = max(uit.diepste_terugval, piek - stand)
            i += 1
            continue
        if uit.ruine:
            break

        # OPGEVEN. De tijd is op en het doel is niet gehaald: plat tegen de
        # laatste koers, met het verlies dat er dan staat.
        j = min(i + grens, n - 1)
        if houd == 0:
            # Geen been gesloten: de mand is vlak, alleen de kosten gaan eraf.
            stand -= kosten_cyclus
            uit.logboek.append((index[j], "vlak", -kosten_cyclus))
        else:
            nu = (opens[j] - entry) if houd > 0 else (entry - opens[j])
            stand += (nu - verlies) * per_punt - kosten_cyclus
            uit.logboek.append(
                (index[j], "opgegeven", (nu - verlies) * per_punt - kosten_cyclus))
            uit.opgegeven += 1
        uit.cycli += 1
        if stand <= 0:
            stand = 0.0
            uit.ruine, uit.ruine_op = True, index[j]
            break
        piek = max(piek, stand)
        uit.diepste_terugval = max(uit.diepste_terugval, piek - stand)
        i = j + 1

    uit.eind = stand
    return uit


def _tijd(minuten: int | None) -> str:
    if minuten is None:
        return "nooit"
    if minuten < 60:
        return f"{minuten} min"
    if minuten < 1440:
        return f"{minuten // 60} uur"
    return f"{minuten // 1440} dagen"


def rapport(m1: pd.DataFrame, *, uitvoering: Uitvoering, eurusd: float,
            kap: float, marge: float) -> list[str]:
    grens = int(len(m1) * (1 - TOETSDEEL))
    delen = (("oefendeel", m1.iloc[:grens]), ("TOETSDEEL", m1.iloc[grens:]))

    regels = [
        "",
        "  " + "=" * 92,
        "   DE HEDGE, MET EEN OPGEEFREGEL",
        "  " + "=" * 92,
        "",
        f"   {len(m1):,} M1-bars   {m1.index[0]:%Y-%m-%d} t/m {m1.index[-1]:%Y-%m-%d}",
        f"   0,01 lot per been   been eruit op {kap:g} punten verlies   "
        f"doel {marge:g} punten over",
        f"   broker: spread {uitvoering.spread:g}, slippage {uitvoering.slippage:g}, "
        f"stop-out {uitvoering.stop_out_niveau:.0%}   EURUSD {eurusd:g}",
        "",
        "   Een nieuwe cyclus begint zodra de vorige plat is. Het aantal cycli per",
        "   dag is dus een UITKOMST en geen instelling.",
    ]

    # TABEL 1: de instapdeur en de opgeeftijd, op EUR 250.
    for naam, deel in delen:
        regels += ["", f"  --- {naam}: {deel.index[0]:%Y-%m-%d} t/m "
                       f"{deel.index[-1]:%Y-%m-%d}   (start EUR 250)", ""]
        regels.append(
            f"  {'instap':>8}  {'opgeven na':>10}  {'eind':>11}  {'rendement':>10}  "
            f"{'cycli':>7}  {'/dag':>5}  {'raak':>6}  {'opgegeven':>9}  {'terugval':>9}")
        regels.append("  " + "-" * 96)
        for deur in DEUREN:
            for opgeven_na in OPGEVEN:
                u = draai(deel, balans=250.0, regels=Regels(
                    kap=kap, marge=marge, opgeven_na=opgeven_na, instap=deur,
                    uitvoering=uitvoering, eurusd=eurusd))
                vlag = "  RUINE" if u.ruine else ""
                regels.append(
                    f"  {deur:>8}  {_tijd(opgeven_na):>10}  "
                    f"{u.eind:>11,.2f}  {u.eind / u.start - 1:>+10.1%}  "
                    f"{u.cycli:>7,}  {u.per_dag:>5.1f}  {u.trefkans:>5.1%}  "
                    f"{u.opgegeven:>9,}  {u.diepste_terugval:>9,.2f}{vlag}")
            regels.append("")

    # TABEL 2: wat de startbalans doet, bij vier uur opgeven.
    regels += ["", "  WAT DE STARTBALANS DOET  (opgeven na 4 uur)", ""]
    regels.append(
        f"  {'periode':>10}  {'instap':>8}  {'balans':>8}  {'eind':>11}  "
        f"{'rendement':>10}  {'terugval':>9}")
    regels.append("  " + "-" * 70)
    for naam, deel in delen:
        for deur in DEUREN:
            for balans in BALANSEN:
                u = draai(deel, balans=balans, regels=Regels(
                    kap=kap, marge=marge, opgeven_na=240, instap=deur,
                    uitvoering=uitvoering, eurusd=eurusd))
                vlag = "  RUINE" if u.ruine else ""
                regels.append(
                    f"  {naam:>10}  {deur:>8}  {balans:>8,.0f}  {u.eind:>11,.2f}  "
                    f"{u.eind / u.start - 1:>+10.1%}  "
                    f"{u.diepste_terugval:>9,.2f}{vlag}")
        regels.append("")

    regels += [
        "  " + "-" * 92,
        "   HET TOETSDEEL IS HET ANTWOORD. Op het oefendeel mag gezocht worden.",
        "",
    ]
    return regels


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--kap", type=float, default=5.0)
    p.add_argument("--marge", type=float, default=2.0)
    p.add_argument("--eurusd", type=float, default=EURUSD)
    p.add_argument("--db", default="runtime/journal.db")
    p.add_argument("--uit", default="runtime/hedge.txt")
    args = p.parse_args()

    m1 = lees_csv(args.csv)
    uitv = laad_uitvoering(args.db)
    regels = rapport(m1, uitvoering=uitv, eurusd=args.eurusd,
                     kap=args.kap, marge=args.marge)
    for r in regels:
        print(r, flush=True)
    if args.uit:
        Path(args.uit).parent.mkdir(parents=True, exist_ok=True)
        Path(args.uit).write_text("\n".join(regels) + "\n", encoding="utf-8")
        print(f"  Ook opgeslagen in {args.uit}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
