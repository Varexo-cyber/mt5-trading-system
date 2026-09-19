"""Welk uur en welke sessie werkten het best -- uit de dry-run-CSV.

WAAROM DIT LOS STAAT VAN DE DRY RUN. `scripts/dry_run_sections.py` schrijft met
`--csv` elke beslissing weg, met een `when` erbij. De uitsplitsing per uur en per
sessie is daarna puur rekenwerk op dat bestand. Dat scheelt een verbouwing aan
een script van vierduizend regels, en het werkt op elke CSV die je al hebt --
ook eentje die in een andere map is gedraaid.

    python scripts/per_uur_en_sessie.py --csv runtime/sectie7-smc.csv

DE VAL WAAR DIT SOORT UITSPLITSINGEN IN TRAPT, en daarom staat hij hier
ingebouwd. Het beste van vierentwintig uren is het MAXIMUM van vierentwintig
ruizige getallen. Dat ziet er altijd goed uit, ook als er niets aan de hand is.
Deze module husselt de trades daarom duizend keer willekeurig over de uren en
kijkt hoe goed het beste uur er dan uitziet. Is het echte beste uur niet beter
dan die verdeling, dan heb je een selectie te pakken en geen vondst.

Datzelfde geldt voor de sessies, maar met vier hokjes in plaats van
vierentwintig is de val kleiner. Hij staat er toch, want klein is niet nul.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

#: Sessies in UTC, dezelfde als in sectie 20. Ze mogen over middernacht heen
#: lopen en Asia doet dat.
SESSIES: dict[str, tuple[int, int]] = {
    "asia": (23, 8),
    "londen": (7, 16),
    "newyork": (12, 21),
}

#: Welke kolom de uitkomst draagt. `managed_r_LIVE` is wat de rekening
#: werkelijk draait (met break-even); `result_r_fixed_stop` is de vaste stop die
#: het onderzoek mat. Allebei beschikbaar, en de keuze staat in het rapport --
#: anders telt een spreadsheet stilletjes de verkeerde op.
KOLOMMEN = ("managed_r_LIVE", "result_r_fixed_stop")


def sessie_van(uur: int) -> str:
    """Welke sessie dit uur is. Overlap krijgt een eigen naam, want dat is
    waar het meeste volume zit en dat hoort niet weggemiddeld te worden."""

    actief = []
    for naam, (start, eind) in SESSIES.items():
        # Een venster mag over middernacht heen lopen; Asia doet dat (23-8).
        binnen = (start <= uur < eind) if start < eind else (uur >= start or uur < eind)
        if binnen:
            actief.append(naam)
    if "londen" in actief and "newyork" in actief:
        return "overlap"
    return actief[0] if actief else "buiten"


def lees(pad: Path, kolom: str) -> list[tuple[datetime, float]]:
    """Alleen beslissingen die een UITKOMST hebben.

    De CSV bevat ook elke weigering -- dat is met opzet en het is nuttig, maar
    een weigering heeft geen resultaat en hoort dus niet in een gemiddelde.
    """

    uit: list[tuple[datetime, float]] = []
    with pad.open(newline="", encoding="utf-8") as f:
        for rij in csv.DictReader(f):
            ruw = (rij.get(kolom) or "").strip()
            if not ruw:
                continue
            try:
                waarde = float(ruw)
                moment = datetime.fromisoformat(rij["when"])
            except (ValueError, KeyError):
                continue
            uit.append((moment, waarde))
    return uit


def _groep(rijen, sleutel) -> dict[object, list[float]]:
    uit: dict[object, list[float]] = defaultdict(list)
    for moment, waarde in rijen:
        uit[sleutel(moment)].append(waarde)
    return uit


def _tabel(groepen: dict[object, list[float]], *, titel: str,
           kop: str, minimum: int) -> list[str]:
    regels = [
        "",
        f"  {titel}",
        f"  {kop:>10}  {'trades':>7}  {'totaal R':>10}  {'R/trade':>9}  {'trefkans':>9}",
        f"  {'-' * 10}  {'-' * 7}  {'-' * 10}  {'-' * 9}  {'-' * 9}",
    ]
    op_gemiddelde = sorted(
        groepen.items(),
        key=lambda kv: sum(kv[1]) / len(kv[1]) if kv[1] else 0.0,
        reverse=True,
    )
    for naam, waarden in op_gemiddelde:
        n = len(waarden)
        gemiddelde = sum(waarden) / n
        raak = sum(1 for w in waarden if w > 0) / n
        dun = "   (te weinig trades om iets van te vinden)" if n < minimum else ""
        regels.append(
            f"  {str(naam):>10}  {n:>7,}  {sum(waarden):>10.2f}  "
            f"{gemiddelde:>9.4f}  {raak:>8.1%}{dun}")
    return regels


def permutatie(rijen, sleutel, *, rondes: int = 1000,
               zaad: int = 20260919) -> tuple[float, float, int]:
    """Hoe goed is het BESTE hokje, vergeleken met puur toeval?

    Husselt de uitkomsten willekeurig over de hokjes en kijkt elke ronde wat het
    beste hokje dan haalt. De p-waarde is hoe vaak toeval het echte beste getal
    evenaart of overtreft.

    Dit is de enige rem op "kijk, tussen 15 en 16 uur werkt het". Met
    vierentwintig hokjes en een paar honderd trades is een uitschieter de
    normaalste zaak van de wereld.
    """

    groepen = _groep(rijen, sleutel)
    if not groepen:
        return 0.0, 1.0, 0
    echt = max(sum(v) / len(v) for v in groepen.values())

    maten = [len(v) for v in groepen.values()]
    alles = np.array([w for _, w in rijen], dtype=float)
    rng = np.random.default_rng(zaad)
    beter = 0
    for _ in range(rondes):
        rng.shuffle(alles)
        pos, beste = 0, -np.inf
        for maat in maten:
            deel = alles[pos:pos + maat]
            pos += maat
            if len(deel):
                beste = max(beste, float(deel.mean()))
        if beste >= echt:
            beter += 1
    return echt, (beter + 1) / (rondes + 1), len(groepen)


def rapport(pad: Path, *, kolom: str, minimum: int, rondes: int) -> list[str]:
    rijen = lees(pad, kolom)
    if not rijen:
        return [f"\n  Geen enkele rij met een ingevulde `{kolom}` in {pad}.",
                "  Draaide de dry run wel met --csv, en stonden er trades in?"]

    regels = [
        "",
        "  " + "=" * 74,
        f"   PER UUR EN PER SESSIE  --  {pad.name}",
        "  " + "=" * 74,
        "",
        f"   {len(rijen):,} trades met een uitkomst",
        f"   van {min(m for m, _ in rijen):%Y-%m-%d} tot {max(m for m, _ in rijen):%Y-%m-%d}",
        f"   gemeten kolom: {kolom}",
        f"   in totaal {sum(w for _, w in rijen):+.2f} R, "
        f"gemiddeld {sum(w for _, w in rijen) / len(rijen):+.4f} R per trade",
    ]

    regels += _tabel(_groep(rijen, lambda m: f"{m.hour:02d}:00"),
                     titel="PER UUR (UTC)", kop="uur", minimum=minimum)
    regels += _tabel(_groep(rijen, lambda m: sessie_van(m.hour)),
                     titel="PER SESSIE", kop="sessie", minimum=minimum)
    regels += _tabel(_groep(rijen, lambda m: m.strftime("%A")),
                     titel="PER WEEKDAG", kop="dag", minimum=minimum)

    beste_uur, p_uur, n_uur = permutatie(
        rijen, lambda m: m.hour, rondes=rondes)
    beste_ses, p_ses, n_ses = permutatie(
        rijen, lambda m: sessie_van(m.hour), rondes=rondes)

    regels += [
        "",
        "  IS HET BESTE UUR ECHT, OF IS HET HET MAXIMUM VAN RUIS?",
        "",
        f"    beste uur     {beste_uur:+.4f} R/trade   p = {p_uur:.3f}   "
        f"({n_uur} hokjes, {rondes} keer gehusseld)",
        f"    beste sessie  {beste_ses:+.4f} R/trade   p = {p_ses:.3f}   "
        f"({n_ses} hokjes)",
        "",
    ]
    for naam, p in (("uur", p_uur), ("sessie", p_ses)):
        if p > 0.05:
            regels.append(
                f"    p boven 0,05 bij {naam}: dit is niet te onderscheiden van "
                f"toeval.")
            regels.append(
                f"    Op zo'n {naam} gaan filteren is de ruis achterna lopen.")
        else:
            regels.append(
                f"    p onder 0,05 bij {naam}: toeval komt hier zelden aan. "
                f"Waard om op een")
            regels.append(
                f"    apart stuk data te hertoetsen voordat je er een regel van maakt.")
    regels.append("")
    return regels


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="de CSV uit dry_run_sections")
    p.add_argument("--kolom", default=KOLOMMEN[0], choices=KOLOMMEN)
    p.add_argument("--minimum", type=int, default=30,
                   help="onder dit aantal trades wordt een hokje als dun gemarkeerd")
    p.add_argument("--rondes", type=int, default=1000)
    p.add_argument("--uit", default="", help="ook naar dit bestand schrijven")
    args = p.parse_args()

    pad = Path(args.csv)
    if not pad.exists():
        print(f"\n  {pad} bestaat niet.")
        print("  Draai eerst de dry run met --csv, zie de kop van dit bestand.\n")
        return 1

    regels = rapport(pad, kolom=args.kolom, minimum=args.minimum,
                     rondes=args.rondes)
    for r in regels:
        print(r)
    if args.uit:
        Path(args.uit).parent.mkdir(parents=True, exist_ok=True)
        Path(args.uit).write_text("\n".join(regels) + "\n", encoding="utf-8")
        print(f"  Ook opgeslagen in {args.uit}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
