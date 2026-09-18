"""Welke balans heb je nodig, en wat houdt EUR 59 uit.

DIT DEEL HEEFT GEEN MARKTDATA NODIG, en dat is met opzet. Een ladder met vaste
stap heeft na X punten tegenbeweging een voorspelbaar aantal benen op
voorspelbare afstanden. Wat een rekening uithoudt is dus rekenwerk. Een
backtest zou hetzelfde antwoord geven plus de suggestie dat het van het
gekozen venster afhangt -- en dat is juist wat het niet doet.

De andere kant van dezelfde som, en de eerlijkere: hoeveel beweging goud
werkelijk maakt. Een dag is 40 tot 80 punten. Een nieuwsweek 150. De correcties
in de reeks 2024-2025 waren er meerdere van 200 tot 300 punten. Zet dat naast
wat je rekening uithoudt en het antwoord op "hoeveel balans" staat er vanzelf.

    python scripts/welke_balans.py
    python scripts/welke_balans.py --lot 0.01 --stap 0.5 --prijs 4000
"""

from __future__ import annotations

import argparse

from scripts.uitvoering import (
    RAW_GOUD,
    ZONDER_STOPOUT,
    Uitvoering,
    balans_voor_beweging,
    benen_bij_beweging,
    laad,
    overleefde_beweging,
    toestand_na_beweging,
)

#: Wat goud in de praktijk doet, om de rekensom een maat te geven. Dit zijn
#: geen voorspellingen maar ordegroottes waar een grafiek naast kan.
BEWEGINGEN: tuple[tuple[float, str], ...] = (
    (5.0, "een rustig half uur"),
    (10.0, "een gewoon uur"),
    (20.0, "een drukke ochtend"),
    (40.0, "een normale dag"),
    (80.0, "een stevige dag"),
    (150.0, "een nieuwsweek"),
    (300.0, "een correctie zoals in 2024 en 2025"),
)

BALANSEN: tuple[float, ...] = (
    59.16, 100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0,
    10_000.0, 25_000.0, 50_000.0, 100_000.0,
)


def tabel_wat_houdt_een_balans_uit(
    *, stap: float, lot: float, prijs: float, uitvoering: Uitvoering,
) -> list[str]:
    uit = [
        "",
        "  WAT EEN BALANS UITHOUDT",
        f"  ladder van {lot:g} lot, elke {stap:g} punt een been, goud op {prijs:,.0f}",
        "",
        f"  {'balans':>12}  {'houdt uit':>11}  {'benen':>6}  {'= dat is':<38}",
        f"  {'-' * 12}  {'-' * 11}  {'-' * 6}  {'-' * 38}",
    ]
    for balans in BALANSEN:
        punten = overleefde_beweging(
            balans, stap=stap, lot=lot, prijs=prijs, uitvoering=uitvoering)
        benen = benen_bij_beweging(punten, stap) if punten != float("inf") else 0
        maat = next((naam for grens, naam in reversed(BEWEGINGEN)
                     if punten >= grens), "niet eens een rustig half uur")
        uit.append(f"  {balans:>12,.2f}  {punten:>9,.1f} pt  {benen:>6,}  {maat:<38}")
    return uit


def tabel_hoeveel_balans_voor(
    *, stap: float, lot: float, prijs: float, uitvoering: Uitvoering,
) -> list[str]:
    uit = [
        "",
        "  EN ANDERSOM: HOEVEEL BALANS EEN BEWEGING VRAAGT",
        "",
        f"  {'beweging':>10}  {'benen':>7}  {'balans nodig':>14}  {'zonder stop-out':>16}",
        f"  {'-' * 10}  {'-' * 7}  {'-' * 14}  {'-' * 16}",
    ]
    for punten, naam in BEWEGINGEN:
        nodig = balans_voor_beweging(
            punten, stap=stap, lot=lot, prijs=prijs, uitvoering=uitvoering)
        kaal = balans_voor_beweging(
            punten, stap=stap, lot=lot, prijs=prijs, uitvoering=ZONDER_STOPOUT)
        benen = benen_bij_beweging(punten, stap)
        uit.append(f"  {punten:>7,.0f} pt  {benen:>7,}  {nodig:>14,.2f}  "
                   f"{kaal:>16,.2f}   {naam}")
    uit += [
        "",
        "  De laatste kolom is wat de VORIGE meting dacht dat je nodig had: die",
        "  keek alleen of de balans op nul stond. De broker wacht daar niet op.",
    ]
    return uit


def het_verhaal_van_negenenvijftig(
    *, balans: float, stap: float, lot: float, prijs: float,
    uitvoering: Uitvoering,
) -> list[str]:
    """Been voor been, tot hij eruit vliegt. Dit is de trade die hij zou doen."""

    uit = [
        "",
        f"  EUR {balans:,.2f}, BEEN VOOR BEEN",
        f"  koop {lot:g} op {prijs:,.2f} en de markt gaat de verkeerde kant op",
        "",
        f"  {'beweging':>9}  {'benen':>6}  {'zwevend':>10}  {'eigen verm.':>12}  "
        f"{'marge':>9}  {'level':>7}",
        f"  {'-' * 9}  {'-' * 6}  {'-' * 10}  {'-' * 12}  {'-' * 9}  {'-' * 7}",
    ]
    punten = 0.0
    while punten <= 200.0:
        n, zwevend, marge, level = toestand_na_beweging(
            balans, punten, stap=stap, lot=lot, prijs=prijs, uitvoering=uitvoering)
        eigen = balans + zwevend
        vlag = ""
        if uitvoering.vliegt_eruit(eigen, marge):
            vlag = "   <<< HIER GOOIT DE BROKER JE ERUIT"
        elif not uitvoering.mag_bijopenen(eigen, marge):
            vlag = "   <<< margin call: je mag niet meer bijkopen"
        uit.append(f"  {punten:>6,.1f} pt  {n:>6,}  {zwevend:>10,.2f}  "
                   f"{eigen:>12,.2f}  {marge:>9,.2f}  {level:>6.0%}{vlag}")
        if vlag.endswith("ERUIT"):
            break
        punten += stap if punten < 5 else max(stap, 2.5)
    return uit


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--balans", type=float, default=59.16)
    p.add_argument("--stap", type=float, default=0.5)
    p.add_argument("--lot", type=float, default=0.01)
    p.add_argument("--prijs", type=float, default=4000.0)
    p.add_argument("--db", default="runtime/journal.db")
    args = p.parse_args()

    uitvoering = laad(args.db)

    regels = [
        "",
        "  " + "=" * 78,
        "  WELKE BALANS HEB JE NODIG",
        "  " + "=" * 78,
        "",
        f"  broker:     spread {uitvoering.spread:g} pt, slippage "
        f"{uitvoering.slippage:g} pt, hefboom 1:{uitvoering.hefboom:g}",
        f"              stop-out op {uitvoering.stop_out_niveau:.0%}, "
        f"margin call op {uitvoering.margin_call_niveau:.0%}",
        f"              commissie {uitvoering.commissie_per_lot_per_kant:g}, "
        f"swap {uitvoering.swap_per_lot_per_nacht:g}  (beide nul op goud)",
        f"  herkomst:   {uitvoering.herkomst}",
        f"  marge:      EUR {uitvoering.marge_voor(args.lot, args.prijs):,.2f} "
        f"per been van {args.lot:g} lot",
    ]
    regels += het_verhaal_van_negenenvijftig(
        balans=args.balans, stap=args.stap, lot=args.lot, prijs=args.prijs,
        uitvoering=uitvoering)
    regels += tabel_wat_houdt_een_balans_uit(
        stap=args.stap, lot=args.lot, prijs=args.prijs, uitvoering=uitvoering)
    regels += tabel_hoeveel_balans_voor(
        stap=args.stap, lot=args.lot, prijs=args.prijs, uitvoering=uitvoering)
    regels += ["", "  " + "=" * 78, ""]

    for r in regels:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
