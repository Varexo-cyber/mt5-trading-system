"""Wat een replay had opgeleverd als handelen niets kostte.

DE VRAAG IS AL BEANTWOORD IN ELKE CSV DIE ER LIGT en dat is het punt van dit
script. `dry_run_sections` schrijft per trade de kolom `cost_r_charged`, en die
is AL van `result_r_fixed_stop` en `managed_r_LIVE` afgetrokken. Het brutocijfer
is dus gewoon net + kosten, per trade, zonder dat er iets opnieuw gedraaid hoeft
te worden. Een replay van 180 of 360 dagen duurt uren; dit leest het bestand dat
er al staat.

WAAROM DAT ERTOE DOET. Netto negatief kan twee totaal verschillende dingen
betekenen, en ze vragen om tegenovergestelde beslissingen:

    bruto positief, netto negatief   -> de instap ziet iets, de kosten eten het
                                        op. Dan is de vraag: kan deze trade
                                        verder lopen, of goedkoper.
    bruto negatief                   -> de instap ziet niets. Dan is er geen
                                        kostenknop die dit redt en is elke
                                        verdere poort tijdverspilling.

Zonder deze splitsing zijn die twee niet uit elkaar te houden, en dan wordt er
maandenlang aan poorten gedraaid voor een instap die geen richting heeft.

WAT DIT NIET IS. Bruto is GEEN haalbaar resultaat. Je kunt niet handelen zonder
spread te betalen. Het is een diagnose van waar het verlies zit, niet een
alternatief scenario. Een sectie die bruto +40R is en netto -10R heeft geen
edge die je kwijtraakte; hij heeft een edge die kleiner is dan de prijs.
"""

from __future__ import annotations

import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path

#: Weigeringen die de replay TOCH heeft uitgelopen. `dry_run_sections` schrijft
#: voor deze setups een volledig resultaat weg met de notitie "COUNTERFACTUAL if
#: gate were off", dus de vraag "wat had die poort gekost" staat al in het
#: bestand. Alleen weigeringen met een resultaat tellen; een setup die op
#: UNDERCAPITALIZED strandde heeft geen uitkomst om op te tellen.
_COUNTERFACTUAL_NOTE = "COUNTERFACTUAL"


def _rows(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Dezelfde reden als in month_attribution: een cmd-export op Nederlands
        # Windows zet CP-1252 in de toelichtingskolom. De cijfers zijn gelijk.
        text = raw.decode("cp1252")
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle)
        needed = {"module", "outcome", "cost_r_charged"}
        missing = needed.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path}: ontbrekende kolommen: {', '.join(sorted(missing))}. "
                "Deze CSV komt van een oudere versie die de kosten nog niet "
                "per trade wegschreef; draai de replay opnieuw."
            )
        return list(reader)


def _taken(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["outcome"].strip().upper() == "TRADE"]


def _refused_but_resolved(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Geweigerde setups die de replay tóch heeft uitgelopen.

    De notitie is de enige betrouwbare markering. Alleen op een leeg
    resultaatveld filteren zou ook setups meenemen die om een andere reden geen
    uitkomst hebben, en die als "de poort kostte niets" rapporteren.
    """

    return [
        row
        for row in rows
        if row["outcome"].strip().upper() != "TRADE"
        and _COUNTERFACTUAL_NOTE in (row.get("note") or "").upper()
        and (
            (row.get("managed_r_LIVE") or "").strip()
            or (row.get("result_r_fixed_stop") or "").strip()
        )
    ]


def _number(row: dict[str, str], *names: str) -> float:
    """Eerste kolom met een leesbaar getal, anders 0.0.

    `managed_r_LIVE` is leeg voor secties op een vaste uitstap, dus de volgorde
    hier is dezelfde als in `month_attribution`: eerst wat de rekening draait,
    dan de vaste route.
    """

    for name in names:
        value = (row.get(name) or "").strip()
        if value:
            try:
                return float(value)
            except ValueError:
                continue
    return 0.0


def report(path: Path) -> None:
    every = _rows(path)
    rows = _taken(every)
    if not rows:
        print(f"{path}: geen enkele regel met outcome TRADE.")
        return

    per_module: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for row in rows:
        net = _number(row, "managed_r_LIVE", "result_r_fixed_stop")
        cost = _number(row, "cost_r_charged")
        values = per_module[row["module"].strip() or "ONBEKEND"]
        values[0] += 1
        values[1] += net
        values[2] += cost

    print(f"\nBRUTO TEGEN NETTO — {path.name}")
    print("  bruto = netto + kosten. De kosten stonden al in de netto-kolom;")
    print("  dit telt ze alleen terug op. Er is niets opnieuw gedraaid.")
    print(
        f"\n  {'sectie':<28}{'trades':>7}{'bruto R':>10}{'kosten R':>10}"
        f"{'netto R':>10}{'per trade':>11}{'kosten%':>9}"
    )

    total = [0.0, 0.0, 0.0]
    for module, (count, net, cost) in sorted(per_module.items()):
        gross = net + cost
        total[0] += count
        total[1] += net
        total[2] += cost
        # Kosten als aandeel van BRUTO, want dat is de vraag "hoeveel van wat
        # de instap verdiende ging op aan handelen". Tegen netto delen geeft
        # onzin zodra netto rond nul of negatief staat.
        share = cost / abs(gross) if gross else 0.0
        print(
            f"  {module:<28}{int(count):>7}{gross:>+10.2f}{cost:>10.2f}"
            f"{net:>+10.2f}{net / count:>+11.3f}{share:>8.0%}"
        )

    count, net, cost = total
    gross = net + cost
    share = cost / abs(gross) if gross else 0.0
    print(f"  {'-' * 83}")
    print(
        f"  {'samen':<28}{int(count):>7}{gross:>+10.2f}{cost:>10.2f}"
        f"{net:>+10.2f}{net / count:>+11.3f}{share:>8.0%}"
    )

    print("\n  HOE JE DIT LEEST")
    if gross > 0.0 and net <= 0.0:
        print("  Bruto positief, netto niet. De instap ziet richting; hij is kleiner")
        print("  dan wat een rondje handelen kost. De vraag is dan of dezelfde trade")
        print("  verder kan lopen (grotere R) of goedkoper kan (andere rekening),")
        print("  NIET of er een betere poort bestaat.")
    elif gross <= 0.0:
        print("  Bruto is al negatief. De instap ziet geen richting, ook niet gratis.")
        print("  Geen enkele kosten- of spreadpoort verandert dat, en een sectie")
        print("  hierop verder afregelen is tijd in een gat gooien.")
    else:
        print("  Bruto en netto allebei positief. De kostenkolom zegt hoeveel marge")
        print("  er tussen zit -- hoe kleiner dat verschil, hoe minder er hoeft te")
        print("  gebeuren voordat dit alsnog omslaat.")
    print("\n  Bruto is GEEN haalbare uitkomst. Handelen zonder spread bestaat niet.")
    print("  Het zegt alleen WAAR het verlies zit, niet dat het te vermijden was.\n")

    _gate_report(path, every, {module: values[1] for module, values in per_module.items()})


def _gate_report(path: Path, every: list[dict[str, str]], taken_net: dict[str, float]) -> None:
    """Wat de poorten hebben geweigerd, en of dat winst of verlies was.

    EEN ANDERE VRAAG DAN BRUTO TEGEN NETTO, en ze worden makkelijk verward.
    Bruto-tegen-netto gaat over de trades die GENOMEN zijn, zonder de kosten
    die eraf gingen. Dit gaat over de setups die NIET genomen zijn omdat een
    poort ze weigerde. De replay heeft die toch uitgelopen en het resultaat
    weggeschreven, dus het staat in hetzelfde bestand.

    POSITIEF BETEKENT DAT DE POORT WINST HEEFT WEGGEHAALD, negatief dat hij
    verlies heeft tegengehouden. Dat is de enige manier om te beoordelen of een
    poort zijn plek verdient: hoeveel hij weigert zegt niets, wat hij weigerde
    zegt alles.
    """

    refused = _refused_but_resolved(every)
    if not refused:
        print("  (Geen uitgelopen weigeringen in dit bestand, dus geen poortrapport.)\n")
        return

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in refused:
        module = row["module"].strip() or "ONBEKEND"
        grouped[(module, row["outcome"].strip())].append(
            _number(row, "managed_r_LIVE", "result_r_fixed_stop")
        )

    print(f"WAT DE POORTEN WEIGERDEN — {path.name}")
    print("  Positieve R = de poort haalde WINST weg. Negatieve R = hij hield")
    print("  VERLIES tegen. Deze setups zijn met dezelfde kosten uitgelopen als")
    print("  de genomen trades, dus de getallen zijn vergelijkbaar.")
    print(f"\n  {'sectie':<28}{'poort':<26}{'setups':>7}{'raak':>7}{'R':>10}")

    per_module_gate: dict[str, float] = defaultdict(float)
    for (module, gate), values in sorted(grouped.items()):
        total = sum(values)
        won = sum(1 for value in values if value > 0.0) / len(values)
        per_module_gate[module] += total
        print(f"  {module:<28}{gate:<26}{len(values):>7}{won:>6.0%}{total:>+10.2f}")

    print(f"\n  {'sectie':<28}{'nu':>12}{'poorten uit':>14}{'verschil':>12}")
    for module in sorted(set(taken_net) | set(per_module_gate)):
        now = taken_net.get(module, 0.0)
        delta = per_module_gate.get(module, 0.0)
        print(f"  {module:<28}{now:>+12.2f}{now + delta:>+14.2f}{delta:>+12.2f}")

    print("\n  DIT IS GEEN REPLAY MET DE POORTEN UIT, en dat verschil is groot.")
    print("  Elke geweigerde setup is los uitgelopen alsof hij er alleen stond.")
    print("  Met de poorten echt uit hadden ze slots bezet, andere trades")
    print("  verdrongen en elkaars risico gedeeld -- het positieboek volgt deze")
    print("  som niet. Lees de kolom `poorten uit` als een richting, niet als")
    print("  een bedrag dat je gemist hebt.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="+", help="replay-CSV uit runtime\\")
    args = parser.parse_args()
    for path in args.csv:
        report(path)


if __name__ == "__main__":
    main()
