"""Sectie 23: de UITSTAP meten -- breakeven, trailen, deels eruit, tijdstop.

DE VRAAG. "Wanneer profit-lock? Elke ATR? Als een trade al dik in winst staat
BE doen -- is dat beter of niet beter?" Daar heeft iedereen een mening over en
vrijwel niemand een meting.

WAT HIER ANDERS IS DAN IN SECTIE 20 EN 21. Daar werd een STRATEGIE gemeten.
Hier staat de strategie vast -- dezelfde ingang, dezelfde eerste stop, hetzelfde
doel -- en verandert alleen wat je ERNA doet. Dat is de enige manier om te weten
of het beheer iets toevoegt, want twee regels met verschillende ingangen
vergelijken meet de ingang.

DE VAL DIE DEZE MODULE MOET BLOOTLEGGEN. Breakeven en trailen VERHOGEN bijna
altijd de trefkans en VERLAGEN bijna altijd de verwachting. Ze knippen
winnaars af terwijl de verliezers even groot blijven. Dat voelt goed -- je
"verliest niet meer" -- en het kost geld. Daarom staat bij elke variant de
trefkans NAAST de verwachting, en wordt alles afgezet tegen de nulhypothese
"niets doen".

R IS ALTIJD DE EERSTE STOP. Niet de verschoven stop, niet de getrailde. Anders
verandert de meeteenheid mee met de variant en vergelijk je appels met peren.

PESSIMISTISCH BINNEN DE BAR. Raakt een bar zowel de stop als het doel, dan wint
de STOP. Wat er binnen die bar eerst gebeurde weet niemand.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scripts.section_twenty_pullback_ladder import (
    CONTRACT, KLOKKEN, _hersample, _lees_csv, _sessie_van, stapel_richting,
)

DAGEN = ("maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag")


@dataclass(frozen=True)
class Uitstap:
    """Wat er met een lopende trade gebeurt. De INGANG staat hier niet in."""

    naam: str
    #: Verschuif de stop naar instap zodra deze winst in R bereikt is.
    be_bij: float | None = None
    #: Trail de stop op zoveel ATR achter de hoogste stand.
    trail_atr: float | None = None
    #: Neem de helft af bij deze winst in R, laat de rest lopen.
    deel_bij: float | None = None
    #: Sluit hoe dan ook na zoveel bars.
    tijd_bars: int | None = None
    #: Vast doel in R. None = geen doel, alleen stop/trail/tijd.
    doel_r: float = 2.0


@dataclass
class Trade:
    moment: pd.Timestamp
    r: float
    bars: int
    mfe: float           # hoe ver hij in je voordeel liep, in R
    mae: float           # hoe ver tegen je in, in R
    reden: str


def _atr(frame: pd.DataFrame, periode: int = 14) -> pd.Series:
    hoog, laag, slot = frame["high"], frame["low"], frame["close"]
    vorig = slot.shift(1)
    tr = pd.concat([hoog - laag, (hoog - vorig).abs(), (laag - vorig).abs()], axis=1).max(axis=1)
    return tr.rolling(periode).mean()


def loop_trade(
    m1: pd.DataFrame, start: int, *, entry: float, stop: float, atr: float,
    uitstap: Uitstap, kosten_r: float, max_bars: int, kant: int = 1,
) -> Trade:
    """Een trade van instap tot uitstap, met R gemeten op de EERSTE stop.

    `kant` is +1 voor long en -1 voor short. Alles hieronder rekent in
    VOORDEEL en NADEEL in plaats van in hoog en laag, zodat er maar een pad
    is -- twee gespiegelde takken is hoe je hier een omgedraaid teken in stopt.
    """

    risico = (entry - stop) * kant
    if risico <= 0:
        return Trade(m1.index[start], 0.0, 0, 0.0, 0.0, "ongeldig")

    huidige_stop = stop
    doel = entry + risico * uitstap.doel_r if uitstap.doel_r else None
    top = entry
    mfe = mae = 0.0
    helft_eruit = 0.0        # in R, al binnengehaald
    nog_open = 1.0           # deel van de positie dat nog loopt

    for offset in range(max_bars):
        pos = start + offset
        if pos >= len(m1):
            break
        bar = m1.iloc[pos]
        hoog, laag = float(bar["high"]), float(bar["low"])
        # Beste en slechtste stand VOOR DEZE RICHTING.
        best = (hoog - entry) * kant if kant > 0 else (entry - laag)
        slechtst = (laag - entry) * kant if kant > 0 else (entry - hoog)
        mfe = max(mfe, best / risico)
        mae = min(mae, slechtst / risico)

        # 1. DE STOP EERST, ALTIJD. Raakt deze bar allebei, dan wint de stop.
        if (laag <= huidige_stop) if kant > 0 else (hoog >= huidige_stop):
            r = (huidige_stop - entry) * kant / risico * nog_open + helft_eruit
            return Trade(m1.index[pos], r - kosten_r, offset + 1, mfe, mae,
                         "stop" if huidige_stop <= stop else "stop verschoven")

        # 2. Deels eruit.
        if (uitstap.deel_bij is not None and nog_open == 1.0
                and best >= risico * uitstap.deel_bij):
            helft_eruit = 0.5 * uitstap.deel_bij
            nog_open = 0.5

        # 3. Doel.
        if doel is not None and best >= risico * uitstap.doel_r:
            r = uitstap.doel_r * nog_open + helft_eruit
            return Trade(m1.index[pos], r - kosten_r, offset + 1, mfe, mae, "doel")

        # 4. Breakeven verschuiven. NA de stopcontrole, want anders krijgt een
        #    trade die in dezelfde bar omhoog en daarna omlaag ging gratis een
        #    beschermde stop die hij live niet had.
        if uitstap.be_bij is not None and best >= risico * uitstap.be_bij:
            huidige_stop = entry

        # 5. Trailen achter de hoogste stand.
        if uitstap.trail_atr is not None:
            if kant > 0:
                top = max(top, hoog)
                huidige_stop = max(huidige_stop, top - atr * uitstap.trail_atr)
            else:
                top = min(top, laag)
                huidige_stop = min(huidige_stop, top + atr * uitstap.trail_atr)

        # 6. Tijdstop.
        if uitstap.tijd_bars is not None and offset + 1 >= uitstap.tijd_bars:
            slot = float(bar["close"])
            r = (slot - entry) * kant / risico * nog_open + helft_eruit
            return Trade(m1.index[pos], r - kosten_r, offset + 1, mfe, mae, "tijd")

    pos = min(start + max_bars, len(m1)) - 1
    slot = float(m1.iloc[pos]["close"])
    r = (slot - entry) * kant / risico * nog_open + helft_eruit
    return Trade(m1.index[pos], r - kosten_r, max_bars, mfe, mae, "einde venster")


def meet(
    m1: pd.DataFrame, stapels: dict[str, pd.DataFrame], uitstap: Uitstap, *,
    stop_pct: float = 0.0025, kosten_r: float = 0.03, om_de: int = 15,
    max_bars: int = 240,
) -> list[Trade]:
    """Dezelfde ingang voor elke variant: de stapel omhoog, stop 0,25% eronder."""

    atr = _atr(m1, 14)
    trades: list[Trade] = []
    i = 20
    while i < len(m1) - 1:
        # BEIDE KANTEN, en dat was hier ook fout. Sectie 23 mat de uitstap op
        # een LONG-ONLY ingang, in een venster waarin goud 70% steeg. Die
        # +0,55 R per trade was dus grotendeels de stijging, net als bij
        # sectie 20. Dezelfde fout, twee bestanden verder.
        kant = stapel_richting(stapels, m1.index[i])
        if kant == 0:
            i += om_de
            continue
        entry = float(m1.iloc[i]["open"])
        a = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else entry * 0.0005
        stop = entry * (1 - stop_pct) if kant > 0 else entry * (1 + stop_pct)
        t = loop_trade(m1, i, entry=entry, stop=stop, atr=a, kant=kant,
                       uitstap=uitstap, kosten_r=kosten_r, max_bars=max_bars)
        trades.append(t)
        i += max(om_de, t.bars)
    return trades


def cijfers(trades: list[Trade], *, euro_per_r: float) -> dict[str, float]:
    if not trades:
        return {}
    r = pd.Series([t.r for t in trades])
    eq = r.cumsum()
    return {
        "trades": len(r),
        "trefkans": float((r > 0).mean()),
        "per_trade": float(r.mean()),
        "totaal_r": float(r.sum()),
        "euro": float(r.sum() * euro_per_r),
        "terugval_r": float((eq.cummax() - eq).max()),
        "t": float(r.mean() / r.std(ddof=1) * np.sqrt(len(r))) if len(r) > 1 and r.std(ddof=1) > 0 else 0.0,
        "gem_bars": float(np.mean([t.bars for t in trades])),
        "gem_mfe": float(np.mean([t.mfe for t in trades])),
    }


def per_groep(trades: list[Trade], sleutel) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    f = pd.DataFrame([{"groep": sleutel(t), "r": t.r} for t in trades])
    return f.groupby("groep").agg(
        trades=("r", "size"), totaal=("r", "sum"), per_trade=("r", "mean"),
        trefkans=("r", lambda s: float((s > 0).mean())),
    ).sort_values("per_trade", ascending=False)


def uur_permutatie(trades: list[Trade], *, rondes: int = 1000, zaad: int = 5) -> dict:
    """Het beste uur tegen zuiver toeval. Zie sectie 21 voor het waarom."""

    if len(trades) < 48:
        return {"echt": float("nan"), "p": float("nan")}
    w = np.array([t.r for t in trades])
    u = np.array([t.moment.hour for t in trades])
    uniek = np.unique(u)
    echt = max(w[u == x].mean() for x in uniek)
    rng = np.random.default_rng(zaad)
    beter = sum(
        1 for _ in range(rondes)
        if max((d := rng.permutation(w))[u == x].mean() for x in uniek) >= echt
    )
    return {"echt": float(echt), "p": (beter + 1) / (rondes + 1)}


def varianten() -> list[Uitstap]:
    """DE HELE WAAIER, want daar is om gevraagd.

    Elke rij verandert precies EEN ding ten opzichte van de nulhypothese, op
    de gecombineerde rijen onderaan na. Anders weet je bij een verschil niet
    waar het vandaan komt.
    """

    uit = [Uitstap("NIETS DOEN (stop + doel 2R)")]
    uit += [Uitstap(f"BE bij +{x:.2f}R", be_bij=x) for x in (0.25, 0.5, 0.75, 1.0, 1.5)]
    uit += [Uitstap(f"trail {x:g} ATR", trail_atr=x, doel_r=0.0) for x in (1.0, 1.5, 2.0, 3.0)]
    uit += [Uitstap(f"trail {x:g} ATR + doel 2R", trail_atr=x) for x in (1.5, 3.0)]
    uit += [Uitstap(f"helft eruit bij +{x:g}R", deel_bij=x) for x in (0.5, 1.0, 1.5)]
    uit += [Uitstap(f"tijdstop {n} bars", tijd_bars=n) for n in (15, 30, 60, 120)]
    uit += [
        Uitstap("BE +0,5R  +  trail 2 ATR", be_bij=0.5, trail_atr=2.0, doel_r=0.0),
        Uitstap("helft +1R  +  BE  +  doel 3R", deel_bij=1.0, be_bij=1.0, doel_r=3.0),
        Uitstap("doel 1R (klein, hoge trefkans)", doel_r=1.0),
        Uitstap("doel 4R (groot, lage trefkans)", doel_r=4.0),
    ]
    return uit


def hoeveel_verliezen_dodelijk(
    trades: list[Trade], start: float, *, stop_punten: float,
    euro_per_punt: float = 0.87,
) -> dict[str, float]:
    """HOEVEEL VERLIEZERS OP RIJ MAKEN DEZE REKENING OP -- en hoe waarschijnlijk.

    WAAROM DIT NAAST DE LADDER MOET. Daar staat dat EUR 59 het GROOTSTE veelvoud
    haalt. Dat leest als een voordeel en het is het tegenovergestelde: die
    rekening vermenigvuldigt zo snel omdat ze GEDWONGEN wordt vijftien procent
    per trade te riskeren. Het minimumlot is een vloer, dus er is geen knop om
    minder te riskeren.

    Hetzelfde dat haar snel laat groeien, maakt haar dood. Een reeks verliezers
    die op een grote rekening een schrammetje is, is hier het einde. Dat getal
    hoort naast het veelvoud te staan, anders is de ladder een verkooppraatje.

    De kans is die van EEN aaneengesloten reeks in dit aantal trades; hij
    veronderstelt onafhankelijke uitkomsten, wat een OPTIMISTISCHE aanname is
    omdat verliezers in de praktijk clusteren.
    """

    if not trades:
        return {}
    verlies_euro = stop_punten * euro_per_punt
    nodig = int(start / verlies_euro) + 1
    p_verlies = float(np.mean([t.r <= 0 for t in trades]))
    n = len(trades)
    kans_reeks = p_verlies ** nodig
    # Ruwe verwachting van het aantal van zulke reeksen in deze steekproef.
    verwacht = max(0, n - nodig + 1) * kans_reeks
    return {
        "verlies_per_trade": verlies_euro,
        "dodelijk_na": nodig,
        "kans_op_die_reeks": kans_reeks,
        "verwacht_aantal": verwacht,
        "p_verlies": p_verlies,
    }


#: WAT DE BROKER JE MAXIMAAL LAAT HANDELEN. Eightcap zit rond de 50 lot per
#: order op XAUUSD; andere brokers zitten in dezelfde orde.
#:
#: ZONDER DEZE GRENS IS DE LADDER ONZIN. Een positieve verwachting samengesteld
#: over 2570 trades explodeert altijd: EUR 59 werd EUR 3.500.000.000.000, en
#: bij die stand zou je 73 miljoen lot moeten handelen -- 7 miljard ounce goud,
#: terwijl de hele COMEX er 25 miljoen per dag doet.
#:
#: De rekensom klopte; hij beschreef alleen een markt die niet bestaat. Een
#: backtest die dat niet afkapt, meet samengestelde groei en geen strategie.
MAX_LOT = 50.0


def balansladder(
    trades: list[Trade], *, stop_punten: float, euro_per_punt_min: float = 0.87,
    risico_deel: float = 0.02, lotstap: float = 0.01, max_lot: float = MAX_LOT,
) -> pd.DataFrame:
    """WAT DEZELFDE REGEL OP EEN ANDERE STARTBALANS HAD GEDAAN.

    DIT IS DE VRAAG WAAR HET OM DRAAIT: "als ik zoveel balans had gehad, had ik
    dan wel dikke winst gehad?" Het antwoord is geen simpele vermenigvuldiging,
    en dat is precies waarom hij hier apart berekend wordt.

    DRIE DINGEN MAKEN HET ONGELIJK:

      1. HET MINIMUMLOT IS EEN VLOER. Onder 0,01 lot bestaat niet. Op een kleine
         rekening kun je dus NIET binnen je risicogrens blijven -- je risico
         wordt bepaald door de broker en niet door jou. Op EUR 59 is een stop
         van 10,75 punten vijftien procent, of je wil of niet.

      2. BOVEN DE DREMPEL SCHAALT HET WEL MEE. Zodra 2% van je balans meer dan
         een minimumlot toelaat, groeit de positie met de rekening mee en gaat
         het samengesteld lopen. Daar zit het verschil tussen "wat meer winst"
         en een heel andere uitkomst.

      3. DE RUINE BLIJFT EEN HARDE STOP. Een reeks die op een grote rekening
         prima doorloopt, is op een kleine al in maand twee afgelopen -- en dan
         tellen de trades daarna niet meer mee. Ze optellen alsof de rekening
         nog bestond is de fout die dit hele project najaagt.
    """

    rijen = []
    for start in (59.16, 100.0, 200.0, 467.0, 1_000.0, 2_500.0, 5_000.0, 10_000.0):
        balans = start
        piek = start
        diepste = 0.0
        kapot = False
        geplafonneerd = False
        gedaan = 0
        for tr in sorted(trades, key=lambda x: x.moment):
            # Lotgrootte uit de risicogrens, met het minimumlot als vloer.
            gewenst = risico_deel * balans / (stop_punten * euro_per_punt_min / lotstap)
            # Vloer EN plafond: kleiner dan 0,01 bestaat niet, groter dan wat de
            # broker accepteert ook niet.
            lot = min(max_lot, max(lotstap, (int(gewenst / lotstap)) * lotstap))
            if lot >= max_lot:
                geplafonneerd = True
            risico_euro = stop_punten * euro_per_punt_min * (lot / lotstap)
            balans += tr.r * risico_euro
            gedaan += 1
            if balans <= 0:
                balans = 0.0
                kapot = True
                break
            piek = max(piek, balans)
            diepste = max(diepste, (piek - balans) / piek)

        eerste_risico = stop_punten * euro_per_punt_min / start
        rijen.append({
            "start": start,
            "risico_eerste_trade": eerste_risico,
            "haalbaar": eerste_risico <= risico_deel,
            "eind": balans,
            "keer": balans / start if start else 0.0,
            "trades": gedaan,
            "terugval": diepste,
            "ruine": kapot,
            "geplafonneerd": geplafonneerd,
        })
    return pd.DataFrame(rijen)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--balans", type=float, required=True)
    p.add_argument("--stop-pct", type=float, default=0.0025)
    p.add_argument("--rapport")
    args = p.parse_args()

    print(f"\n  Bars laden uit {args.csv} ...", flush=True)
    m1 = _lees_csv(args.csv)
    stapels = {n: _hersample(m1, r) for n, r in KLOKKEN}
    # Wat 1 R je kost in euro, bij het minimumlot op deze balans.
    koers = float(m1.iloc[-1]["close"])
    euro_per_r = koers * args.stop_pct * 0.01 * CONTRACT
    print(f"  {len(m1):,} bars   1 R = {koers * args.stop_pct:.2f} punten "
          f"= EUR {euro_per_r:.2f} bij 0,01 lot", flush=True)
    print(f"  dat is {euro_per_r / args.balans:.1%} van je balans van "
          f"EUR {args.balans:.2f} PER TRADE\n", flush=True)

    regels = []
    nul = None
    alle: dict[str, list[Trade]] = {}
    for v in varianten():
        trades = meet(m1, stapels, v, stop_pct=args.stop_pct)
        c = cijfers(trades, euro_per_r=euro_per_r)
        alle[v.naam] = trades
        if nul is None:
            nul = c
        verschil = c["per_trade"] - nul["per_trade"]
        vlag = "  <-- NULHYPOTHESE" if c is nul else (
            f"  {verschil:+.4f} R" + ("  BETER" if verschil > 0 else "  slechter"))
        regels.append(
            f"  {v.naam:34s} {c['trades']:>6,}tr  raak {c['trefkans']:>5.1%}  "
            f"{c['per_trade']:+.4f} R/tr  t {c['t']:+6.2f}  "
            f"EUR {c['euro']:>10,.2f}  DD {c['terugval_r']:>6.1f}R{vlag}"
        )
        print(regels[-1], flush=True)

    beste = max(alle, key=lambda k: cijfers(alle[k], euro_per_r=euro_per_r)["per_trade"])
    staart = [
        "",
        "  " + "=" * 74,
        f"  BESTE VARIANT: {beste}",
        "  " + "=" * 74,
        "",
        "  LEES DIT EERST. Breakeven en trailen verhogen bijna altijd de",
        "  TREFKANS en verlagen bijna altijd de VERWACHTING: ze knippen",
        "  winnaars af terwijl de verliezers even groot blijven. Voelt goed,",
        "  kost geld. Kijk dus naar de R/tr-kolom en niet naar 'raak'.",
        "",
        "  EN DE BESTE VAN TWINTIG VARIANTEN IS HET MAXIMUM VAN TWINTIG",
        "  RUIZIGE GETALLEN. Zonder vooruittest is dat geen keuze maar een",
        "  selectie.",
        "",
    ]
    for r in staart:
        print(r, flush=True)

    kopjes = []
    for naam, sleutel in (("JAAR", lambda t: t.moment.year),
                          ("SESSIE", lambda t: _sessie_van(t.moment)),
                          ("WEEKDAG", lambda t: DAGEN[t.moment.weekday()]),
                          ("UUR (UTC)", lambda t: t.moment.hour)):
        kopjes.append(f"\n  {beste}  --  PER {naam}")
        kopjes.append(per_groep(alle[beste], sleutel).to_string())
        print(kopjes[-2], flush=True)
        print(kopjes[-1], flush=True)

    ladder = balansladder(alle[beste], stop_punten=koers * args.stop_pct)
    lad = ["", "  " + "=" * 74,
           f"  BALANSLADDER  --  {beste}, lot groeit mee met 2% risico",
           "  " + "=" * 74, ""]
    for _, rij in ladder.iterrows():
        merk = ("RUINE " if rij["ruine"] else "MAXLOT" if rij["geplafonneerd"]
                else "      " if rij["haalbaar"] else "TE KLEIN")
        lad.append(
            f"  start EUR {rij['start']:>9,.2f}  {merk}  eind EUR {rij['eind']:>13,.2f}"
            f"   x{rij['keer']:>8,.1f}   risico 1e trade {rij['risico_eerste_trade']:>6.1%}"
            f"   terugval {rij['terugval']:>5.1%}")
    lad += [""]
    for s in (59.16, 467.0, 5000.0):
        d = hoeveel_verliezen_dodelijk(alle[beste], s, stop_punten=koers * args.stop_pct)
        if not d:
            continue
        lad.append(
            f"  EUR {s:>9,.2f}:  {d['dodelijk_na']:>3} verliezers op rij is het einde"
            f"   (kans op zo'n reeks {d['kans_op_die_reeks']:.2%},"
            f" verwacht {d['verwacht_aantal']:.2f}x in deze reeks)")
    lad += ["",
            f"  'MAXLOT' betekent: de positie liep tegen de {MAX_LOT:.0f} lot aan die",
            "  je broker maximaal accepteert. Vanaf daar groeit hij niet meer mee",
            "  en is het rendement in procenten dus geen extrapolatie meer.",
            "",
            "  EN LEES GEEN EINDBEDRAG MET TIEN NULLEN ALS EEN VOORSPELLING.",
            "  Elke positieve verwachting samengesteld over duizenden trades",
            "  explodeert -- dat is de rekensom en niet de strategie. Bij zulke",
            "  standen zou je meer goud moeten handelen dan er per dag omgaat,",
            "  en dan bestaat de voorsprong al lang niet meer.",
            "",
            "  DAT EUR 59 HET GROOTSTE VEELVOUD HAALT IS GEEN VOORDEEL.",
            "  Die rekening groeit zo snel omdat ze GEDWONGEN wordt vijftien",
            "  procent per trade te riskeren -- het minimumlot is een vloer, er",
            "  is geen knop om minder te nemen. Precies datzelfde maakt haar",
            "  dood: een reeks verliezers die op een grote rekening een",
            "  schrammetje is, is hier het einde.",
            "",
            "  'TE KLEIN' betekent: je eerste trade riskeert meer dan 2% en je",
            "  kunt niet kleiner dan 0,01 lot. Het minimumlot is een VLOER --",
            "  op die rekeningen bepaalt de broker je risico, niet jij.",
            "",
            "  Boven de drempel groeit de positie mee en loopt het samengesteld.",
            "  Daar zit het verschil tussen 'wat meer winst' en een heel andere",
            "  uitkomst. Onder de drempel is het geen kleinere versie van",
            "  hetzelfde -- het is een ander spel.",
            ""]
    for r in lad:
        print(r, flush=True)
    slot_extra = lad

    toets = uur_permutatie(alle[beste])
    slot = [
        f"\n  BESTE UUR: {toets['echt']:+.4f} R per trade, p = {toets['p']:.3f}",
        "  p boven 0,05: dat uur is het maximum van 24 ruizige getallen.",
        "",
    ]
    for r in slot:
        print(r, flush=True)

    if args.rapport:
        with open(args.rapport, "a", encoding="utf-8") as f:
            f.write("\n".join(regels + staart + kopjes + slot_extra + slot) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
