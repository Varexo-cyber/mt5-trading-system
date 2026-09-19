"""DE ROBOT. Eén regel, van instap tot uitstap, gemeten als één bedrag.

WAT DIT IS. Geen meetgereedschap en geen onderzoek: een complete handelsregel
die je kunt aanzetten, doorgerekend over twee jaar tegen jouw echte balans.

WAAR DE INSTAP VANDAAN KOMT. `scripts/voorsprong.py` toetste vijf ideeen op
528.271 M1-bars goud en er bleven er twee over die hun eigen kosten verdienen:

    terugkeer 2 sigma    60m    +0,5666 punt   p = 0,001
    dagbereik gisteren  240m    +0,8025 punt   p = 0,001

Kosten per rondje zijn 0,16 spread + 0,10 slippage = 0,26 punt, dus daar blijft
respectievelijk +0,31 en +0,54 punt van over. De drie andere ideeen -- momentum,
uitbraak en de klokkenstapel uit sectie 20 -- kwamen NEGATIEF uit en zijn er
niet in gekomen.

DE ZES KEUZES DIE IK HEB GEMAAKT, en waarom:

  1. EEN POSITIE TEGELIJK. Het signaal vuurt duizenden keren; met overlappende
     posities meet je niet de regel maar de hefboom. Staat er een trade open,
     dan wordt er niet bijgeopend.

  2. INSTAP OP DE VOLGENDE BAR. Het signaal komt uit gesloten bars, dus de
     vroegste prijs die je echt kunt krijgen is de opening van de bar erna.

  3. ER IS ALTIJD EEN STOP. Dat is een harde projectregel en het is ook de enige
     reden dat deze rekening een slechte reeks overleeft. De stop staat op een
     veelvoud van de ATR, dus hij ademt mee met de markt.

  4. DE TIJD IS DE UITSTAP. De gemeten voorsprong is de gemiddelde beweging over
     een vaste horizon. Daar een winstdoel op plakken meet iets anders dan wat
     er gemeten is. Dus: eruit na precies die horizon, of eerder op de stop.

  5. DE STOP WINT DE BAR. Raakt een bar zowel de stop als het einde van de
     horizon, dan telt de stop. Binnen een bar weet niemand de volgorde.

  6. GOUD REKENT IN DOLLARS. Een punt op 0,01 lot is $1,00 en niet EUR 1,00.
     Dat werd in de vorige metingen door elkaar gehaald. Hier wordt alles in USD
     gerekend en met een vaste koers naar euro gezet, en die koers staat in het
     rapport.

WAT ER NIET IN ZIT, en dat hoort er eerlijk bij: geen nieuwsblokkade (historisch
nieuws is niet betrouwbaar terug te halen), geen verbredende spread rond nieuws,
geen weekendgaten, en tickdata ontbreekt -- binnen een minuut wordt de
pessimistische volgorde aangenomen.

    python scripts/robot.py --csv runtime/xauusd_m1.csv --balans 59.16
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.uitvoering import RAW_GOUD, Uitvoering, laad as laad_uitvoering
from scripts.voorsprong import (
    TOETSDEEL,
    lees_csv,
    signaal_dagbereik,
    signaal_terugkeer,
)

#: Eén punt koers per lot XAUUSD: 100 troy ounce.
CONTRACT = 100.0

#: EURUSD waartegen de winst wordt omgerekend.
#:
#: DIT IS EEN VASTE KOERS EN DAT IS EEN VEREENVOUDIGING. XAUUSD rekent in
#: dollars; jouw rekening staat in euro. In het gemeten venster liep EURUSD
#: ongeveer tussen 1,02 en 1,17. Eén vast getal is dus een paar procent naast de
#: waarheid, maar het is oneindig veel beter dan dollars en euro's door elkaar
#: halen -- wat de vorige metingen deden. Het getal staat in het rapport zodat
#: iedereen ziet waar de omrekening vandaan komt.
#:
#: 1,1489 IS GEEN GOK MAAR EEN AFGELEIDE METING. `config/gemeten_broker.json`
#: geeft `marge_per_lot: 762.45` in euro en `hefboom: 500`. Bij een goudprijs
#: van 4380 is de marge in dollars 100 x 4380 / 500 = 876. De verhouding
#: 876 / 762,45 = 1,1489 is dus de koers waar de broker zelf mee rekende op
#: het moment van meten. Mijn eerdere 1,10 was een ronde greep.
EURUSD = 1.1489


@dataclass(frozen=True)
class Regels:
    """De robot in tien getallen. Alles wat hij doet staat hier."""

    naam: str = "terugkeer 2 sigma"
    #: "terugkeer" of "dagbereik"
    signaal: str = "terugkeer"
    #: Venster voor het gemiddelde bij terugkeer.
    lengte: int = 60
    sigma: float = 2.0
    #: Hoeveel minuten de positie maximaal blijft staan.
    horizon: int = 60
    #: Stop op zoveel keer de ATR. Nooit None -- zie keuze 3 in de kop.
    stop_atr: float = 2.0
    atr_lengte: int = 60
    #: Deel van de balans dat per trade op het spel staat.
    risico: float = 0.02
    max_lot: float = 50.0
    uitvoering: Uitvoering = RAW_GOUD
    eurusd: float = EURUSD


#: De twee regels die de kostentoets overleefden. De namen zijn wat er in het
#: rapport komt te staan.
ROBOTS: tuple[Regels, ...] = (
    Regels(naam="terugkeer 2 sigma, 60m", signaal="terugkeer",
           lengte=60, sigma=2.0, horizon=60, stop_atr=2.0),
    Regels(naam="dagbereik gisteren, 240m", signaal="dagbereik",
           horizon=240, stop_atr=2.0, atr_lengte=240),
)


@dataclass
class Trade:
    geopend: pd.Timestamp
    richting: int
    entry: float
    stop: float
    lot: float
    gesloten: pd.Timestamp | None = None
    exit: float = 0.0
    reden: str = ""
    nachten: int = 0
    euro: float = 0.0
    r: float = 0.0
    bars: int = 0


@dataclass
class Uitslag:
    naam: str
    periode: str
    start: float
    eind: float
    trades: list[Trade] = field(default_factory=list)
    ruine: bool = False
    ruine_op: pd.Timestamp | None = None
    diepste_terugval: float = 0.0
    te_klein: int = 0

    @property
    def raak(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.euro > 0) / len(self.trades)

    @property
    def per_trade_r(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.r for t in self.trades) / len(self.trades)


def _atr(m1: pd.DataFrame, lengte: int) -> pd.Series:
    """Gemiddelde beweging per bar, over gesloten bars.

    De `shift(1)` staat er om dezelfde reden als overal in dit project: een ATR
    die de huidige bar meerekent, weet hoe groot die bar wordt.
    """

    hoog, laag, vorig = m1["high"], m1["low"], m1["close"].shift(1)
    bereik = pd.concat([hoog - laag, (hoog - vorig).abs(), (laag - vorig).abs()],
                       axis=1).max(axis=1)
    return bereik.rolling(lengte).mean().shift(1)


def _nachten(open_op: pd.Timestamp, dicht_op: pd.Timestamp) -> int:
    """Hoeveel keer de rollover gepasseerd is tussen openen en sluiten.

    BENADERING, EN DIE MOET GEZEGD WORDEN. MT5 boekt swap op de rollover van de
    broker (rond middernacht servertijd) en driedubbel op woensdag, voor het
    weekend. Hier wordt geteld hoeveel UTC-dagovergangen er tussen zitten, met
    de woensdagvermenigvuldiging erbij. Servertijd is bij deze broker niet UTC,
    dus rond middernacht kan het er eentje naast zitten.

    Een dag te veel of te weinig op een reeks van honderden trades verandert de
    orde van grootte niet, en de orde van grootte is hier het punt: EUR 0,81 per
    nacht tegenover EUR 0,26 per rondje.
    """

    nachten = (dicht_op.normalize() - open_op.normalize()).days
    if nachten <= 0:
        return 0
    # Woensdagnacht telt voor drie: dan wordt het weekend vooruit geboekt.
    extra = sum(2 for n in range(nachten)
                if (open_op.normalize() + pd.Timedelta(days=n + 1)).weekday() == 2)
    return nachten + extra


def _stopprijs(stop: float, bar_open: float, richting: int) -> float:
    """Waarop een geraakte stop WERKELIJK vult.

    EEN STOP IS EEN VERZOEK, GEEN PRIJSGARANTIE, en dat verschil is precies de
    reden dat dit een eigen functie is geworden.

    Hier stond `exit = stop`, altijd. Dat is de vriendelijke aanname, en ze is
    erger dan ze lijkt: als je stop altijd exact vult, kun je per definitie nooit
    meer dan een R verliezen en word je dus ook nooit geliquideerd. De hele
    staart van de verliesverdeling verdwijnt, en juist daar zit het risico.

    In het echt opent een bar na een weekend of bij nieuws gewoon voorbij je
    niveau. Je stop wordt dan omgezet in een marktorder en je vult op die
    opening. Dus: de SLECHTSTE van de twee.

    Aparte functie omdat dit een regel is en geen detail, en omdat hij zich dan
    rechtstreeks laat testen -- ik zat anders een kunstmatige markt te kneden om
    hem via een hele simulatie te raken, en dat is de verkeerde volgorde.
    """

    return min(stop, bar_open) if richting > 0 else max(stop, bar_open)


def _lot_voor(balans_eur: float, stop_punten: float, regels: Regels) -> float:
    """Positiegrootte, tussen de vloer van 0,01 en het plafond.

    DE VLOER IS HET PUNT. Onder 0,01 lot bestaat niet. Op een kleine rekening
    kun je dus NIET binnen je eigen risicogrens blijven, en dat is geen detail
    maar de reden dat een kleine rekening anders werkt dan een grote. Trades
    waar 0,01 lot te groot is worden geteld als `te_klein`.
    """

    risico_usd = balans_eur * regels.risico * regels.eurusd
    per_lot = stop_punten * CONTRACT
    if per_lot <= 0:
        return 0.01
    gewenst = risico_usd / per_lot
    return min(regels.max_lot, max(0.01, round(gewenst, 2)))


def draai(m1: pd.DataFrame, *, regels: Regels, balans: float,
          periode: str = "") -> Uitslag:
    """Bar voor bar, één positie tegelijk, met de broker erbij."""

    sein = (signaal_terugkeer(m1, regels.lengte, regels.sigma)
            if regels.signaal == "terugkeer" else signaal_dagbereik(m1))
    atr = _atr(m1, regels.atr_lengte)

    uitv = regels.uitvoering
    uit = Uitslag(naam=regels.naam, periode=periode, start=balans, eind=balans)
    stand = balans
    piek = balans
    open_trade: Trade | None = None
    tot_bar = 0

    seinen = sein.to_numpy()
    atrs = atr.to_numpy()
    opens = m1["open"].to_numpy(dtype=float)
    hoogs = m1["high"].to_numpy(dtype=float)
    laags = m1["low"].to_numpy(dtype=float)
    index = m1.index

    for i in range(len(m1) - 1):
        if open_trade is not None:
            t = open_trade
            t.bars += 1
            # 1. DE STOP WINT DE BAR. Raakt deze bar de stop, dan is dat de
            #    uitkomst -- ook als hij daarna nog de goede kant op ging.
            geraakt = (laags[i] <= t.stop) if t.richting > 0 else (hoogs[i] >= t.stop)
            if geraakt:
                # EEN GAT DOOR JE STOP HEEN VULT NIET OP JE STOP.
                #
                # Hier stond gewoon `t.exit = t.stop`, en dat is de vriendelijke
                # aanname: je stop is een VERZOEK, geen prijsgarantie. Opent de
                # bar al voorbij je stop -- na het weekend, bij nieuws, bij een
                # gat -- dan word je op die opening gevuld en niet op je niveau.
                #
                # Dit is ook de enige manier waarop de marge-stop-out echt
                # bereikbaar is: met een stop die altijd precies vult, kun je per
                # definitie niet meer verliezen dan je risico en word je nooit
                # geliquideerd. Dat klinkt geruststellend en het is niet waar.
                t.exit, t.reden = _stopprijs(t.stop, opens[i], t.richting), "stop"
            elif t.bars >= regels.horizon:
                t.exit, t.reden = opens[i], "tijd"
            if t.reden:
                punten = (t.exit - t.entry) * t.richting
                # SWAP PER NACHT, en die is op goud groter dan de spread.
                # Een long betaalt 80,67 per lot per nacht, een short krijgt
                # 23,83. Op 0,01 lot is dat EUR 0,81 tegen EUR 0,26 voor een
                # heel rondje. Een regel die over de nacht heen gaat wordt
                # hier gemaakt of gebroken, en ik rekende hem op nul.
                t.nachten = _nachten(t.geopend, index[i])
                kosten = (uitv.kosten_per_been(t.lot)
                          + uitv.slippage_kosten(t.lot)
                          + uitv.swap_kosten(t.lot, t.nachten,
                                             richting=t.richting))
                usd = punten * t.lot * CONTRACT - kosten
                t.euro = usd / regels.eurusd
                risico_usd = abs(t.entry - t.stop) * t.lot * CONTRACT
                t.r = usd / risico_usd if risico_usd else 0.0
                t.gesloten = index[i]
                uit.trades.append(t)
                stand += t.euro
                piek = max(piek, stand)
                uit.diepste_terugval = max(uit.diepste_terugval, piek - stand)
                open_trade = None
                if stand <= 0:
                    uit.ruine, uit.ruine_op, uit.eind = True, index[i], 0.0
                    return uit
            else:
                # 2. DE BROKER KIJKT MEE TERWIJL DE TRADE LOOPT.
                slechtst = laags[i] if t.richting > 0 else hoogs[i]
                zwevend_usd = (slechtst - t.entry) * t.richting * t.lot * CONTRACT
                # DE MARGE KOMT IN DOLLARS TERUG en de balans staat in euro.
                # Die twee direct vergelijken was precies de fout die Codex
                # bij de vorige metingen aanwees, en ik maakte hem hier
                # opnieuw -- tien procent scheef, in de gunstige richting.
                marge = uitv.marge_voor(t.lot, t.entry) / regels.eurusd
                if uitv.vliegt_eruit(stand + zwevend_usd / regels.eurusd, marge):
                    t.exit, t.reden = slechtst, "uitgegooid"
                    t.euro = max(-stand, zwevend_usd / regels.eurusd)
                    t.gesloten = index[i]
                    uit.trades.append(t)
                    uit.ruine, uit.ruine_op, uit.eind = True, index[i], 0.0
                    return uit
                continue

        # 3. INSTAP OP DE OPENING VAN DE VOLGENDE BAR, nooit op deze.
        richting = int(seinen[i])
        if richting == 0 or np.isnan(atrs[i]) or atrs[i] <= 0:
            continue
        entry = opens[i + 1]
        stop_punten = regels.stop_atr * float(atrs[i])
        lot = _lot_voor(stand, stop_punten, regels)

        # 4. PAST DEZE TRADE BINNEN DE MARGE? Zo niet, dan bestaat hij niet.
        # Ook hier: marge in dollars, balans in euro.
        if uitv.marge_voor(lot, entry) / regels.eurusd > stand * 0.5:
            uit.te_klein += 1
            continue
        # En blijft hij binnen het eigen risico? Op 0,01 lot vaak niet, en dat
        # wordt geteld in plaats van weggemoffeld.
        if stop_punten * lot * CONTRACT / regels.eurusd > stand * regels.risico * 1.5:
            uit.te_klein += 1

        open_trade = Trade(
            geopend=index[i + 1], richting=richting, entry=entry,
            stop=entry - richting * stop_punten, lot=lot)
        tot_bar = i

    uit.eind = stand
    return uit


def koop_en_hou(m1: pd.DataFrame, *, balans: float, eurusd: float = EURUSD) -> float:
    """DE NULHYPOTHESE. Koop 0,01 lot op de eerste bar en doe verder niets.

    Goud steeg in het gemeten venster ruim 70%. Verslaat een robot dit niet, dan
    heeft hij de stijging gemeten en niet zichzelf -- precies de fout die sectie
    20 twee jaar lang verborg.
    """

    entry = float(m1.iloc[0]["open"])
    punten = float(m1.iloc[-1]["close"]) - entry
    eind = balans + punten * 0.01 * CONTRACT / eurusd

    # JE KUNT NIET MINDER DAN NUL OVERHOUDEN, en dat stond er niet.
    #
    # Op het toetsdeel gaf dit -EUR 413,47, oftewel -798%. Dat is geen
    # rendement maar een rekenfout in het rapport: goud zakte 543 punten en op
    # 0,01 lot is dat EUR 473 op een rekening van EUR 59. Je bent dan al lang
    # geliquideerd; wat er daarna met de koers gebeurt gaat jou niet meer aan.
    #
    # Dat de NULHYPOTHESE zelf ook omvalt is trouwens het echte nieuws van die
    # regel, en dan hoort er nul te staan en geen fantasiegetal.
    if eind <= 0:
        return 0.0

    # EN ONDERWEG MOET HIJ HET OOK UITGEHOUDEN HEBBEN. De diepste stand bepaalt
    # of de broker je eruit gooide voordat de koers terugkwam.
    diepste = float(m1["low"].min()) - entry
    if balans + diepste * 0.01 * CONTRACT / eurusd <= 0:
        return 0.0
    return eind


def regel(u: Uitslag) -> str:
    vlag = "  RUINE" if u.ruine else "       "
    rendement = (u.eind / u.start - 1.0) if u.start else 0.0
    return (f"  {u.naam:<26}{u.periode:<10}{vlag}  EUR {u.eind:>10,.2f}  "
            f"{rendement:>+9.1%}  {len(u.trades):>6,} tr  "
            f"raak {u.raak:>5.1%}  {u.per_trade_r:>+7.3f} R/tr  "
            f"terugval EUR {u.diepste_terugval:>8,.2f}")


def rapport(m1: pd.DataFrame, *, balans: float, uitvoering: Uitvoering,
            eurusd: float) -> list[str]:
    grens = int(len(m1) * (1 - TOETSDEEL))
    delen = (("oefendeel", m1.iloc[:grens]), ("TOETSDEEL", m1.iloc[grens:]))

    regels_uit = [
        "",
        "  " + "=" * 96,
        f"   DE ROBOT  --  start EUR {balans:,.2f}",
        "  " + "=" * 96,
        "",
        f"   {len(m1):,} M1-bars   {m1.index[0]:%Y-%m-%d} t/m {m1.index[-1]:%Y-%m-%d}",
        f"   broker: spread {uitvoering.spread:g} pt, slippage "
        f"{uitvoering.slippage:g} pt, hefboom 1:{uitvoering.hefboom:g}, "
        f"stop-out {uitvoering.stop_out_niveau:.0%}",
        f"   herkomst: {uitvoering.herkomst}",
        f"   EURUSD vast op {eurusd:g}  (goud rekent in dollars, jouw rekening in euro)",
        "",
    ]

    for naam, deel in delen:
        regels_uit.append(f"  --- {naam}: {len(deel):,} bars, "
                          f"{deel.index[0]:%Y-%m-%d} t/m {deel.index[-1]:%Y-%m-%d}")
        for basis in ROBOTS:
            u = draai(deel, regels=replace(basis, uitvoering=uitvoering,
                                           eurusd=eurusd),
                      balans=balans, periode=naam)
            regels_uit.append(regel(u))
            if u.te_klein:
                regels_uit.append(
                    f"  {'':<36}  {u.te_klein:,} keer was 0,01 lot te groot voor "
                    f"de eigen risicogrens")
            if u.ruine:
                regels_uit.append(
                    f"  {'':<36}  rekening op {u.ruine_op:%Y-%m-%d %H:%M}")
        dom = koop_en_hou(deel, balans=balans, eurusd=eurusd)
        regels_uit += [
            f"  {'goud kopen en niks doen':<26}{naam:<10}         "
            f"EUR {dom:>10,.2f}  {dom / balans - 1:>+9.1%}       1 tr",
            "",
        ]

    regels_uit += [
        "  " + "-" * 96,
        "   HET TOETSDEEL IS HET ANTWOORD. Op het oefendeel is gezocht, dus daar",
        "   hoort iets moois te staan -- dat zegt niets. Wat op het TOETSDEEL",
        "   overeind blijft EN 'goud kopen en niks doen' verslaat, is echt.",
        "",
    ]
    return regels_uit


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--balans", type=float, default=59.16)
    p.add_argument("--eurusd", type=float, default=EURUSD)
    p.add_argument("--db", default="runtime/journal.db")
    p.add_argument("--uit", default="runtime/robot.txt")
    args = p.parse_args()

    m1 = lees_csv(args.csv)
    uitv = laad_uitvoering(args.db)
    regels_uit = rapport(m1, balans=args.balans, uitvoering=uitv,
                         eurusd=args.eurusd)
    for r in regels_uit:
        print(r, flush=True)
    if args.uit:
        Path(args.uit).parent.mkdir(parents=True, exist_ok=True)
        Path(args.uit).write_text("\n".join(regels_uit) + "\n", encoding="utf-8")
        print(f"  Ook opgeslagen in {args.uit}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
