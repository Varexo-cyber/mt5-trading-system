"""Sectie 20: vooraf vastgelegde meting van de terugval-ladder op XAUUSD.

WAT DIT IS. De eigenaar vroeg om een sectie die met de trend meeloopt en op
elke terugval van een halve punt nog 0,01 bijkoopt, zonder stop, tot de mand
in winst staat. Dat is een grid met averaging down, en `risk_manager.
assert_not_forbidden` weigert het vandaag met zoveel woorden.

Deze module handelt niets. Hij MEET, en hij is met opzet zo gebouwd dat de
uitkomst niet mooier kan lijken dan hij is:

  * DE VULLING IS PESSIMISTISCH. Binnen een bar worden eerst alle bereikte
    ladderprijzen gevuld en pas daarna wordt op winst gekeken. Raakt een bar
    zowel een nieuw been als het doel, dan wint het been. Andersom zou elke
    ladder er ondiep uitzien.

  * DE KOSTEN WORDEN PER BEEN GEREKEND, twee keer -- in en uit. Een grid opent
    veel meer posities dan een gewone regel, dus kosten per BEEN in plaats van
    per mand is het hele verschil.

  * DE RAPPORTAGE STAAT OMGEDRAAID. Ruines, diepste onderwaterstand en meeste
    benen komen voor trefkans en winst. Een grid heeft bij vrijwel elke
    steekproef een trefkans boven de 90% en een positief totaal, tot de dag dat
    de rekening weg is. Trefkans bovenaan zetten zou de uitkomst verbergen in
    plaats van hem tonen.

WAAROM ER EEN BEGRENSDE VARIANT NAAST STAAT. Zonder vergelijking is niet te
zien of de winst uit de regel komt of uit de staart die de rekening opblaast.
Doet `begrensd` bijna hetzelfde zonder ruines, dan is de onbegrensde versie
nergens voor nodig; doet hij niets, dan zat alle winst in die staart.

DRAAIEN: closemomentum.cmd draait op MT5, en dat is Windows. Op Linux laden de
tests dit bestand wel en draaien ze de rekenkern op verzonnen bars, zodat de
mechaniek getest is zonder MT5.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from datetime import UTC

import numpy as np
import pandas as pd

from scripts.uitvoering import RAW_GOUD, Uitvoering

# Sessies in UTC. Ze mogen over middernacht heen lopen en dat doet Asia ook.
SESSIES: dict[str, tuple[int, int]] = {
    "asia": (23, 8),
    "londen": (7, 16),
    "newyork": (12, 21),
}
OVERLAP = ("londen", "newyork")

#: Wat één punt koers waard is per lot op XAUUSD: 100 troy ounce per lot.
CONTRACT = 100.0

#: COMMISSIE OP GOUD IS NUL, EN DAT IS EEN METING, GEEN AANNAME.
#:
#: Ik las `commission_per_lot_per_side: 2.75` uit `config/eightcap.yaml` en
#: rekende die op elk been door. Dat getal geldt alleen voor FOREX. Twee regels
#: lager in datzelfde bestand staat `commission_by_asset_class` met
#: `metal: 0.0`, en de toelichting erbij is uit de rekening zelf gemeten: negen
#: gesloten trades op 24 augustus, samen EUR 0,67 commissie, waarvan USDJPY
#: alleen al 0,12 lot x 5,50 = EUR 0,66. De XAUUSD-, SPX500-, SG30- en
#: BTCUSD-trades in datzelfde venster droegen samen NUL bij, en het detailpaneel
#: van de SPX500-deal zei het letterlijk: `Commission: 0.00`.
#:
#: Dus dezelfde fout als de rest van deze week: er staan twee beschrijvingen van
#: een regel in het project en ik pakte de verkeerde. Bij goud zit de kost in de
#: spread en nergens anders.
#:
#: De constante blijft staan omdat de kostenformule dan op EEN plek woont, en
#: `tests/test_section_twenty_pullback_ladder.py` zet hem vast tegen
#: `commission_by_asset_class['metal']` uit de config, zodat hij niet opnieuw
#: kan wegdrijven. Hij LEEST uit `scripts/uitvoering.py`, zodat er ook geen
#: tweede plek is waar hij anders kan gaan staan.
COMMISSIE_PER_LOT_PER_KANT = RAW_GOUD.commissie_per_lot_per_kant


def kosten_per_been(spread: float, lot: float) -> float:
    """Wat één been kost van openen tot sluiten: spread heen en terug.

    Doorgeefluik naar `Uitvoering.kosten_per_been`, zodat de oude aanroepen
    blijven werken en er toch maar één formule bestaat.
    """

    return replace(RAW_GOUD, spread=spread).kosten_per_been(lot)

# DE VIJF KLOKKEN, OP EEN PLEK.
#
# Gevraagd is "M1 M2 M3 M5 en M15". De eerste versie van dit bestand nam er
# DRIE -- M1, M5 en M15 -- terwijl de hypothese ernaast er vijf opsomde. Dat is
# precies de fout waar dit project telkens over struikelt: twee beschrijvingen
# van een regel, en het scherm toont het getal van de ene naast de trade van de
# andere. M2 en M3 eruit laten maakt de instapvoorwaarde bovendien losser dan
# gevraagd, dus meer trades met een zwakkere eis.
KLOKKEN: tuple[tuple[str, str], ...] = (
    ("M1", "1min"), ("M2", "2min"), ("M3", "3min"), ("M5", "5min"),
    ("M15", "15min"), ("M30", "30min"), ("M60", "60min"),
)


@dataclass(frozen=True)
class Instelling:
    """Eén configuratie. Er worden er 72 gedraaid; dat getal hoort bij de uitslag."""

    stap: float = 0.5
    #: None = onbegrensd, precies zoals gevraagd.
    max_benen: int | None = None
    #: None = geen mandstop. Anders: deel van de startbalans.
    mandstop_deel: float | None = None
    lot: float = 0.01
    #: Spread in punten. Blijft hier staan omdat het rooster hem varieert; de
    #: rest van de broker (marge, stop-out, slippage) komt uit `uitvoering`.
    spread: float = 0.14
    #: DE BROKER. Marge, stop-out-niveau, slippage, commissie, swap. Stond hier
    #: niet, en daarom mat de vorige versie een ladder die pas doodging bij
    #: balans nul -- terwijl de broker bij 50% margin level liquideert.
    uitvoering: Uitvoering = RAW_GOUD

    @property
    def uit(self) -> Uitvoering:
        """De broker met de spread van DEZE configuratie erin.

        Eén plek waar de twee bij elkaar komen, zodat er geen configuratie kan
        ontstaan waarin het rooster 2,0 spread zwaait en de kostenregel nog 0,14
        rekent. Dat soort stille tweespalt is deze week al twee keer langsgekomen.
        """

        return replace(self.uitvoering, spread=self.spread)

    @property
    def naam(self) -> str:
        benen = "open" if self.max_benen is None else f"max{self.max_benen}"
        stop = "geen" if self.mandstop_deel is None else f"{self.mandstop_deel:.0%}"
        return f"stap{self.stap:g}·{benen}·stop{stop}·lot{self.lot:g}"


@dataclass
class Mand:
    """Eén ladder van openen tot sluiten."""

    geopend: pd.Timestamp
    #: +1 = koopmand (bijkopen op de dip), -1 = verkoopmand (bijverkopen op de rally)
    kant: int = 1
    benen: list[float] = field(default_factory=list)
    #: Diepste onderwaterstand in euro, gemeten op elke bar.
    diepste_euro: float = 0.0
    diepste_punten: float = 0.0
    gesloten: pd.Timestamp | None = None
    resultaat_euro: float = 0.0
    #: True wanneer de mand door de mandstop is gesloten en niet in winst.
    afgekapt: bool = False
    #: True wanneer de BROKER hem sloot: margin level onder het stop-out-niveau.
    #: Verschilt van `afgekapt` omdat dit niet jouw keuze was.
    uitgegooid: bool = False
    #: True wanneer de margin call het bijvullen stilzette. De mand loopt door,
    #: maar niet meer als ladder -- en dat is iets anders dan wat gemeten wordt.
    bevroren: bool = False

    @property
    def aantal_benen(self) -> int:
        return len(self.benen)

    @property
    def bars_onder_water(self) -> int:
        return self._bars

    @property
    def minuten_geblokkeerd(self) -> int:
        """Minuten dat deze mand bevroren stond en dus alles tegenhield."""

        return self._bevroren_bars

    _bars: int = 0
    _bevroren_bars: int = 0


def stapel_richting(frames: dict[str, pd.DataFrame], stamp: pd.Timestamp) -> int:
    """+1 als alle klokken omhoog staan, -1 als ze allemaal omlaag staan, anders 0.

    DE FOUT DIE DE EIGENAAR VOND, EN HIJ IS GROOT.
    ============================================================
    Hier stond alleen `stapel_omhoog`. Die sectie KON dus niet shorten -- hij
    kocht of hij deed niets.

    Dat maakte de hele meting waardeloos zonder dat er iets aan de rekensom
    mankeerde. Goud ging in het gemeten venster van 2570 naar 4385, ruim 70%
    omhoog. Een regel die alleen koopt komt in zo'n markt bij elke terugval
    vanzelf goed: de mand loopt weg, je koopt bij, en de trend haalt je terug.
    Dat is geen strategie maar de stijging, en het verklaart de EUR 132.847.

    Wat hij vroeg was TRENDVOLGEND: "als de markt downtrend is voor M1 en M2 en
    M3 en M5 en M15 en M30 en M60, dan moet hij verkopen." Precies het
    spiegelbeeld, en dat stond er niet in.

    EN ER KWAMEN TWEE KLOKKEN BIJ. Hij noemde M30 en M60; ik had er vijf. Met
    zeven klokken is de instapeis strenger en zijn er dus minder maar schonere
    momenten.
    """

    kanten = set()
    for frame in frames.values():
        pos = frame.index.searchsorted(stamp, side="right") - 1
        if pos < 1:
            return 0
        rij, vorige = frame.iloc[pos], frame.iloc[pos - 1]
        if rij["close"] > rij["open"] and rij["close"] > vorige["close"]:
            kanten.add(1)
        elif rij["close"] < rij["open"] and rij["close"] < vorige["close"]:
            kanten.add(-1)
        else:
            return 0
        if len(kanten) > 1:
            return 0
    return kanten.pop() if kanten else 0


def stapel_omhoog(frames: dict[str, pd.DataFrame], stamp: pd.Timestamp) -> bool:
    """Zijn M1, M2, M3, M5 en M15 het alle vijf eens over omhoog?

    DE LAATSTE GESLOTEN BAR EN NIET DE LOPENDE. Een klok die met de lopende bar
    meebeweegt, weet op het moment van instappen dingen die je live niet hebt.
    Dat is dezelfde vooruitkijkfout die dit project al eens een week gekost
    heeft, en hij is hier extra verleidelijk omdat de M1 zo vaak ververst.
    """

    for frame in frames.values():
        pos = frame.index.searchsorted(stamp, side="right") - 1
        if pos < 1:
            return False
        rij = frame.iloc[pos]
        vorige = frame.iloc[pos - 1]
        if not (rij["close"] > rij["open"] and rij["close"] > vorige["close"]):
            return False
    return True


def _loopt(uur: int, start: int, eind: int) -> bool:
    """Een sessievenster mag over middernacht heen lopen, en Asia doet dat."""

    if start < eind:
        return start <= uur < eind
    return uur >= start or uur < eind


def _sessie_van(stamp: pd.Timestamp) -> str:
    uur = stamp.hour
    actief = [naam for naam, (start, eind) in SESSIES.items() if _loopt(uur, start, eind)]
    if all(naam in actief for naam in OVERLAP):
        return "overlap"
    return actief[0] if actief else "buiten"


def _pnl_euro(benen: list[float], prijs: float, lot: float, kant: int = 1) -> float:
    """Open resultaat van de hele mand, kosten nog niet afgetrokken.

    EXPLICIET PER KANT en niet met een tekentruc. Een koopmand verdient als de
    prijs boven de benen staat; een verkoopmand als hij eronder staat. Dat met
    een `* kant` oplossen is precies hoe je hier een omgedraaid teken in stopt
    dat niemand meer terugvindt -- dezelfde afweging als in sectie 21.
    """

    if kant > 0:
        return sum(prijs - been for been in benen) * lot * CONTRACT
    return sum(been - prijs for been in benen) * lot * CONTRACT


def simuleer(
    m1: pd.DataFrame,
    stapels: dict[str, pd.DataFrame],
    *,
    instelling: Instelling,
    balans: float,
    nieuws: pd.DatetimeIndex | None = None,
) -> tuple[list[Mand], bool]:
    """Draai de ladder over de M1-bars. Geeft de manden en of de rekening omviel.

    DE RUINE-CONTROLE STAAT IN DE LUS EN NIET IN HET RAPPORT ACHTERAF. Een mand
    die de rekening halverwege opblaast, sluit niet meer -- de posities worden
    door de broker geliquideerd. Achteraf sommeren alsof hij netjes uitkwam,
    zou van een ruine een gewone verliezer maken.
    """

    manden: list[Mand] = []
    mand: Mand | None = None
    kapot = False
    uitv = instelling.uit
    kosten = uitv.kosten_per_been(instelling.lot)
    nieuws_set = set(nieuws or [])

    # DE BALANS LOOPT MEE, EN DAT DEED HIJ NIET.
    # ========================================================================
    # Hier stond `balans` als vaste startwaarde in elke marge-controle, terwijl
    # `eindresultaat._loop_balans` diezelfde manden BUITEN deze functie tegen
    # een GROEIENDE balans afrekende. Twee rekeningen in een getal: de stop-out
    # keek naar EUR 59,16 terwijl het rapport EUR 2.945 claimde.
    #
    # Dat is precies de tweespalt die deze week al drie keer is langsgekomen --
    # dubbele spread, forex-commissie op goud, long-only ladder. Eén bedrag, op
    # één plek, dat meebeweegt met wat er werkelijk gebeurde.
    stand = balans

    for stamp, bar in m1.iterrows():
        if kapot:
            break

        if mand is None:
            if stamp.floor("h") in nieuws_set:
                continue
            # BEIDE KANTEN. Alle klokken omhoog -> kopen en bijkopen op de dip.
            # Alle klokken omlaag -> verkopen en bijverkopen op de rally.
            kant = stapel_richting(stapels, stamp)
            if kant == 0:
                continue
            mand = Mand(geopend=stamp, kant=kant, benen=[float(bar["open"])])
            # EN DAN VALT HIJ DOOR NAAR HET BEHEER VAN DEZELFDE BAR.
            #
            # De eerste versie sprong hier met `continue` naar de volgende bar.
            # Dat is te vriendelijk: je stapt in op de open, en als die bar
            # vervolgens tien punten wegzakt zijn die benen in werkelijkheid
            # wél gevuld. Overslaan maakt elke ladder één bar ondieper dan hij
            # was, en juist de eerste bar is degene waarin een instap op de
            # verkeerde plek zich meteen wreekt.

        mand._bars += 1
        # HOELANG DE REKENING STILSTOND. Een bevroren mand sluit nooit uit
        # zichzelf, en zolang hij openstaat opent er geen nieuwe -- er valt
        # niets om, er gebeurt alleen niets meer. Precies dat verzweeg het
        # rapport: 4.282 manden in drie weken en daarna 700 dagen stilte,
        # gepresenteerd als +4779%.
        if mand.bevroren:
            mand._bevroren_bars += 1

        # 1. EERST BIJVULLEN, en pas daarna op winst kijken. Dit is de
        #    pessimistische volgorde en ze is met opzet gekozen.
        #
        #    Bij een KOOPmand liggen de volgende benen LAGER en vult de low ze;
        #    bij een VERKOOPmand liggen ze HOGER en vult de high ze.
        #    EN DE MARGIN CALL KAN DE LADDER MIDDEN IN HET BIJVULLEN STILZETTEN.
        #    Onder 100% margin level weigert de broker nieuwe posities. Wat er
        #    dan overblijft is geen ladder meer maar een stapel verliezers die
        #    op eigen kracht terug moet -- en dat is een ander mechanisme dan
        #    het mechanisme dat hier gemeten wordt, dus het hoort zichtbaar te
        #    zijn en niet stilletjes doorgerekend.
        laatste = mand.benen[-1]
        benen_voor = len(mand.benen)

        def _mag_erbij(prijs_nu: float) -> bool:
            zwevend_nu = _pnl_euro(mand.benen, prijs_nu, instelling.lot, mand.kant)
            zwevend_nu -= kosten * len(mand.benen)
            marge_nu = uitv.marge_voor(
                instelling.lot, mand.benen[0], benen=len(mand.benen))
            return uitv.mag_bijopenen(stand + zwevend_nu, marge_nu)

        if mand.kant > 0:
            while float(bar["low"]) <= laatste - instelling.stap:
                if (instelling.max_benen is not None
                        and len(mand.benen) >= instelling.max_benen):
                    break
                if not _mag_erbij(laatste - instelling.stap):
                    mand.bevroren = True
                    break
                laatste = laatste - instelling.stap
                mand.benen.append(laatste)
        else:
            while float(bar["high"]) >= laatste + instelling.stap:
                if (instelling.max_benen is not None
                        and len(mand.benen) >= instelling.max_benen):
                    break
                if not _mag_erbij(laatste + instelling.stap):
                    mand.bevroren = True
                    break
                laatste = laatste + instelling.stap
                mand.benen.append(laatste)

        # 2. Diepste stand van deze bar, op de LOW -- dat is waar de rekening
        #    het krapst stond, en niet op de close.
        slechtste_prijs = float(bar["low"]) if mand.kant > 0 else float(bar["high"])
        onder = _pnl_euro(mand.benen, slechtste_prijs, instelling.lot, mand.kant)
        onder -= kosten * len(mand.benen)
        if onder < mand.diepste_euro:
            mand.diepste_euro = onder
            mand.diepste_punten = sum(
                (slechtste_prijs - been) * mand.kant for been in mand.benen
            ) / len(mand.benen)

        # 3. DE BROKER GAAT VOOR ALLES, EN HIJ WACHT NIET OP BALANS NUL.
        #
        #    Hier stond `balans + onder <= 0`. Dat is de regel voor iemand
        #    zonder broker. Echt gaat het zo: de broker houdt marge in per been,
        #    dus bij een ladder GROEIT de inhouding terwijl je eigen vermogen
        #    ZAKT -- twee bewegingen naar elkaar toe. Zodra eigen vermogen
        #    gedeeld door marge onder het stop-out-niveau komt, liquideert hij,
        #    en op EUR 59 met 0,01 lot goud gebeurt dat na 9 benen: vier en een
        #    halve punt. Dat is een rustig half uur op goud.
        #
        #    Het verlies is dan niet netjes `onder`. Liquidatie is een
        #    marktorder op alle benen tegelijk, in precies het slechtste moment,
        #    dus de slippage komt er nog bij.
        marge = uitv.marge_voor(instelling.lot, mand.benen[0], benen=len(mand.benen))
        if uitv.vliegt_eruit(stand + onder, marge):
            mand.gesloten = stamp
            mand.resultaat_euro = max(
                -stand, onder - uitv.slippage_kosten(instelling.lot, len(mand.benen)))
            mand.afgekapt = True
            mand.uitgegooid = True
            manden.append(mand)
            kapot = True
            break

        # 4. Mandstop, als die er is. Ook dat is een marktorder, dus ook die slipt.
        slip = uitv.slippage_kosten(instelling.lot, len(mand.benen))
        if instelling.mandstop_deel is not None:
            grens = -abs(instelling.mandstop_deel) * stand
            if onder <= grens:
                mand.gesloten = stamp
                mand.resultaat_euro = onder - slip
                mand.afgekapt = True
                manden.append(mand)
                stand += mand.resultaat_euro
                mand = None
                continue

        # 5. En pas nu: staat de mand op de HIGH in winst?
        #
        #    DE DREMPEL IS DE SLIPPAGE EN NIET NUL, en dat is geen strengheid
        #    maar de enige samenhangende keuze. Sluiten is een marktorder op
        #    alle benen tegelijk; die slipt. Een mand die op +EUR 0,01 dichtgaat
        #    komt er netto negatief uit, en dit mechanisme leeft van precies
        #    zulke flinterdunne winsten -- bij 200.000 benen is dat het verschil
        #    tussen werken en niet werken.
        #
        #    De slippage wordt EEN keer geteld: hij zit in de drempel omdat de
        #    bot zijn eigen kosten kent, en hij gaat van het resultaat af omdat
        #    hij werkelijk betaald wordt. Twee keer aftrekken zou de ladder
        #    straffen voor het feit dat hij kan rekenen.
        #    EN DIT IS DE ZWAARSTE REGEL VAN HET HELE BESTAND.
        #    ================================================================
        #    Heeft deze bar NIEUWE BENEN gevuld, dan mag hij NIET ook op zijn
        #    gunstige uiterste sluiten. Dat deed hij wel, en het verzon geld.
        #
        #    Bewijs, een enkele bar: open 4000, hoog 4000, laag 3996, slot 3996.
        #    De prijs gaat alleen maar OMLAAG en eindigt op 3996. De ladder
        #    vulde zes benen op de weg naar beneden en sloot ze in diezelfde bar
        #    af op 4000 -- de hoogste prijs van de bar, die de OPENING was, van
        #    vóór de daling. Resultaat: +EUR 6,06 uit een bar waarin niets
        #    terugkwam.
        #
        #    Binnen een bar weet niemand de volgorde. Benen vullen op de low en
        #    dan afrekenen op de high is niet pessimistisch maar het gunstigst
        #    denkbare pad, elke bar opnieuw. Op M1-goud met halve punten vuurt
        #    dat aan de lopende band, en dat is precies wat "4.282 manden,
        #    gemiddeld EUR 0,66, +4779% in drie weken" was.
        #
        #    Dus: bar die benen vulde -> afrekenen op het SLOT, dat we echt
        #    hebben gezien. Bar zonder nieuwe benen -> het uiterste mag, want
        #    dan is er geen volgorde om over te liegen.
        vulde_benen = len(mand.benen) > benen_voor
        if vulde_benen:
            beste_prijs = float(bar["close"])
        else:
            beste_prijs = float(bar["high"]) if mand.kant > 0 else float(bar["low"])
        boven = _pnl_euro(mand.benen, beste_prijs, instelling.lot, mand.kant)
        boven -= kosten * len(mand.benen)
        if boven > slip:
            mand.gesloten = stamp
            mand.resultaat_euro = boven - slip
            manden.append(mand)
            stand += mand.resultaat_euro
            mand = None

    # DE MAND DIE AAN HET EIND NOG OPENSTAAT, EN DIT IS GEEN DETAIL.
    #
    # De eerste versie liet hem vallen. Dat is precies de fout die een grid op
    # papier onverslaanbaar maakt: een mand sluit alleen wanneer hij in winst
    # komt, dus alles wat NIET terugkwam blijft open -- en als je alleen de
    # gesloten manden telt, is elke mand in de uitslag een winnaar.
    #
    # Dat is overlevingsselectie in zijn zuiverste vorm, en bij dit mechanisme
    # is de nog-open mand juist degene waar het antwoord in zit. Hij gaat dus
    # mee, gewaardeerd op de laatste koers, met zijn diepste stand erbij.
    if mand is not None and not kapot:
        slot = float(m1.iloc[-1]["close"])
        mand.resultaat_euro = _pnl_euro(mand.benen, slot, instelling.lot, mand.kant)
        mand.resultaat_euro -= kosten * len(mand.benen)
        mand.gesloten = None
        manden.append(mand)

    return manden, kapot


def rapport(manden: list[Mand], kapot: bool, *, balans: float) -> dict[str, object]:
    """De volgorde is de boodschap: ruine en diepte vóór trefkans en winst."""

    if not manden:
        return {"manden": 0}

    resultaten = pd.Series([m.resultaat_euro for m in manden])
    diepste = pd.Series([m.diepste_euro for m in manden])
    benen = pd.Series([m.aantal_benen for m in manden])
    ergste = min(manden, key=lambda m: m.diepste_euro)

    return {
        # 1-5: wat een grid stukmaakt.
        "ruine": kapot,
        "diepste_euro": float(diepste.min()),
        "diepste_deel_balans": float(diepste.min()) / balans,
        "meeste_benen": int(benen.max()),
        "langste_onder_water_bars": int(max(m.bars_onder_water for m in manden)),
        "ergste_mand": {
            "geopend": str(ergste.geopend),
            "benen": ergste.aantal_benen,
            "diepste_euro": round(ergste.diepste_euro, 2),
            "resultaat_euro": round(ergste.resultaat_euro, 2),
        },
        "afgekapt": int(sum(1 for m in manden if m.afgekapt)),
        # 5b. WAT DE BROKER DEED, en dat is iets anders dan wat jij deed.
        #     `uitgegooid` is liquidatie op margin level -- niet jouw mandstop
        #     en niet balans nul. `bevroren` is de zone ervoor, waarin de ladder
        #     niet meer mocht bijvullen en dus geen ladder meer was.
        "uitgegooid_door_broker": int(sum(1 for m in manden if m.uitgegooid)),
        "bevroren_door_margin_call": int(sum(1 for m in manden if m.bevroren)),
        # 5c. HOELANG DE REKENING STILSTOND, EN DIT ONTBRAK HELEMAAL.
        #
        #     De uitslag zei "+4779%, 4.282 manden" over 730 dagen. Wat er niet
        #     stond: de laatste mand bevroor in week drie en bleef daarna open,
        #     en zolang die openstaat opent er geen nieuwe. Er viel niets om, er
        #     gebeurde alleen niets meer -- en dat las als winst.
        #
        #     Staat er een datum bij `handel_gestopt_op`, dan is het eindbedrag
        #     de stand van DIE dag en niet van het eind van de meting.
        #
        #     EN DE BLOKKADE VRAAGT GEEN MARGIN CALL. Elke mand die nooit meer
        #     sluit houdt alles tegen, bevroren of niet -- er opent er namelijk
        #     geen nieuwe zolang deze openstaat. Mijn eerste versie keek alleen
        #     naar `bevroren` en miste daarmee de gewone variant: een mand van
        #     een enkel been dat gewoon nooit meer in winst kwam.
        "handel_gestopt_op": (
            str(manden[-1].geopend) if manden[-1].gesloten is None else None),
        "dagen_stil_aan_het_eind": round(
            manden[-1].bars_onder_water / 1440.0, 2
        ) if manden[-1].gesloten is None else 0.0,
        "dagen_bevroren": round(
            sum(m.minuten_geblokkeerd for m in manden) / 1440.0, 2),
        # 6: en pas hierna het vrolijke deel.
        "manden": len(manden),
        "trefkans": float((resultaten > 0).mean()),
        "netto_euro": float(resultaten.sum()),
        "per_mand_euro": float(resultaten.mean()),
        "mediaan_benen": float(benen.median()),
    }


def per_groep(manden: list[Mand], sleutel) -> pd.DataFrame:
    """Uitsplitsing per jaar, sessie of uur -- alle drie apart, zoals gevraagd."""

    if not manden:
        return pd.DataFrame()
    rijen = [
        {
            "groep": sleutel(m),
            "resultaat": m.resultaat_euro,
            "diepste": m.diepste_euro,
            "benen": m.aantal_benen,
        }
        for m in manden
    ]
    frame = pd.DataFrame(rijen)
    uit = frame.groupby("groep").agg(
        manden=("resultaat", "size"),
        netto=("resultaat", "sum"),
        per_mand=("resultaat", "mean"),
        trefkans=("resultaat", lambda s: float((s > 0).mean())),
        diepste=("diepste", "min"),
        meeste_benen=("benen", "max"),
    )
    return uit.sort_values("netto", ascending=False)


def rooster() -> list[Instelling]:
    """De 72 configuraties uit de vooraf vastgelegde hypothese."""

    uit = []
    for stap in (0.5, 1.0, 2.0):
        for max_benen in (None, 5, 10, 20):
            for mandstop in (None, 0.02, 0.05):
                for lot in (0.01, 0.49):
                    uit.append(
                        Instelling(
                            stap=stap, max_benen=max_benen, mandstop_deel=mandstop, lot=lot
                        )
                    )
    return uit


# ============================================================================
#  DE BRON VAN DE BARS
# ============================================================================
#
#  MT5 IS WINDOWS-ONLY, en dat maakt elke meting afhankelijk van een machine
#  die toevallig aanstaat. Daarom twee wegen naar dezelfde bars:
#
#    --csv        een uitgevoerd bestand, overal te draaien
#    (standaard)  rechtstreeks uit de terminal
#
#  De CSV wordt geschreven door exporteer-bars.cmd en is hetzelfde formaat dat
#  `backtesting.replay.archive_frame` al gebruikte: een index `time` in UTC met
#  open/high/low/close. Een uitgevoerd bestand is bovendien HERHAALBAAR -- twee
#  runs op verschillende dagen meten dan dezelfde bars.


def _lees_csv(pad: str) -> pd.DataFrame:
    frame = pd.read_csv(pad, index_col="time", parse_dates=["time"])
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    ontbreekt = {"open", "high", "low", "close"} - set(frame.columns)
    if ontbreekt:
        raise SystemExit(f"  De CSV mist kolommen: {sorted(ontbreekt)}")
    return frame.sort_index()


def _hersample(m1: pd.DataFrame, regel: str) -> pd.DataFrame:
    """M5 en M15 uit M1 opbouwen in plaats van apart ophalen.

    EEN BRON IS BETER DAN DRIE. Losse reeksen ophalen geeft reeksen met eigen
    gaten en eigen laatste bar, en dan kijkt de stapel op het ene tijdframe
    naar een kaars die op het andere nog niet bestaat.
    """

    uit = m1.resample(regel, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    return uit.dropna()


def _haal_uit_mt5(args) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    from datetime import datetime, timedelta

    from backtesting.replay import fetch_mt5_history
    from config.loader import load_credentials, load_settings, terminal_path_from_env
    from core.mt5_connector import MT5Connector
    from core.types import Timeframe

    settings = load_settings(overlay=args.config, env_overrides=False)
    # DE OVERLAY WERD GELADEN EN NIET GEBRUIKT. Het symbool moet hier doorheen,
    # anders vraag je "XAUUSD" aan een broker die het "XAUUSD.i" noemt en krijg
    # je niets terug -- precies de fout waar benen.cmd op stukliep.
    symbool = settings.instruments.broker_symbol(args.symbol)
    eind = datetime.now(UTC)
    start = eind - timedelta(days=args.days)
    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=False),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    connector.connect()
    try:
        m1 = fetch_mt5_history(connector, symbool, Timeframe.M1, start, eind)
    finally:
        connector.shutdown()
    stapels = {naam: _hersample(m1, regel) for naam, regel in KLOKKEN}
    return m1, stapels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--balans", type=float, required=True,
                        help="echte startbalans in euro -- de ruinekans hangt eraan")
    parser.add_argument("--alle-configs", action="store_true",
                        help="alle 72 draaien in plaats van alleen de gevraagde")
    # DE OVERLAY, EN EEN BESTAANDE BEWAKER VING DAT IK HEM VERGAT.
    #
    # `load_settings()` zonder overlay geeft de basisconfig, en daar staat geen
    # symbol_suffix, geen commissie, geen slippage en geen kostenplafond in.
    # Een onderzoeksrun zonder overlay beprijst dus een ANDERE REKENING dan de
    # rekening die handelt -- stil, en in de richting waarin de basisdefaults
    # toevallig wijzen.
    #
    # Dat is hier extra pijnlijk, want de hele uitslag van deze meting hangt
    # aan de kosten per been: een grid opent tien keer zoveel posities als een
    # gewone regel, dus tien keer zoveel commissie. De verkeerde config maakt
    # een ladder goedkoper dan hij is, en goedkoop is precies wat een grid
    # nodig heeft om op papier te winnen.
    parser.add_argument("--config", default="config/eightcap.yaml")
    parser.add_argument("--csv", help="uitgevoerde M1-bars in plaats van MT5")
    args = parser.parse_args()

    if args.csv:
        # DE BRUG. MT5 draait alleen op Windows; met een uitgevoerde CSV kan
        # deze meting overal draaien, ook waar geen terminal is.
        m1 = _lees_csv(args.csv)
        stapels = {naam: _hersample(m1, regel) for naam, regel in KLOKKEN}
    else:
        m1, stapels = _haal_uit_mt5(args)

    configs = rooster() if args.alle_configs else [Instelling()]
    print(f"\n  SECTIE 20 -- terugval-ladder op {args.symbol}, {args.days} dagen")
    print(f"  balans EUR {args.balans:.2f}   configuraties: {len(configs)}")
    print("  " + "-" * 70)
    for instelling in configs:
        manden, kapot = simuleer(m1, stapels, instelling=instelling, balans=args.balans)
        uit = rapport(manden, kapot, balans=args.balans)
        if not uit.get("manden"):
            print(f"  {instelling.naam:34s}  geen manden")
            continue
        vlag = "RUINE" if uit["ruine"] else "     "
        print(
            f"  {instelling.naam:34s} {vlag}  diepste EUR {uit['diepste_euro']:9.2f}"
            f"  ({uit['diepste_deel_balans']:+.0%})  benen {uit['meeste_benen']:3d}"
            f"  netto EUR {uit['netto_euro']:9.2f}  trefkans {uit['trefkans']:.1%}"
        )

    if len(configs) == 1 and manden:
        for naam, sleutel in (
            ("JAAR", lambda m: m.geopend.year),
            ("SESSIE", lambda m: _sessie_van(m.geopend)),
            ("UUR (UTC)", lambda m: m.geopend.hour),
        ):
            print(f"\n  PER {naam}")
            print(per_groep(manden, sleutel).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
