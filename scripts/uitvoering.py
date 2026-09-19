"""Alles wat tussen "de strategie zegt koop" en "het geld op de rekening" staat.

TOT NU TOE REKENDE ELKE SECTIE ZIJN EIGEN KOSTEN, en dat is precies hoe twee
beschrijvingen van dezelfde regel uit elkaar gaan lopen. Erger: ze rekenden
alleen SPREAD, en daarmee ontbrak het enige mechanisme dat een ladder op een
kleine rekening werkelijk doodt -- de broker die je eruit gooit voordat je
balans op is.

VIER DINGEN STAAN HIER, EN DE VOLGORDE IS DE VOLGORDE VAN HUN GEWICHT:

  1. MARGE EN STOP-OUT. Je rekening gaat niet dood op balans nul. Hij gaat dood
     wanneer je EIGEN VERMOGEN onder een percentage van je INGEHOUDEN MARGE
     zakt, en bij een ladder groeit die marge met elk been terwijl het eigen
     vermogen zakt. Twee bewegingen naar elkaar toe. Dit is met afstand de
     grootste post die ik niet meerekende.

  2. SLIPPAGE. Een stop is een verzoek, geen garantie. Gemeten op deze rekening
     via `order_attempts.slippage_pips`; wat hier staat is een terugval voor
     als die tabel nog leeg is.

  3. SPREAD. Raw account, dus klein -- maar bij een grid met 200.000 benen is
     klein keer heel vaak nog steeds geld.

  4. SWAP. Nul op deze rekening, op gezag van de rekeninghouder: raw account,
     swapvrij. Het SYMBOOL draagt wel een swapregeling (-80,67 long, +23,83
     short per lot per nacht) en die staat in `gemeten_broker.json`, maar dat is
     de regeling van het instrument en niet van het account. Zou hij toch
     gelden, dan kost een nacht op 0,01 lot EUR 0,81 tegenover EUR 0,26 voor een
     heel rondje handelen -- daarom zit het als schakelaar in de meting en niet
     als stilzwijgende nul.

  5. COMMISSIE. Wel nul, en dat IS gemeten: `commission_by_asset_class` zet
     `metal: 0.0`, teruggerekend uit de deals van 24 augustus.

WAT HIER MET OPZET NIET IN ZIT: verzonnen precisie. Elk getal hieronder is
ofwel gemeten, ofwel een expliciet gemarkeerde terugval die `meet_uit_terminal`
overschrijft zodra MT5 bereikbaar is. Een aanname die zich voordoet als een
meting is erger dan een ontbrekende post, want niemand controleert hem nog.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Waar `meet_uit_terminal` zijn metingen neerlegt. Bestaat dit bestand, dan
#: gebruikt `laad()` het en zegt erbij dat het een meting is.
GEMETEN = ROOT / "config" / "gemeten_broker.json"

#: Eén punt koers per lot op XAUUSD: 100 troy ounce.
CONTRACT = 100.0


@dataclass(frozen=True)
class Uitvoering:
    """De broker, in getallen. Alles in KOERSPUNTEN, niet in pips.

    Pips zijn bij goud geen afspraak maar een ruzie -- de ene broker noemt 0,10
    een pip en de andere 0,01. `core/instrument.py` kiest daarom punt = pip voor
    alles wat geen forex is, en dit bestand rekent consequent in punten koers.
    Eén punt = één dollar goud = EUR 1,00 per 0,01 lot.
    """

    #: Spread in punten. Raw account, dus smal. Wordt overschreven door de
    #: gemeten spread zodra de terminal is uitgelezen.
    spread: float = 0.14

    #: Slippage in punten op een MARKTORDER. Terugval, geen meting: zie
    #: `slippage_uit_db`, die dit vervangt door de mediaan van je eigen fills.
    slippage: float = 0.10

    #: Nul op goud, gemeten uit de deals. Staat hier alleen zodat de formule
    #: klopt als dezelfde module ooit forex moet rekenen.
    commissie_per_lot_per_kant: float = 0.0

    #: NUL, OP GEZAG VAN DE REKENINGHOUDER. Hij heeft een raw account zonder
    #: swap en heeft dat twee keer bevestigd.
    #:
    #: `symbol_info("XAUUSD").swap_long` gaf -80,67 en `swap_short` +23,83,
    #: en dat staat ook in `config/gemeten_broker.json`. Maar dat veld is de
    #: swapREGELING VAN HET SYMBOOL, niet van het account: bij een swapvrije
    #: rekening blijft dat getal gewoon staan terwijl er niets wordt geboekt.
    #:
    #: Wat het definitief beslecht is de dealhistorie -- net zoals de
    #: commissie is vastgesteld uit negen echte trades. Tot die meting er is
    #: geldt wat de rekeninghouder zegt, en `robot.py --swap-schema` laat
    #: zien wat het zou kosten als hij zich vergist.
    swap_long_per_lot: float = 0.0
    swap_short_per_lot: float = 0.0

    #: 1:500. Bepaalt de marge per been en daarmee wanneer je eruit vliegt.
    hefboom: float = 500.0

    #: De broker liquideert wanneer eigen vermogen / marge hieronder zakt.
    #: 0,50 = het gangbare 50%-niveau.
    stop_out_niveau: float = 0.50

    #: Hieronder waarschuwt de broker (margin call). Hij sluit nog niets, maar
    #: je kunt ook niets meer bijopenen -- en dat breekt een ladder al.
    margin_call_niveau: float = 1.00

    #: Zet `False` om terug te vallen op het oude gedrag (alleen balans <= 0).
    #: Bestaat om te kunnen LATEN ZIEN wat het verschil is, niet om uit te zetten.
    stopout_actief: bool = True

    #: Waar deze getallen vandaan komen. Verschijnt in elk rapport.
    herkomst: str = "terugval (niet gemeten)"

    # -- kosten ------------------------------------------------------------

    def kosten_per_been(self, lot: float) -> float:
        """Wat één been kost van openen tot sluiten: spread + commissie.

        DE SPREAD TELT ÉÉN KEER, EN DAT IS EEN CORRECTIE. Sectie 20 en 21
        rekenden `spread * 2` per been. Dat is dubbel geteld: de spread IS het
        gat tussen bied en laat, en je betaalt het één keer per rondje. Je koopt
        op de laatkoers en verkoopt op de biedkoers, en de bars uit MT5 zijn
        biedkoersen -- dus `(uit - in) - spread`, één spread.

        `runner/service.py::_round_trip_cost_price` doet het live precies zo, en
        zegt er in zoveel woorden bij dat de spread er niet nog eens bij mag
        "omdat dat hem twee keer zou tellen". Mijn meting deed wat de live-code
        expliciet vermijdt.

        Deze correctie gaat de ANDERE kant op dan de stop-out die er nu bij
        komt: de spread was te zwaar belast, de marge helemaal niet. Welke van
        de twee zwaarder weegt, moet de meting zeggen en niet ik.

        Commissie telt WEL twee keer -- dat is een vast bedrag per kant. Op goud
        is het nul, dus hier verandert het niets.

        Slippage zit er NIET in, want die hangt af van HOE je eruit gaat. Een
        been dat met een limietorder wordt gevuld slipt niet; een marktexit wel,
        en een stop-out het hardst. Dat onderscheid wegmiddelen zou de ladder
        te zwaar belasten op de instap en te licht op de uitstap.
        """

        return (self.spread * lot * CONTRACT
                + self.commissie_per_lot_per_kant * lot * 2)

    def slippage_kosten(self, lot: float, benen: int = 1) -> float:
        """Wat een MARKTEXIT van `benen` posities extra kost."""

        return self.slippage * lot * CONTRACT * benen

    def swap_kosten(self, lot: float, nachten: int, *, richting: int = 1,
                    benen: int = 1) -> float:
        """Financiering per nacht. POSITIEF betekent dat het je geld KOST.

        Per richting verschillend, en dat verschil is enorm: een long betaalt
        80,67 per lot per nacht, een short KRIJGT 23,83. Bij goud is dat geen
        detail maar de reden dat een regel die over de nacht heen gaat aan de
        longkant bijna niet kan werken en aan de shortkant een meewind heeft.
        """

        per_nacht = (self.swap_long_per_lot if richting > 0
                     else self.swap_short_per_lot)
        return -per_nacht * lot * nachten * benen

    # -- marge -------------------------------------------------------------

    def marge_voor(self, lot: float, prijs: float, benen: int = 1) -> float:
        """Ingehouden marge voor `benen` posities van `lot` op `prijs`.

        MT5 rekent CFD-marge als contractgrootte x volume x prijs / hefboom.
        Op 0,01 lot goud van 4000 bij 1:500 is dat EUR 8,00 per been -- en op
        een rekening van EUR 59 is dat het getal waar alles om draait.
        """

        return CONTRACT * lot * prijs / self.hefboom * benen

    def margin_level(self, eigen_vermogen: float, marge: float) -> float:
        """Eigen vermogen gedeeld door ingehouden marge. Oneindig zonder posities."""

        if marge <= 0:
            return math.inf
        return eigen_vermogen / marge

    def vliegt_eruit(self, eigen_vermogen: float, marge: float) -> bool:
        """Liquideert de broker nu?

        DIT IS DE REGEL DIE ONTBRAK. De oude controle was `balans + zwevend <= 0`
        en die is te vriendelijk met precies de factor waar het om gaat: bij
        50% stop-out ben je weg terwijl er nog een half maandsalaris aan marge
        tegenover staat.
        """

        if not self.stopout_actief:
            return eigen_vermogen <= 0
        return self.margin_level(eigen_vermogen, marge) < self.stop_out_niveau

    def mag_bijopenen(self, eigen_vermogen: float, marge: float) -> bool:
        """Boven de margin call mag je bij; eronder bevriest de ladder.

        Een ladder die niet meer mag bijkopen is geen ladder meer -- hij is een
        losse verliezende positie die op zijn eentje terug moet komen. Dat is
        een ander mechanisme dan het mechanisme dat gemeten wordt, en het hoort
        dus zichtbaar te zijn in de uitslag.
        """

        return self.margin_level(eigen_vermogen, marge) >= self.margin_call_niveau


#: De rekening waar dit project op draait: Eightcap raw, XAUUSD.
RAW_GOUD = Uitvoering()

#: Om te kunnen laten zien wat de stop-out doet: dezelfde broker zonder hem.
ZONDER_STOPOUT = replace(RAW_GOUD, stopout_actief=False,
                         herkomst="terugval, stop-out UIT (vergelijking)")


# ---------------------------------------------------------------------------
# De vraag waar het om gaat: hoeveel beweging houdt een rekening uit?
# ---------------------------------------------------------------------------

def benen_bij_beweging(punten: float, stap: float) -> int:
    """Hoeveel benen een ladder heeft na `punten` tegenbeweging."""

    if stap <= 0:
        raise ValueError("stap moet groter dan nul zijn")
    return int(punten / stap) + 1


def toestand_na_beweging(
    balans: float, punten: float, *, stap: float, lot: float, prijs: float,
    uitvoering: Uitvoering = RAW_GOUD,
) -> tuple[int, float, float, float]:
    """Benen, zwevend verlies, marge en margin level na `punten` tegen je in.

    ZONDER MARKTDATA, EN DAT IS HET PUNT. Een ladder met vaste stap heeft na een
    beweging van X punten een voorspelbaar aantal benen op voorspelbare
    afstanden. Wat een rekening uithoudt is dus rekenwerk, geen backtest -- en
    het antwoord verandert niet als de data toevallig een gunstig venster is.
    """

    n = benen_bij_beweging(punten, stap)
    # Been i staat op (i * stap) boven de huidige prijs, i = 0..n-1.
    verlies_punten = sum(i * stap for i in range(n))
    zwevend = -verlies_punten * lot * CONTRACT
    # Kosten van alle benen tellen mee: ze zijn al betaald.
    zwevend -= uitvoering.kosten_per_been(lot) * n
    marge = uitvoering.marge_voor(lot, prijs, benen=n)
    eigen = balans + zwevend
    return n, zwevend, marge, uitvoering.margin_level(eigen, marge)


def overleefde_beweging(
    balans: float, *, stap: float, lot: float, prijs: float,
    uitvoering: Uitvoering = RAW_GOUD, grens_punten: float = 2000.0,
) -> float:
    """Hoeveel punten tegen je in deze rekening uithoudt voordat hij eruit vliegt.

    In punten koers, dus in dollars goud: het getal dat je met een grafiek kunt
    vergelijken. Een normale goudDAG is 40 tot 80 punten.

    HET GETAL IS WAT HIJ NOG HAALT, niet waar hij omvalt, en dat scheelt precies
    één stap. De eerste versie gaf het punt waarOP de rekening eruit vloog; de
    omkering in `balans_voor_beweging` rekent uit wat je nodig hebt om een
    beweging nog te HALEN. Die twee lagen er één stap naast en de test die ze
    tegen elkaar houdt, viel er meteen over.
    """

    punten = 0.0
    while punten <= grens_punten:
        _, zwevend, marge, _ = toestand_na_beweging(
            balans, punten, stap=stap, lot=lot, prijs=prijs, uitvoering=uitvoering)
        if uitvoering.vliegt_eruit(balans + zwevend, marge):
            return max(0.0, punten - stap)
        punten += stap
    return math.inf


def balans_voor_beweging(
    punten: float, *, stap: float, lot: float, prijs: float,
    uitvoering: Uitvoering = RAW_GOUD,
) -> float:
    """Welke startbalans een tegenbeweging van `punten` net overleeft.

    Directe omkering van de stop-out-regel in plaats van zoeken:

        eigen vermogen  >=  stop_out_niveau x marge
        balans + zwevend >= stop_out_niveau x marge
        balans           >= stop_out_niveau x marge - zwevend

    (zwevend is negatief, dus dat tweede deel telt op.)
    """

    n = benen_bij_beweging(punten, stap)
    verlies = sum(i * stap for i in range(n)) * lot * CONTRACT
    kosten = uitvoering.kosten_per_been(lot) * n
    marge = uitvoering.marge_voor(lot, prijs, benen=n)
    drempel = uitvoering.stop_out_niveau if uitvoering.stopout_actief else 0.0
    return drempel * marge + verlies + kosten


# ---------------------------------------------------------------------------
# Meten in plaats van aannemen
# ---------------------------------------------------------------------------

def slippage_uit_db(db_pad: Path | str, symbool: str = "XAUUSD",
                    punt: float = 0.01) -> float | None:
    """Mediane slippage op dit symbool, uit je eigen fills.

    `order_attempts.slippage_pips` staat in PUNTEN (zie `core/instrument.py`:
    voor niet-forex is pip = punt). Bij XAUUSD met twee decimalen is een punt
    EUR 0,01 koers, dus het getal moet nog x `punt` om in koerspunten te komen
    waar de rest van dit bestand mee rekent.

    Geeft None wanneer er te weinig fills zijn. Dan is de terugval eerlijker
    dan een mediaan over drie waarnemingen.
    """

    pad = Path(db_pad)
    if not pad.exists():
        return None
    try:
        db = sqlite3.connect(f"file:{pad}?mode=ro", uri=True)
        rijen = db.execute(
            "SELECT slippage_pips FROM order_attempts "
            "WHERE ok = 1 AND slippage_pips IS NOT NULL AND symbol LIKE ?",
            (f"{symbool}%",),
        ).fetchall()
        db.close()
    except sqlite3.Error:
        return None

    waarden = sorted(abs(float(r[0])) for r in rijen)
    if len(waarden) < 5:
        return None
    midden = len(waarden) // 2
    mediaan = (waarden[midden] if len(waarden) % 2
               else (waarden[midden - 1] + waarden[midden]) / 2)
    return mediaan * punt


def laad(db_pad: Path | str | None = None) -> Uitvoering:
    """De uitvoering zoals hij nu bekend is, met zijn herkomst erbij.

    De herkomst gaat mee in elk rapport. Een meting op terugvalgetallen is niet
    fout, maar hij moet wel zeggen dat hij dat is -- dat is het verschil tussen
    een schatting en een schatting die zich voordoet als een meting.
    """

    uit = RAW_GOUD
    bronnen: list[str] = []

    if GEMETEN.exists():
        try:
            with open(GEMETEN, encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError):
            m = {}
        velden = {k: m[k] for k in
                  ("spread", "hefboom", "stop_out_niveau", "margin_call_niveau",
                   "commissie_per_lot_per_kant")
                  if k in m}
        if velden:
            uit = replace(uit, **velden)
            bronnen.append(f"terminal ({m.get('gemeten_op', 'onbekend')})")

    if db_pad is not None:
        gemeten = slippage_uit_db(db_pad)
        if gemeten is not None:
            uit = replace(uit, slippage=gemeten)
            bronnen.append("slippage uit eigen fills")

    return replace(uit, herkomst=" + ".join(bronnen) if bronnen
                   else "terugval (niet gemeten)")


def meet_uit_terminal(symbool: str = "XAUUSD") -> dict[str, object]:
    """Lees hefboom, spread, marge en swap uit de OPENSTAANDE terminal.

    Geen inlog, geen wachtwoord: `mt5.initialize()` haakt aan op de terminal die
    al draait en al is ingelogd. Draait alleen op de Windows-machine.
    """

    import MetaTrader5 as mt5  # noqa: PLC0415 -- alleen op Windows aanwezig

    if not mt5.initialize():
        raise RuntimeError(f"MT5 niet bereikbaar: {mt5.last_error()}")
    try:
        rekening = mt5.account_info()
        info = mt5.symbol_info(symbool)
        if info is None:
            mt5.symbol_select(symbool, True)
            info = mt5.symbol_info(symbool)
        if rekening is None or info is None:
            raise RuntimeError(f"geen gegevens voor {symbool}")

        tick = mt5.symbol_info_tick(symbool)
        spread = (tick.ask - tick.bid) if tick else info.spread * info.point
        marge_1lot = mt5.order_calc_margin(
            mt5.ORDER_TYPE_BUY, symbool, 1.0, tick.ask if tick else info.ask)

        uit: dict[str, object] = {
            "symbool": symbool,
            "gemeten_op": __import__("datetime").datetime.now(
                __import__("datetime").UTC).isoformat(timespec="seconds"),
            "spread": round(float(spread), 4),
            "punt": float(info.point),
            "digits": int(info.digits),
            "hefboom": float(rekening.leverage),
            "stop_out_niveau": float(rekening.margin_so_so) / 100.0,
            "margin_call_niveau": float(rekening.margin_so_call) / 100.0,
            "swap_long_per_lot": float(info.swap_long),
            "swap_short_per_lot": float(info.swap_short),
            "commissie_per_lot_per_kant": 0.0,
            "marge_per_lot": float(marge_1lot) if marge_1lot else None,
            "balans": float(rekening.balance),
            "valuta": rekening.currency,
        }
        # DE MARGE UIT DE TERMINAL IS DE WAARHEID, niet mijn formule. Staat hij
        # er, dan wordt de hefboom eruit teruggerekend zodat beide kloppen.
        if marge_1lot and tick:
            uit["hefboom_uit_marge"] = round(
                CONTRACT * 1.0 * tick.ask / float(marge_1lot), 1)
        return uit
    finally:
        mt5.shutdown()


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Meet de broker uit de terminal.")
    p.add_argument("--symbool", default="XAUUSD")
    p.add_argument("--db", default=str(ROOT / "runtime" / "journal.db"))
    args = p.parse_args()

    try:
        gemeten = meet_uit_terminal(args.symbool)
    except Exception as fout:  # noqa: BLE001 -- de reden hoort op het scherm
        print(f"\n  Terminal niet uitgelezen: {fout}")
        print("  De meting draait door op terugvalgetallen; die staan in het rapport.\n")
        return 1

    GEMETEN.parent.mkdir(parents=True, exist_ok=True)
    with open(GEMETEN, "w", encoding="utf-8") as f:
        json.dump(gemeten, f, indent=2)

    print(f"\n  Gemeten op {gemeten['symbool']}:")
    for sleutel in ("spread", "hefboom", "hefboom_uit_marge", "stop_out_niveau",
                    "margin_call_niveau", "marge_per_lot", "swap_long_per_lot",
                    "balans", "valuta"):
        if sleutel in gemeten:
            print(f"    {sleutel:<24} {gemeten[sleutel]}")

    slip = slippage_uit_db(args.db)
    print(f"    {'slippage (eigen fills)':<24} "
          + (f"{slip:.4f} punten" if slip is not None
             else "te weinig fills -- terugval 0,10"))
    print(f"\n  Opgeslagen in {GEMETEN}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
