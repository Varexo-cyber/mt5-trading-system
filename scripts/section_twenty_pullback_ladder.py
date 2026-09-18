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
from dataclasses import dataclass, field
from datetime import UTC

import numpy as np
import pandas as pd

# Sessies in UTC. Ze mogen over middernacht heen lopen en dat doet Asia ook.
SESSIES: dict[str, tuple[int, int]] = {
    "asia": (23, 8),
    "londen": (7, 16),
    "newyork": (12, 21),
}
OVERLAP = ("londen", "newyork")

#: Wat één punt koers waard is per lot op XAUUSD: 100 troy ounce per lot.
CONTRACT = 100.0

# DE VIJF KLOKKEN, OP EEN PLEK.
#
# Gevraagd is "M1 M2 M3 M5 en M15". De eerste versie van dit bestand nam er
# DRIE -- M1, M5 en M15 -- terwijl de hypothese ernaast er vijf opsomde. Dat is
# precies de fout waar dit project telkens over struikelt: twee beschrijvingen
# van een regel, en het scherm toont het getal van de ene naast de trade van de
# andere. M2 en M3 eruit laten maakt de instapvoorwaarde bovendien losser dan
# gevraagd, dus meer trades met een zwakkere eis.
KLOKKEN: tuple[tuple[str, str], ...] = (
    ("M1", "1min"), ("M2", "2min"), ("M3", "3min"), ("M5", "5min"), ("M15", "15min"),
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
    #: Spread in punten, twee keer gerekend per been.
    spread: float = 0.16

    @property
    def naam(self) -> str:
        benen = "open" if self.max_benen is None else f"max{self.max_benen}"
        stop = "geen" if self.mandstop_deel is None else f"{self.mandstop_deel:.0%}"
        return f"stap{self.stap:g}·{benen}·stop{stop}·lot{self.lot:g}"


@dataclass
class Mand:
    """Eén ladder van openen tot sluiten."""

    geopend: pd.Timestamp
    benen: list[float] = field(default_factory=list)
    #: Diepste onderwaterstand in euro, gemeten op elke bar.
    diepste_euro: float = 0.0
    diepste_punten: float = 0.0
    gesloten: pd.Timestamp | None = None
    resultaat_euro: float = 0.0
    #: True wanneer de mand door de mandstop is gesloten en niet in winst.
    afgekapt: bool = False

    @property
    def aantal_benen(self) -> int:
        return len(self.benen)

    @property
    def bars_onder_water(self) -> int:
        return self._bars

    _bars: int = 0


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


def _pnl_euro(benen: list[float], prijs: float, lot: float) -> float:
    """Open resultaat van de hele mand, kosten nog niet afgetrokken."""

    return sum(prijs - been for been in benen) * lot * CONTRACT


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
    kosten_per_been = instelling.spread * 2 * instelling.lot * CONTRACT
    nieuws_set = set(nieuws or [])

    for stamp, bar in m1.iterrows():
        if kapot:
            break

        if mand is None:
            if stamp.floor("h") in nieuws_set:
                continue
            if not stapel_omhoog(stapels, stamp):
                continue
            mand = Mand(geopend=stamp, benen=[float(bar["open"])])
            # EN DAN VALT HIJ DOOR NAAR HET BEHEER VAN DEZELFDE BAR.
            #
            # De eerste versie sprong hier met `continue` naar de volgende bar.
            # Dat is te vriendelijk: je stapt in op de open, en als die bar
            # vervolgens tien punten wegzakt zijn die benen in werkelijkheid
            # wél gevuld. Overslaan maakt elke ladder één bar ondieper dan hij
            # was, en juist de eerste bar is degene waarin een instap op de
            # verkeerde plek zich meteen wreekt.

        mand._bars += 1

        # 1. EERST BIJVULLEN, en pas daarna op winst kijken. Dit is de
        #    pessimistische volgorde en ze is met opzet gekozen.
        laatste = mand.benen[-1]
        while float(bar["low"]) <= laatste - instelling.stap:
            if instelling.max_benen is not None and len(mand.benen) >= instelling.max_benen:
                break
            laatste = laatste - instelling.stap
            mand.benen.append(laatste)

        # 2. Diepste stand van deze bar, op de LOW -- dat is waar de rekening
        #    het krapst stond, en niet op de close.
        onder = _pnl_euro(mand.benen, float(bar["low"]), instelling.lot)
        onder -= kosten_per_been * len(mand.benen)
        if onder < mand.diepste_euro:
            mand.diepste_euro = onder
            mand.diepste_punten = sum(
                float(bar["low"]) - been for been in mand.benen
            ) / len(mand.benen)

        # 3. Ruine gaat voor alles. Staat het verlies onder de hele balans, dan
        #    is er geen mand meer om te sluiten.
        if balans + onder <= 0:
            mand.gesloten = stamp
            mand.resultaat_euro = -balans
            mand.afgekapt = True
            manden.append(mand)
            kapot = True
            break

        # 4. Mandstop, als die er is.
        if instelling.mandstop_deel is not None:
            grens = -abs(instelling.mandstop_deel) * balans
            if onder <= grens:
                mand.gesloten = stamp
                mand.resultaat_euro = onder
                mand.afgekapt = True
                manden.append(mand)
                mand = None
                continue

        # 5. En pas nu: staat de mand op de HIGH in winst?
        boven = _pnl_euro(mand.benen, float(bar["high"]), instelling.lot)
        boven -= kosten_per_been * len(mand.benen)
        if boven > 0:
            mand.gesloten = stamp
            mand.resultaat_euro = boven
            manden.append(mand)
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
        mand.resultaat_euro = _pnl_euro(mand.benen, slot, instelling.lot)
        mand.resultaat_euro -= kosten_per_been * len(mand.benen)
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
