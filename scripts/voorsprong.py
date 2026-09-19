"""Heeft dit instapmoment uberhaupt voorsprong? Een getal, per idee.

WAAROM DIT HET EERSTE SCRIPT IS EN NIET HET LAATSTE.
============================================================================
Sectie 23 draaide 1.948 trades over twee jaar goud en alle TWAALF
uitstapvarianten kwamen negatief uit -- break-even, trailen op 1, 1,5, 2 en 3
ATR, deels eruit, tijdstop, alles. Dat is geen uitstapprobleem. Als elke manier
om eruit te gaan verliest, dan had het moment van INSTAPPEN geen voorsprong, en
dan repareert geen enkele stop, ladder of hedge dat nog.

Die conclusie kostte een week bouwen. Dit bestand stelt diezelfde vraag in een
paar minuten, VOORDAT er iets omheen gebouwd wordt:

    beweegt de markt na dit moment anders dan na een willekeurig moment?

Geen strategie, geen stops, geen doelen, geen kosten. Alleen het signaal en wat
de prijs daarna deed. Kosten en uitvoering komen pas als er iets te meten valt;
een signaal zonder voorsprong wordt door kosten alleen maar erger.

DE DRIE VALKUILEN ZITTEN INGEBOUWD, want elk van de drie heeft deze week een
uitslag opgeblazen:

  * VOORUITKIJKEN. Elk signaal wordt opgebouwd uit GESLOTEN bars en dan nog een
    bar verschoven. `test_voorsprong.py` eist dat het oordeel op tijdstip T niet
    verandert als je alles vanaf T omgooit. De klokkenstapel las de slotkoers
    van de bar die nog liep en "verdiende" daarmee EUR 398.572.

  * HET MAXIMUM VAN RUIS. Twintig ideeen doorrekenen en het beste pakken is
    hetzelfde als twintig keer een munt opgooien en de mooiste reeks uitkiezen.
    Elk idee krijgt daarom een permutatietoets, en de uitslag noemt hoeveel
    ideeen er zijn getoetst.

  * GEEN NULHYPOTHESE. "+0,8 punt na dit signaal" zegt niets als de markt na
    ELK moment gemiddeld +0,8 punt doet. Daarom staat overal het verschil met
    de gewone achtergrondbeweging, niet het rendement zelf.

EN EEN TOETSDEEL DAT PAS AAN HET EIND OPENGAAT. De nieuwste 30% van de bars
wordt apart gehouden. Alles wat je op het oefendeel vindt is een hypothese; wat
op het toetsdeel overeind blijft is een bevinding.

    python scripts/voorsprong.py --csv runtime/xauusd_m1.csv
    python scripts/voorsprong.py --csv runtime/xauusd_m1.csv --toets
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Hoe ver vooruit gekeken wordt, in minuten. Een signaal dat op vijf minuten
#: werkt en op vier uur niet is iets anders dan een signaal dat overal werkt, en
#: dat verschil bepaalt welke kosten je je kunt veroorloven.
HORIZONNEN: tuple[int, ...] = (5, 15, 30, 60, 240)

#: Deel van de bars dat wordt vastgehouden tot het eind.
TOETSDEEL = 0.30


# ---------------------------------------------------------------------------
#  Bars
# ---------------------------------------------------------------------------

def lees_csv(pad: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(pad, index_col="time", parse_dates=["time"])
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    ontbreekt = {"open", "high", "low", "close"} - set(frame.columns)
    if ontbreekt:
        raise SystemExit(f"  De CSV mist kolommen: {sorted(ontbreekt)}")
    return frame.sort_index()


def splits(m1: pd.DataFrame, *, toets: bool) -> pd.DataFrame:
    """Oudste 70% om in te zoeken, nieuwste 30% om mee af te rekenen.

    De splitsing zit hier en niet bij de aanroeper, zodat er geen pad bestaat
    waarin per ongeluk op alles wordt gemeten.
    """

    grens = int(len(m1) * (1 - TOETSDEEL))
    return m1.iloc[grens:] if toets else m1.iloc[:grens]


# ---------------------------------------------------------------------------
#  Signalen. Elk geeft +1 (koop), -1 (verkoop) of 0 (niets) per bar.
# ---------------------------------------------------------------------------
#
#  DE REGEL WAAR ALLES HIER AAN MOET VOLDOEN: een signaal op bar T mag alleen
#  gegevens uit bar T-1 en eerder gebruiken. Elke functie hieronder eindigt
#  daarom op `.shift(1)`. Die ene regel is het verschil tussen een meting en
#  een fantasie, en hij is deze week EUR 398.572 waard gebleken.


def _sein(waarde: pd.Series) -> pd.Series:
    """Naar +1 / -1 / 0, en dan een bar opschuiven zodat bar T alleen weet wat
    er op T-1 al gesloten was."""

    return np.sign(waarde).fillna(0.0).shift(1).fillna(0.0).astype(int)


def signaal_momentum(m1: pd.DataFrame, lengte: int = 60) -> pd.Series:
    """Simpelste vorm: staat de prijs hoger dan `lengte` bars geleden?"""

    return _sein(m1["close"] - m1["close"].shift(lengte))


def signaal_terugkeer(m1: pd.DataFrame, lengte: int = 60,
                      sigma: float = 2.0) -> pd.Series:
    """Het omgekeerde: ver boven het gemiddelde -> verkopen, ver eronder -> kopen."""

    gem = m1["close"].rolling(lengte).mean()
    spreiding = m1["close"].rolling(lengte).std()
    afwijking = (m1["close"] - gem) / spreiding.replace(0.0, np.nan)
    ruw = pd.Series(0.0, index=m1.index)
    ruw[afwijking > sigma] = -1.0
    ruw[afwijking < -sigma] = 1.0
    return _sein(ruw)


def signaal_uitbraak(m1: pd.DataFrame, lengte: int = 240) -> pd.Series:
    """Boven de hoogste high van de afgelopen `lengte` bars, of onder de laagste."""

    hoog = m1["high"].rolling(lengte).max()
    laag = m1["low"].rolling(lengte).min()
    ruw = pd.Series(0.0, index=m1.index)
    ruw[m1["close"] >= hoog] = 1.0
    ruw[m1["close"] <= laag] = -1.0
    return _sein(ruw)


def signaal_klokkenstapel(m1: pd.DataFrame) -> pd.Series:
    """DE CONTROLE, en met opzet degene waarvan we het antwoord al weten.

    Dit is het signaal uit sectie 20: M1 tot en met M60 allemaal dezelfde kant
    op. Sectie 23 liet zien dat er geen voorsprong in zit. Komt daar hier iets
    ANDERS uit, dan meet dit bestand verkeerd -- en dat wil je weten voordat je
    de uitslag van een nieuw idee gelooft.
    """

    kanten = []
    for regel in ("1min", "2min", "3min", "5min", "15min", "30min", "60min"):
        bars = m1.resample(regel).agg(
            {"open": "first", "close": "last"}).dropna()
        omhoog = (bars["close"] > bars["open"]) & (bars["close"] > bars["close"].shift(1))
        omlaag = (bars["close"] < bars["open"]) & (bars["close"] < bars["close"].shift(1))
        kant = omhoog.astype(int) - omlaag.astype(int)
        # SHIFT OP DE EIGEN KLOK, en pas daarna terug naar M1. Een H1-bar is pas
        # bekend als hij AF is; hem op M1 projecteren zonder die shift laat elke
        # minuut binnen dat uur de uitkomst van dat uur zien.
        kanten.append(kant.shift(1).reindex(m1.index, method="ffill").fillna(0))
    samen = pd.concat(kanten, axis=1)
    eens = samen.apply(lambda r: r.iloc[0] if (r == r.iloc[0]).all() else 0, axis=1)
    return eens.fillna(0).astype(int)


def signaal_dagbereik(m1: pd.DataFrame) -> pd.Series:
    """Breekt hij door de high of low van GISTEREN?

    Een klassieker, en meetbaar zonder enige interpretatie: gisteren is af.
    """

    dag = m1.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    hoog = dag["high"].shift(1).reindex(m1.index, method="ffill")
    laag = dag["low"].shift(1).reindex(m1.index, method="ffill")
    ruw = pd.Series(0.0, index=m1.index)
    ruw[m1["close"] > hoog] = 1.0
    ruw[m1["close"] < laag] = -1.0
    return _sein(ruw)


SIGNALEN = {
    "momentum 60m": signaal_momentum,
    "terugkeer 2 sigma": signaal_terugkeer,
    "uitbraak 240m": signaal_uitbraak,
    "dagbereik gisteren": signaal_dagbereik,
    "klokkenstapel (controle)": signaal_klokkenstapel,
}


# ---------------------------------------------------------------------------
#  De meting
# ---------------------------------------------------------------------------

@dataclass
class Uitslag:
    naam: str
    horizon: int
    aantal: int
    met_signaal: float      # gemiddelde beweging MET de richting mee, in punten
    achtergrond: float      # dezelfde beweging na een willekeurige bar
    verschil: float
    p: float

    @property
    def regel(self) -> str:
        vlag = ""
        if self.aantal < 100:
            vlag = "  (te weinig signalen)"
        elif self.p <= 0.05:
            vlag = "  <<< p <= 0,05"
        return (f"  {self.horizon:>5}m  {self.aantal:>8,}  "
                f"{self.met_signaal:>+9.4f}  {self.achtergrond:>+11.4f}  "
                f"{self.verschil:>+9.4f}  {self.p:>6.3f}{vlag}")


def _vooruit(m1: pd.DataFrame, horizon: int) -> np.ndarray:
    """Beweging in punten van deze bar naar `horizon` bars later."""

    slot = m1["close"].to_numpy(dtype=float)
    vooruit = np.full(len(slot), np.nan)
    vooruit[:-horizon] = slot[horizon:] - slot[:-horizon]
    return vooruit


def meet(m1: pd.DataFrame, sein: pd.Series, *, naam: str, horizon: int,
         rondes: int = 1000, zaad: int = 20260919) -> Uitslag:
    """Wat doet de prijs na dit signaal, vergeleken met na een willekeurige bar?

    IN DE RICHTING VAN HET SIGNAAL. Een verkoopsignaal is goed als de prijs
    ZAKT, dus de beweging wordt met de richting vermenigvuldigd. Anders middelt
    een signaal dat beide kanten op werkt zichzelf netjes naar nul weg.

    De p-waarde komt uit een permutatie: de signaalmomenten worden willekeurig
    over de reeks verplaatst, met hetzelfde aantal en dezelfde verhouding koop
    en verkoop. Zo hoort toeval te presteren; is het echte getal daar niet
    beter dan, dan is er niets gevonden.
    """

    vooruit = _vooruit(m1, horizon)
    richting = sein.to_numpy(dtype=float)
    bruikbaar = ~np.isnan(vooruit)

    raak = bruikbaar & (richting != 0)
    if raak.sum() == 0:
        return Uitslag(naam, horizon, 0, 0.0, 0.0, 0.0, 1.0)

    met = float(np.mean(vooruit[raak] * richting[raak]))
    # DE ACHTERGROND, en dit is de nulhypothese die ontbrak. Dezelfde mix van
    # koop en verkoop, maar dan op willekeurige momenten.
    kanten = richting[raak]
    alle = vooruit[bruikbaar]
    rng = np.random.default_rng(zaad)
    achtergronden = np.empty(rondes)
    beter = 0
    for i in range(rondes):
        trek = rng.choice(alle, size=raak.sum(), replace=False)
        nep = float(np.mean(trek * rng.permutation(kanten)))
        achtergronden[i] = nep
        if nep >= met:
            beter += 1

    achtergrond = float(achtergronden.mean())
    return Uitslag(naam, horizon, int(raak.sum()), met, achtergrond,
                   met - achtergrond, (beter + 1) / (rondes + 1))


def rapport(m1: pd.DataFrame, *, toets: bool, rondes: int) -> list[str]:
    deel = splits(m1, toets=toets)
    kop = "TOETSDEEL (een keer, en dit telt)" if toets else "OEFENDEEL (hier mag je zoeken)"
    regels = [
        "",
        "  " + "=" * 78,
        f"   HEEFT DIT SIGNAAL VOORSPRONG?   --   {kop}",
        "  " + "=" * 78,
        "",
        f"   {len(deel):,} M1-bars   {deel.index[0]:%Y-%m-%d} t/m {deel.index[-1]:%Y-%m-%d}",
        f"   {len(SIGNALEN)} ideeen getoetst, elk op {len(HORIZONNEN)} horizonnen",
        "",
        "   'met signaal' en 'achtergrond' zijn de beweging IN DE RICHTING van",
        "   het signaal, in koerspunten. Het verschil is wat het signaal toevoegt",
        "   bovenop wat de markt sowieso al deed. Kosten zitten er NIET in.",
    ]

    alles: list[Uitslag] = []
    for naam, maak in SIGNALEN.items():
        sein = maak(deel)
        regels += [
            "",
            f"  {naam}",
            f"  {'horizon':>6}  {'signalen':>8}  {'met signaal':>9}  "
            f"{'achtergrond':>11}  {'verschil':>9}  {'p':>6}",
            f"  {'-' * 6}  {'-' * 8}  {'-' * 9}  {'-' * 11}  {'-' * 9}  {'-' * 6}",
        ]
        for horizon in HORIZONNEN:
            uit = meet(deel, sein, naam=naam, horizon=horizon, rondes=rondes)
            alles.append(uit)
            regels.append(uit.regel)

    # DE STRAF VOOR VEEL PROBEREN. Vijf ideeen keer vijf horizonnen is
    # vijfentwintig toetsen; bij p = 0,05 verwacht je er dan ruim EEN die er
    # goed uitziet zonder dat er iets is. Dat getal hoort naast de uitslag.
    toetsen = len(alles)
    gehaald = [u for u in alles if u.p <= 0.05 and u.aantal >= 100]
    regels += [
        "",
        "  " + "-" * 78,
        f"   {toetsen} toetsen gedaan. Bij p = 0,05 en puur toeval verwacht je er",
        f"   {toetsen * 0.05:.1f} die 'significant' lijken zonder dat er iets is.",
        f"   Gevonden: {len(gehaald)}.",
        "",
    ]
    if not gehaald:
        regels += [
            "   NIETS. Geen enkel idee doet meer dan de achtergrond.",
            "   Dat is een antwoord en geen mislukking: het scheelt een week bouwen",
            "   op een instap die niets voorspelt.",
        ]
    elif len(gehaald) <= toetsen * 0.05:
        regels += [
            "   Niet meer dan je bij toeval zou verwachten. Behandel dit als NIETS",
            "   tot het op het toetsdeel overeind blijft.",
        ]
    else:
        regels += ["   Meer dan toeval verwacht. De kandidaten:"]
        for u in sorted(gehaald, key=lambda x: x.p):
            regels.append(
                f"     {u.naam:<26} {u.horizon:>4}m   "
                f"verschil {u.verschil:+.4f} pt   p = {u.p:.3f}")
        if not toets:
            regels += [
                "",
                "   Dit is het OEFENDEEL. Dit zijn hypothesen, geen bevindingen.",
                "   Draai nu een keer met --toets en gebruik dat als antwoord.",
            ]
    regels.append("")
    return regels


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--toets", action="store_true",
                   help="meet op het vastgehouden deel. Een keer, aan het eind.")
    p.add_argument("--rondes", type=int, default=1000)
    p.add_argument("--uit", default="")
    args = p.parse_args()

    m1 = lees_csv(args.csv)
    regels = rapport(m1, toets=args.toets, rondes=args.rondes)
    for r in regels:
        print(r, flush=True)
    if args.uit:
        Path(args.uit).parent.mkdir(parents=True, exist_ok=True)
        Path(args.uit).write_text("\n".join(regels) + "\n", encoding="utf-8")
        print(f"  Ook opgeslagen in {args.uit}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
