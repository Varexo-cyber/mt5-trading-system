"""Sectie 22: een gepubliceerd trackrecord toetsen aan zijn eigen beweringen.

WAAROM DIT GEEN BACKTEST IS. Bij een bot die je wordt aangeboden ken je de
strategie niet -- de code is dicht. Wat je wel hebt zijn de GEPUBLICEERDE
CIJFERS en de BEWEERDE EIGENSCHAPPEN, en die twee moeten met elkaar te rijmen
zijn. Zijn ze dat niet, dan is er geen backtest nodig om dat vast te stellen.

Dit is dus geen strategiemeting maar een consistentietoets. Hij heeft geen
koersdata nodig en draait overal.

DE VIJF TOETSEN, en de derde is degene die beslist:

  1. REKENT HET UIT ZICHZELF OP? Profit factor, verwachting en totale winst
     zijn af te leiden uit trefkans, gemiddelde winst en gemiddeld verlies.
     Kloppen die niet, dan is er iets met de cijfers zelf.

  2. IS ER UBERHAUPT EEN VOORSPRONG? Trefkans zegt niets zonder de
     win/verlies-verhouding ernaast. 95% raak bij een verlies van vijf keer de
     winst is quitte bij 83,3%.

  3. ZIJN DE TRADES ONAFHANKELIJK? Dit is de scherpste. Wordt beweerd dat elke
     trade op zichzelf staat met een eigen stop, dan is een reeks van N trades
     zonder een enkele verliezer te toetsen: de kans is p^N. Komt daar een
     getal uit met tien nullen erachter, dan staan die trades niet los van
     elkaar -- hoe de bewering ook luidt.

  4. IS DE TREFKANS VERSPRONGEN? Een boek dat van 79% naar 99% gaat terwijl
     het aantal trades verviervoudigt, beschrijft twee verschillende systemen
     of een systeem dat is opgehouden met verliezers sluiten.

  5. IS DE KOP EEN RENDEMENT? "Gain" op myfxbook is tijdgewogen en wordt
     opgeblazen door stortingen en opnames. "Abs. Gain" is winst gedeeld door
     inleg. Staat daar een factor tien tussen, dan is de kop geen rendement
     dat iemand verdiend heeft.

WAT DEZE MODULE NIET DOET. Hij zegt niet of iemand oplicht. Hij zegt of de
gepubliceerde getallen en de gepubliceerde beweringen tegelijk waar kunnen
zijn. Dat is een smallere en veel hardere vraag.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Trackrecord:
    """De cijfers zoals ze op de statistiekenpagina staan."""

    naam: str
    trades: int
    winnaars: int
    gem_winst: float
    gem_verlies: float          # positief opgegeven; het is een verlies
    #: Wat de aanbieder zelf publiceert, om tegenaan te rekenen.
    opgegeven_pf: float | None = None
    opgegeven_verwachting: float | None = None
    stortingen: float | None = None
    opnames: float | None = None
    winst: float | None = None
    kop_gain: float | None = None
    abs_gain: float | None = None
    drawdown: float | None = None
    #: (aantal trades, aantal winnaars) van een deelperiode zonder verliezers.
    reeks: tuple[int, int] | None = None
    #: (trades, trefkans) van het laatste jaar, om drift te zien.
    dit_jaar: tuple[int, float] | None = None
    #: Wat de verkoopbrochure belooft.
    beweert_onafhankelijk: bool = False
    beweert_max_drawdown: float | None = None

    @property
    def verliezers(self) -> int:
        return self.trades - self.winnaars

    @property
    def trefkans(self) -> float:
        return self.winnaars / self.trades


def rekent_het_op(t: Trackrecord) -> dict[str, object]:
    """TOETS 1. Profit factor en verwachting uit de losse delen."""

    bruto_winst = t.winnaars * t.gem_winst
    bruto_verlies = t.verliezers * t.gem_verlies
    pf = bruto_winst / bruto_verlies if bruto_verlies else float("inf")
    verwachting = (bruto_winst - bruto_verlies) / t.trades
    uit: dict[str, object] = {
        "profit_factor": pf,
        "verwachting": verwachting,
        "totale_winst": bruto_winst - bruto_verlies,
    }
    if t.opgegeven_pf is not None:
        uit["pf_wijkt_af"] = abs(pf - t.opgegeven_pf) > 0.05
    if t.opgegeven_verwachting is not None:
        uit["verwachting_wijkt_af"] = (
            abs(verwachting - t.opgegeven_verwachting) > 0.02 * abs(t.opgegeven_verwachting)
        )
    return uit


def is_er_voorsprong(t: Trackrecord) -> dict[str, float]:
    """TOETS 2. Trefkans zegt niets zonder de verhouding ernaast.

    DIT IS WAAR IK ZELF DE MIST IN GING. Een gemiddeld verlies van vijf keer de
    gemiddelde winst ziet eruit als een grid, en dat is het niet per se: een
    WIJDE STOP met een KLEIN DOEL geeft exact dezelfde vorm en is volstrekt
    legitiem. Wat telt is de trefkans MIN het quitte-punt, niet de vorm.
    """

    verhouding = t.gem_verlies / t.gem_winst
    breakeven = verhouding / (1.0 + verhouding)
    return {
        "verhouding": verhouding,
        "breakeven_trefkans": breakeven,
        "voorsprong": t.trefkans - breakeven,
    }


def zijn_de_trades_onafhankelijk(t: Trackrecord) -> dict[str, object] | None:
    """TOETS 3, EN DEZE BESLIST.

    Wordt beweerd dat elke trade op zichzelf staat met een eigen stop, dan is
    een reeks van N trades zonder een enkele verliezer eenvoudig te toetsen:
    bij een trefkans p is die kans p^N.

    Er is geen strategie en geen marktinzicht voor nodig om dit te lezen. Komt
    er 1 op een miljard uit, dan zijn die trades niet onafhankelijk -- en dan
    is de bewering onjuist, ongeacht wat er verder klopt.
    """

    if t.reeks is None:
        return None
    n, winnaars = t.reeks
    if winnaars < n:
        return {"reeks": n, "verliezers_in_reeks": n - winnaars, "toetsbaar": False}
    kans = t.trefkans ** n
    return {
        "reeks": n,
        "verliezers_in_reeks": 0,
        "toetsbaar": True,
        "kans": kans,
        "een_op": (1.0 / kans) if kans > 0 else float("inf"),
        # Onder de 1 op een miljoen noemen we de bewering weerlegd. Dat is een
        # ruime grens: bij honderd van dit soort accounts verwacht je er nog
        # steeds geen enkele.
        "weerlegt_onafhankelijkheid": kans < 1e-6,
    }


def is_de_trefkans_versprongen(t: Trackrecord) -> dict[str, float] | None:
    """TOETS 4. Twee periodes, twee verschillende systemen?"""

    if t.dit_jaar is None:
        return None
    jaar_trades, jaar_trefkans = t.dit_jaar
    jaar_winnaars = round(jaar_trades * jaar_trefkans)
    oud_trades = t.trades - jaar_trades
    oud_winnaars = t.winnaars - jaar_winnaars
    if oud_trades <= 0:
        return None
    return {
        "nu_trades": jaar_trades,
        "nu_trefkans": jaar_winnaars / jaar_trades,
        "eerder_trades": oud_trades,
        "eerder_trefkans": oud_winnaars / oud_trades,
        "sprong": jaar_winnaars / jaar_trades - oud_winnaars / oud_trades,
    }


def is_de_kop_een_rendement(t: Trackrecord) -> dict[str, object] | None:
    """TOETS 5. Wat er op de inleg verdiend is, tegenover de kop."""

    if t.stortingen is None or t.winst is None:
        return None
    echt = t.winst / t.stortingen
    uit: dict[str, object] = {"rendement_op_inleg": echt}
    if t.kop_gain is not None:
        uit["kop"] = t.kop_gain
        uit["factor"] = t.kop_gain / echt if echt else float("inf")
    if t.abs_gain is not None:
        uit["abs_gain_klopt"] = abs(echt - t.abs_gain) < 0.01
    return uit


def audit(t: Trackrecord) -> str:
    """Alle vijf, in de volgorde waarin ze ertoe doen."""

    r = [f"\n  {t.naam}", "  " + "=" * 68]

    een = rekent_het_op(t)
    r.append(f"\n  1. REKENT HET UIT ZICHZELF OP?")
    r.append(f"     profit factor  {een['profit_factor']:.2f}"
             + (f"   opgegeven {t.opgegeven_pf:.2f}" if t.opgegeven_pf else ""))
    r.append(f"     verwachting    {een['verwachting']:+,.2f}"
             + (f"   opgegeven {t.opgegeven_verwachting:+,.2f}"
                if t.opgegeven_verwachting else ""))
    afwijking = een.get("pf_wijkt_af") or een.get("verwachting_wijkt_af")
    r.append("     -> " + ("WIJKT AF van wat er gepubliceerd staat"
                           if afwijking else "klopt met wat er gepubliceerd staat"))

    twee = is_er_voorsprong(t)
    r.append(f"\n  2. IS ER EEN VOORSPRONG?")
    r.append(f"     verlies is {twee['verhouding']:.2f}x de winst"
             f"  ->  quitte bij {twee['breakeven_trefkans']:.1%}")
    r.append(f"     trefkans {t.trefkans:.1%}  ->  voorsprong {twee['voorsprong']:+.1%}")
    r.append("     -> " + ("JA, en de vorm op zichzelf is niet verdacht: een wijde "
                           "stop\n        met een klein doel geeft precies dit"
                           if twee["voorsprong"] > 0 else "NEE"))

    drie = zijn_de_trades_onafhankelijk(t)
    r.append(f"\n  3. ZIJN DE TRADES ONAFHANKELIJK?"
             + ("   (dit wordt BEWEERD)" if t.beweert_onafhankelijk else ""))
    if drie is None:
        r.append("     geen foutloze reeks opgegeven -- niet te toetsen")
    elif not drie["toetsbaar"]:
        r.append(f"     de reeks bevat {drie['verliezers_in_reeks']} verliezers, "
                 "dus hier is niets tegenstrijdigs aan")
    else:
        r.append(f"     {drie['reeks']} trades op rij zonder EEN verliezer")
        r.append(f"     kans als ze onafhankelijk zijn: {drie['kans']:.2g}"
                 f"   (1 op {drie['een_op']:,.0f})")
        r.append("     -> " + ("WEERLEGD. Deze trades staan niet los van elkaar."
                               if drie["weerlegt_onafhankelijkheid"]
                               else "niet in tegenspraak"))

    vier = is_de_trefkans_versprongen(t)
    if vier:
        r.append(f"\n  4. IS DE TREFKANS VERSPRONGEN?")
        r.append(f"     eerder : {vier['eerder_trades']:>5,} trades  "
                 f"{vier['eerder_trefkans']:.1%}")
        r.append(f"     nu     : {vier['nu_trades']:>5,} trades  {vier['nu_trefkans']:.1%}")
        r.append(f"     -> sprong {vier['sprong']:+.1%}")

    vijf = is_de_kop_een_rendement(t)
    if vijf:
        r.append(f"\n  5. IS DE KOP EEN RENDEMENT?")
        r.append(f"     op de inleg verdiend: {vijf['rendement_op_inleg']:.1%}")
        if "kop" in vijf:
            r.append(f"     kop op de pagina    : {vijf['kop']:.1%}"
                     f"   ({vijf['factor']:.1f}x zo groot)")

    if t.beweert_max_drawdown is not None and t.drawdown is not None:
        r.append(f"\n  EN DE BELOFTE NAAST DE METER:")
        r.append(f"     belooft max {t.beweert_max_drawdown:.0%} drawdown, "
                 f"staat op {t.drawdown:.1%}")

    return "\n".join(r)


#: De twee accounts die de eigenaar voorgelegd heeft. Cijfers overgenomen van
#: hun eigen statistiekenpagina, niets aan gerekend.
ELIO = Trackrecord(
    naam="Elio's IA bot  --  FXGiants, MT4, 1:500",
    trades=2785, winnaars=1386 + 1260,
    gem_winst=1502.78, gem_verlies=7178.30,
    opgegeven_pf=3.99, opgegeven_verwachting=1069.51,
    stortingen=210_000.0, opnames=957_000.0, winst=2_978_582.06,
    kop_gain=21.8120, abs_gain=14.1837, drawdown=0.5501,
    reeks=(416, 416),                 # deze maand: 416 trades, 100% win
    dit_jaar=(2219, 0.99),
    beweert_onafhankelijk=True,       # "not a Martingale or grid system"
    beweert_max_drawdown=0.35,        # Aggressive-tier in de brochure
)

AURIX = Trackrecord(
    naam="Aurix-Pulse  --  Levels, MT5, 1:100",
    trades=3497, winnaars=963 + 1013,
    gem_winst=31.47, gem_verlies=24.71,
    opgegeven_pf=1.65, opgegeven_verwachting=7.04,
    stortingen=49_754.25, opnames=73_788.49, winst=24_602.36,
    kop_gain=4.9569, abs_gain=0.4943, drawdown=0.1099,
    dit_jaar=(1241, 0.51),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--welke", default="alle", choices=["alle", "elio", "aurix"])
    args = parser.parse_args()
    keuze = {"elio": [ELIO], "aurix": [AURIX], "alle": [ELIO, AURIX]}[args.welke]
    for t in keuze:
        print(audit(t))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
