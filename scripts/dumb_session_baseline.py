"""Is sectie zes een model, of een klok met een model eromheen?

DE VRAAG. Sectie zes handelt XAUUSD tussen 20:00 en 02:00 UTC, alleen long, op
een bevroren nonlineair model met een drempel van 0,15. Dat venster overlapt de
Aziatische handelssessie, waarvan al dertig jaar beweerd wordt dat goud er
gemiddeld in oploopt.

Als die bewering klopt, dan kan het zijn dat sectie zes geen model is maar een
dure manier om een sessie-effect te handelen. Dat is te testen met de domst
mogelijke tegenpartij: koop goud aan het begin van het venster, verkoop aan het
eind, elke dag, geen model, geen drempel, geen filter, geen enkele parameter om
op te passen.

    de domme versie doet het NET ZO GOED  ->  het model is versiering, en je
                                              houdt iets over met vier
                                              parameters in plaats van veertig
    de domme versie doet het SLECHTER     ->  het model verdient zijn plek, en
                                              dat is voor het eerst aangetoond

Allebei die uitkomsten zijn winst, en dat is zeldzaam genoeg om dit als eerste
te doen.

WAAROM DIT GEEN KANDIDAAT IS. Dit voegt niets toe aan de best-of-N-straf uit
`_bonferroni_t`. Er wordt niets gekozen, niets afgesteld en niets geoptimaliseerd
-- er is precies één regel, hij stond van tevoren vast, en hij heeft geen knop.
Het is een NULHYPOTHESE waar sectie zes overheen moet, geen concurrent die
gepromoveerd kan worden.

DE UUR-SWEEP IS ER OM DE VERKEERDE REDEN NIET. `--sweep` toont wél alle
venstercombinaties, en dat is expres een DIAGNOSE en geen zoektocht: als
20:00-02:00 er willekeurig tussen ligt is het effect breed en waarschijnlijk
echt, en als het een eenzame piek is dan is het venster zelf al een keuze uit
vierentwintig en telt sectie zes zijn eigen multipliciteit niet mee. Het beste
vakje uit die tabel plukken is exact de fout die dit project al drie keer
gemaakt heeft.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from backtesting.replay import fetch_mt5_history
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from core.types import Timeframe


def _sessions(frame: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
    """Rendement per venster: koop op de eerste bar, verkoop op de laatste.

    HET VENSTER MAG OVER MIDDERNACHT HEEN, en dat is niet cosmetisch: 20:00-02:00
    is precies zo'n venster, en een naïeve `start <= uur < end` levert daar een
    lege selectie op -- nul trades, wat leest als "geen effect" in plaats van
    als een bug. Elk venster krijgt daarom een eigen datumstempel die met de
    STARTdag meebeweegt, zodat de uren na middernacht bij de avond ervoor horen.
    """

    hours = frame.index.hour
    if start_hour == end_hour:
        # HET HELE ETMAAL, en dat is geen randgeval maar de controle: de hele
        # dag long, om te zien of dit venster meer doet dan zijn aandeel in de
        # uren. `start <= end` zou hier `uur >= 0 & uur < 0` opleveren, dus nul
        # rijen -- en een lege controle leest als "geen trend in goud", precies
        # de conclusie die de test moet kunnen weerleggen.
        inside = np.ones(len(frame), dtype=bool)
        day = frame.index.normalize()
    elif start_hour < end_hour:
        inside = (hours >= start_hour) & (hours < end_hour)
        day = frame.index.normalize()
    else:
        inside = (hours >= start_hour) | (hours < end_hour)
        # Alles vóór het einduur hoort bij de vorige kalenderdag.
        day = (frame.index - pd.Timedelta(hours=end_hour)).normalize()

    picked = frame[inside]
    if picked.empty:
        return pd.DataFrame(columns=["open", "close", "bars"])
    grouped = picked.groupby(day[inside])
    out = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "close": grouped["close"].last(),
            "bars": grouped["close"].size(),
            # HET MOMENT WAAROP HET VENSTER OPENT, want de R hoort bij de
            # volatiliteit van dat moment. Op de dagstempel afgaan zou voor een
            # venster van 20:00 tot 02:00 de ATR van twintig uur eerder pakken.
            "opened_at": pd.Series(picked.index, index=picked.index).groupby(day[inside]).first(),
        }
    )
    # EEN VENSTER MET EEN HANDVOL BARS IS EEN FEESTDAG, geen handelsdag. Zonder
    # deze regel telt een halve kerstsessie even zwaar mee als een volle dag.
    expected = out["bars"].median()
    return out[out["bars"] >= expected * 0.5]


def _drawdown(values: pd.Series) -> float:
    """Diepste terugval in R, en dat getal ontbrak in de eerste versie.

    ZONDER DIT IS DE VERGELIJKING ONEERLIJK IN HET VOORDEEL VAN DE DOMME
    VERSIE. Sectie zes heeft een stop van 0,8 ATR; de klok heeft er geen en zit
    zes uur lang vol in de markt. Een nacht waarin goud vijftig dollar zakt kost
    de klok dertig R en sectie zes één R. Twee totalen naast elkaar zetten
    zonder dat erbij te vertellen is precies hoe een strategie die de rekening
    opblaast er beter uitziet dan een die dat niet doet.
    """

    if values.empty:
        return 0.0
    equity = values.cumsum()
    return float((equity.cummax() - equity).max())


def _stats(returns: pd.Series, cost_per_trade_r: float) -> dict:
    """Totaal, per trade, trefkans, t, terugval en de ergste losse dag."""

    if returns.empty:
        return {}
    net = returns - cost_per_trade_r
    n = len(net)
    mean = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "totaal": float(net.sum()),
        "per_trade": mean,
        "raak": float((net > 0).mean()),
        "t": (mean / (sd / np.sqrt(n))) if sd > 0 else 0.0,
        "bruto_totaal": float(returns.sum()),
        "bruto_per_trade": float(returns.mean()),
        "terugval": _drawdown(net),
        "ergste_dag": float(net.min()),
    }


def _as_r(sessions: pd.DataFrame, stop_atr: float, atr: pd.Series) -> pd.Series:
    """Vensterrendement uitgedrukt in R, met dezelfde stopbreedte als sectie zes.

    IN R EN NIET IN PROCENT, want alles waar dit tegen afgezet wordt staat in R.
    Sectie zes gebruikt een stop van 0,8 x ATR, dus een venster dat een halve ATR
    oploopt is +0,625 R. Zonder die omrekening vergelijk je twee eenheden en
    lijkt het antwoord wat je wil dat het is.
    """

    # DE ATR OP DE OPENINGSBAR, en dat is geen detail: `atr` loopt op M5 en
    # `sessions` is per dag gestempeld. Op de dagstempel reindexen zou voor een
    # venster van 20:00 tot 02:00 de volatiliteit van twintig uur eerder pakken.
    at_open = atr.reindex(pd.DatetimeIndex(sessions["opened_at"]), method="ffill")
    risk = pd.Series(at_open.to_numpy() * stop_atr, index=sessions.index).replace(0.0, np.nan)
    return ((sessions["close"] - sessions["open"]) / risk).dropna()


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Exact `analysis.section_six_adaptive._atr`, en dat is hier het hele punt.

    DIT IS EEN KEER FOUT GEGAAN EN HET FLATTEERDE SECTIE ZES. De eerste versie
    nam het GEMIDDELDE van de M5 true range per dag en vermenigvuldigde dat met
    14, in plaats van er een voortschrijdend gemiddelde over 14 bars van te
    nemen. Dat maakt de noemer veertien keer te groot, dus elke R veertien keer
    te klein, en de domme sessietest kwam uit op +5,91 R bruto terwijl sectie
    zes over hetzelfde venster +140 R deed. Dat las als "het model verdient zijn
    plek" en dat was rekenfout, geen bewijs.

    Twee definities van dezelfde grootheid is de fout die dit hele project
    achtervolgt, dus deze wordt niet nagebouwd maar GEIMPORTEERD. Wijzigt sectie
    zes zijn ATR, dan wijzigt die van de nulhypothese mee -- automatisch, en
    zonder dat iemand eraan hoeft te denken.
    """

    from analysis.section_six_adaptive import _atr as section_six_atr

    if period != 14:
        raise ValueError(
            "sectie zes rekent ATR(14) en de nulhypothese moet exact dezelfde "
            "eenheid gebruiken; een andere periode maakt de vergelijking zinloos"
        )
    return section_six_atr(frame)


def run(args) -> None:
    settings = load_settings(overlay=args.config, env_overrides=False)
    symbol = settings.instruments.broker_symbol("XAUUSD")
    end = datetime.now(UTC) if args.end_date is None else args.end_date
    start = args.start_date or (end - timedelta(days=args.days))

    credentials = load_credentials(required=True)
    connector = MT5Connector(
        settings.mt5,
        credentials,
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    connector.connect()
    try:
        frame = fetch_mt5_history(connector, symbol, Timeframe.M5, start - timedelta(days=5), end)
    finally:
        connector.shutdown()

    if frame.empty:
        print(f"Geen M5-historie voor {symbol} in dit venster.")
        return
    frame = frame[(frame.index >= start) & (frame.index <= end)]
    atr = _atr(frame)

    cfg = settings.analysis.section_six_gold_m5
    stop_atr = getattr(cfg, "stop_atr_multiple", 0.8)

    print(f"\n{'=' * 78}")
    print("DE DOMME SESSIETEST — koop bij de opening, verkoop bij het slot")
    print(f"{'=' * 78}")
    print(f"  {symbol}  M5  {frame.index[0]:%d-%m-%Y} t/m {frame.index[-1]:%d-%m-%Y}")
    print(f"  R = {stop_atr} x ATR, dezelfde stopbreedte als sectie zes.")
    print(f"  Kosten: {args.cost_r:.3f} R per trade, afgetrokken van elk venster.")
    print("\n  GEEN model, GEEN drempel, GEEN filter, GEEN parameter om te kiezen.")
    print("  Dit is de lat waar sectie zes overheen hoort te komen, niet een")
    print("  strategie die live mag.")

    main = _stats(
        _as_r(_sessions(frame, args.start_hour, args.end_hour), stop_atr, atr), args.cost_r
    )
    if not main:
        print("\n  Geen enkel volledig venster in deze periode.")
        return
    print(
        f"\n  HET VENSTER VAN SECTIE ZES — {args.start_hour:02d}:00 tot {args.end_hour:02d}:00 UTC"
    )
    print(
        f"    {main['n']} dagen   {main['raak']:.1%} raak   "
        f"bruto {main['bruto_totaal']:+.2f} R ({main['bruto_per_trade']:+.4f}/dag)   "
        f"netto {main['totaal']:+.2f} R ({main['per_trade']:+.4f}/dag)   t={main['t']:.2f}"
    )
    # NIET-SIGNIFICANT IS NIET HETZELFDE ALS NUL, en de eerste versie van deze
    # regel gooide die twee op een hoop: bij t=0,82 en +91,46 R stond er "het
    # venster alleen doet niets". Dat is precies verkeerd. De vergelijking met
    # sectie zes gaat over het TOTAAL, en dat totaal was bijna gelijk aan wat
    # sectie zes met veertig parameters ophaalt.
    #
    # DE LAT IS 1,96 EN NIET HOGER, want dit is EEN vooraf vastgelegde regel en
    # geen keuze uit vele. Zou dit uit de sweep hieronder geplukt worden, dan
    # gold de Bonferroni-lat voor vierentwintig vensters en die is 3,03.
    if abs(main["t"]) >= 1.96:
        richting = "POSITIEF" if main["per_trade"] > 0 else "NEGATIEF"
        print(f"    -> op zichzelf {richting}: t haalt de 1,96.")
    else:
        print(f"    -> de spreiding is te groot voor significantie (t={main['t']:.2f}); dit")
        print("       venster staat op zichzelf niet vast. Dat is IETS ANDERS dan nul:")
        print(f"       het totaal is {main['totaal']:+.2f} R en daarmee vergelijk je hieronder.")

    print(f"    terugval {main['terugval']:.2f} R   ergste dag {main['ergste_dag']:+.2f} R")

    # DE CONTROLE DIE ONTBRAK, en zonder haar is het hele cijfer waardeloos.
    #
    # Goud steeg in 2024-2025 hard. Long zitten in WELK venster dan ook verdient
    # dan geld, en dan meet dit niet een sessie-effect maar een bullmarkt. De
    # eerlijke vraag is of dit venster MEER oplevert dan zijn aandeel in de
    # uren: zes van de vierentwintig is een kwart, dus een venster zonder eigen
    # effect hoort ongeveer een kwart van de hele beweging te pakken.
    hours = (args.end_hour - args.start_hour) % 24 or 24
    hold = _stats(_as_r(_sessions(frame, 0, 0), stop_atr, atr), args.cost_r)
    print("\n  DE CONTROLE: DE HELE DAG LONG, over precies dezelfde dagen")
    if hold:
        share = hours / 24.0
        expected = hold["totaal"] * share
        print(
            f"    24 uur long   {hold['n']} dagen   netto {hold['totaal']:+.2f} R   "
            f"terugval {hold['terugval']:.2f} R"
        )
        print(f"    {hours} van de 24 uur is {share:.0%}, dus een venster zonder eigen effect")
        print(f"    hoort rond {expected:+.2f} R uit te komen.")
        # DE VERHOUDING ZELF, want daar leest de lezer hem aan af. De eerste
        # versie noemde alleen een oordeel met banden van 0,5 tot 1,5, en 0,61
        # kwam daardoor naar buiten als "ongeveer zijn aandeel" terwijl het
        # venster bijna veertig procent ONDER zijn uren presteerde. Een band is
        # altijd een keuze; het getal is dat niet.
        ratio = main["totaal"] / expected if expected else float("nan")
        print(f"    Dit venster doet {main['totaal']:+.2f} R = {ratio:.0%} van dat aandeel.")
        if ratio > 1.5:
            print("    -> MEER dan zijn aandeel. Er zit iets in dit venster zelf.")
        elif ratio < 0.8:
            print("    -> MINDER dan zijn aandeel. Deze uren zijn juist het ZWAKKE deel")
            print("       van de dag; er is hier geen sessie-effect om te oogsten.")
        else:
            print("    -> ONGEVEER zijn aandeel. Dan meet je de trend in het onderliggende")
            print("       en geen sessie-effect, en dan zegt dit getal niets over de klok.")
    else:
        print("    (niet te berekenen op deze reeks)")

    print("\n  HOE JE DIT LEEST, en dit is het hele punt:")
    print("    Zet `netto totaal` naast wat sectie zes over dezelfde periode deed")
    print("    (uit kosten.cmd). Doet dit domme ding het net zo goed, dan is het")
    print("    model versiering. Doet het het slechter, dan verdient het model")
    print("    zijn plek -- en dat is dan voor het eerst aangetoond.")
    print("\n    MAAR LEES DE TERUGVAL ERBIJ. De klok heeft GEEN stop en zit zes uur")
    print("    vol in de markt; sectie zes wordt op 0,8 ATR uitgestopt. Een groter")
    print("    totaal met een veel diepere terugval is geen betere strategie, dat")
    print("    is dezelfde strategie met meer hefboom.")

    if not args.sweep:
        print("\n  `--sweep` toont alle vensters. Lees dat als diagnose, niet als")
        print("  zoektocht: het beste vakje eruit plukken is de fout die sectie")
        print("  vijf drie keer van teken deed wisselen.\n")
        return

    print(f"\n{'=' * 78}")
    print("ALLE VENSTERS — DIAGNOSE, GEEN ZOEKTOCHT")
    print(f"{'=' * 78}")
    print("  Ligt 20:00-02:00 ergens tussen even goede buren, dan is het effect")
    print("  breed en waarschijnlijk echt. Is het een eenzame piek, dan is het")
    print("  venster zelf al een keuze uit 24 en telt sectie zes een")
    print("  multipliciteit mee die nooit verrekend is.")
    print("\n  PLUK HIER NIETS UIT. Bij 24 vensters is de eerlijke lat t>3,03,")
    print("  niet t>1,96.")
    print(f"\n  {'venster':<16}{'dagen':>7}{'raak':>8}{'netto R':>10}{'per dag':>10}{'t':>8}")
    rows = []
    for begin in range(24):
        for length in (4, 6, 8):
            finish = (begin + length) % 24
            stats = _stats(_as_r(_sessions(frame, begin, finish), stop_atr, atr), args.cost_r)
            if stats and stats["n"] >= 30:
                rows.append((f"{begin:02d}:00-{finish:02d}:00", stats))
    for label, stats in sorted(rows, key=lambda item: item[1]["t"], reverse=True):
        mark = (
            "  <-- sectie zes"
            if label == f"{args.start_hour:02d}:00-{args.end_hour:02d}:00"
            else ""
        )
        print(
            f"  {label:<16}{stats['n']:>7}{stats['raak']:>8.1%}"
            f"{stats['totaal']:>+10.2f}{stats['per_trade']:>+10.4f}{stats['t']:>8.2f}{mark}"
        )
    print(f"\n  {len(rows)} vensters. Bonferroni-lat voor deze tabel: t>3.03.\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=360)
    parser.add_argument("--start-date", type=datetime.fromisoformat, default=None)
    parser.add_argument("--end-date", type=datetime.fromisoformat, default=None)
    parser.add_argument("--start-hour", type=int, default=20, help="sectie zes: 20")
    parser.add_argument("--end-hour", type=int, default=2, help="sectie zes: 2")
    parser.add_argument(
        "--cost-r",
        type=float,
        default=0.05,
        help="kosten per trade in R; kosten.cmd noemt 0,048 tot 0,066 voor sectie zes",
    )
    parser.add_argument("--sweep", action="store_true", help="alle vensters, als diagnose")
    parser.add_argument("--config", default="config/eightcap.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in ("start_date", "end_date"):
        value = getattr(args, name)
        if value is not None and value.tzinfo is None:
            setattr(args, name, value.replace(tzinfo=UTC))
    run(args)


if __name__ == "__main__":
    main()
