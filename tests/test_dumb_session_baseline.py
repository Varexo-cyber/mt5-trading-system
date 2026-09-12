"""De domme sessietest moet dom zijn, en vooral: kloppen over middernacht."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.dumb_session_baseline import _as_r, _sessions, _stats, build_parser

ROOT = Path(__file__).resolve().parents[1]


def _frame(days: int = 10, step: str = "5min") -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=days * 288, freq=step, tz="UTC")
    price = np.linspace(2000.0, 2100.0, len(index))
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price,
            "volume": 100.0,
            "spread": 1.0,
        },
        index=index,
    )


def test_a_window_that_crosses_midnight_is_not_silently_empty() -> None:
    """20:00-02:00 is precies zo'n venster, en dat is geen randgeval hier.

    Een naive `start <= uur < end` levert voor 20 tot 2 een lege selectie op.
    Nul trades leest als "geen effect" en niet als een bug, en dat is exact het
    soort stilte waar dit project aan lijdt: het antwoord zou "sectie zes wint"
    zijn geweest omdat de tegenpartij nooit gehandeld had.
    """

    sessions = _sessions(_frame(days=10), 20, 2)

    assert not sessions.empty, "een venster over middernacht mag nooit leeg zijn"
    # Zes uur op M5 is 72 bars; de laatste dag is afgekapt door het einde van
    # de reeks en valt onder de halve-mediaan-drempel.
    assert sessions["bars"].median() == 72
    assert 8 <= len(sessions) <= 10


def test_the_hours_after_midnight_belong_to_the_evening_before() -> None:
    """Anders wordt een venster in tweeen gehakt en telt elke helft los mee.

    Koop om 20:00 op maandag en verkoop om 02:00 op dinsdag is EEN trade. Zou
    de datumstempel met de kalender meelopen, dan werd het er twee: een van
    20:00 tot middernacht en een van middernacht tot 02:00, allebei met hun
    eigen open en close. Dat verdubbelt het aantal trades en halveert de R per
    trade -- twee getallen die er allebei plausibel uitzien.
    """

    frame = _frame(days=6)
    sessions = _sessions(frame, 20, 2)

    # Elk venster beslaat zes uur, dus open en close liggen zes uur uit elkaar.
    # Op een oplopende reeks betekent dat elke trade strikt positief is.
    assert (sessions["close"] > sessions["open"]).all()
    # En de stempels zijn dagen, niet dagen-en-nachten apart.
    assert len(set(sessions.index)) == len(sessions)


def test_a_normal_window_still_works() -> None:
    """De niet-middernacht-tak mag niet sneuvelen aan de reparatie ernaast."""

    sessions = _sessions(_frame(days=10), 8, 12)

    assert not sessions.empty
    assert sessions["bars"].median() == 48  # vier uur op M5


def test_a_holiday_half_session_does_not_count_as_a_full_day() -> None:
    """Anders weegt een halve kerstsessie even zwaar als een volle dag."""

    frame = _frame(days=10)
    # Sloop het venster van een dag tot een handvol bars.
    day = pd.Timestamp("2026-01-05", tz="UTC")
    keep = ~(
        (frame.index >= day + pd.Timedelta(hours=20))
        & (frame.index < day + pd.Timedelta(hours=25, minutes=30))
    )
    thin = frame[keep | (frame.index < day + pd.Timedelta(hours=20, minutes=30))]

    sessions = _sessions(thin, 20, 2)

    assert (sessions["bars"] >= sessions["bars"].median() * 0.5).all()


def test_costs_are_charged_per_trade_and_move_the_verdict() -> None:
    """Zonder kostenaftrek meet dit iets wat niemand kan handelen.

    Dat is precies de fout die dit project pas gisteren op de eigen cijfers
    gevonden heeft, dus hij hoort hier niet opnieuw in te sluipen.
    """

    returns = pd.Series([0.10] * 20)

    gratis = _stats(returns, 0.0)
    duur = _stats(returns, 0.12)

    assert gratis["per_trade"] == pytest.approx(0.10)
    assert duur["per_trade"] == pytest.approx(-0.02)
    assert gratis["totaal"] > 0 > duur["totaal"]


def test_the_result_is_expressed_in_r_and_not_in_price() -> None:
    """Alles waar dit tegen afgezet wordt staat in R.

    Sectie zes gebruikt een stop van 0,8 x ATR, dus een venster dat een halve
    ATR oploopt is +0,625 R. Zonder die omrekening vergelijk je prijs met R en
    lijkt het antwoord wat je wil dat het is.
    """

    opened = pd.Timestamp("2026-01-02 20:00", tz="UTC")
    sessions = pd.DataFrame(
        {"open": [2000.0], "close": [2010.0], "bars": [72], "opened_at": [opened]},
        index=pd.DatetimeIndex(["2026-01-02"], tz="UTC"),
    )
    atr = pd.Series([25.0], index=pd.DatetimeIndex([opened]))

    # 10 punten winst, stop is 0,8 x 25 = 20 punten -> +0,5 R
    assert _as_r(sessions, 0.8, atr).iloc[0] == pytest.approx(0.5)


def test_the_atr_is_section_sixs_own_and_not_a_second_definition() -> None:
    """DIT IS EEN KEER FOUT GEGAAN EN HET FLATTEERDE SECTIE ZES.

    De eerste versie nam het gemiddelde van de M5 true range per dag maal 14 in
    plaats van een voortschrijdend gemiddelde over 14 bars. De noemer werd
    daarmee veertien keer te groot, elke R veertien keer te klein, en de domme
    sessietest kwam uit op +5,91 R terwijl sectie zes over hetzelfde venster
    +140 R deed. Dat las als "het model verdient zijn plek" en het was een
    rekenfout.

    Twee definities van dezelfde grootheid is de fout die dit project
    achtervolgt. Deze test eist dat het er letterlijk EEN is.
    """

    from analysis.section_six_adaptive import _atr as section_six_atr
    from scripts.dumb_session_baseline import _atr as baseline_atr

    frame = _frame(days=5)

    pd.testing.assert_series_equal(baseline_atr(frame), section_six_atr(frame))


def test_a_different_atr_period_is_refused_rather_than_quietly_used() -> None:
    """Een andere periode maakt de vergelijking zinloos zonder dat het opvalt."""

    from scripts.dumb_session_baseline import _atr as baseline_atr

    with pytest.raises(ValueError, match="ATR\\(14\\)"):
        baseline_atr(_frame(days=3), period=20)


def test_the_r_uses_the_volatility_at_the_moment_the_window_opens() -> None:
    """Niet die van middernacht, want 20:00-02:00 is per dag gestempeld.

    Op de dagstempel reindexen pakt de ATR van twintig uur voor de instap. Dat
    is geen vooruitkijken, maar het is wel de verkeerde volatiliteit, en op een
    dag waarop de markt 's avonds losbarst is het verschil groot.
    """

    opened = pd.Timestamp("2026-01-02 20:00", tz="UTC")
    sessions = pd.DataFrame(
        {"open": [2000.0], "close": [2010.0], "bars": [72], "opened_at": [opened]},
        index=pd.DatetimeIndex(["2026-01-02"], tz="UTC"),
    )
    atr = pd.Series(
        [100.0, 25.0],
        index=pd.DatetimeIndex(["2026-01-02 00:00", "2026-01-02 20:00"], tz="UTC"),
    )

    # Met de ATR van middernacht (100) zou dit +0,125 R zijn; met die van de
    # opening (25) is het +0,5 R.
    assert _as_r(sessions, 0.8, atr).iloc[0] == pytest.approx(0.5)


def test_the_sweep_is_off_by_default() -> None:
    """De tabel met 24 vensters is een diagnose en nodigt uit tot plukken.

    Standaard aan zou van deze nulhypothese een zoektocht maken, en dat is de
    fout die sectie vijf drie keer van teken deed wisselen.
    """

    assert build_parser().parse_args([]).sweep is False
    assert build_parser().parse_args(["--sweep"]).sweep is True


def test_the_default_window_is_the_one_section_six_actually_trades() -> None:
    """Een andere standaard zou een andere vraag beantwoorden."""

    args = build_parser().parse_args([])

    assert (args.start_hour, args.end_hour) == (20, 2)


def test_the_launcher_says_the_sweep_is_not_a_search() -> None:
    """Zonder die waarschuwing wordt de beste rij eruit geplukt. Gegarandeerd."""

    launcher = (ROOT / "domtest.cmd").read_text(encoding="utf-8")
    flat = " ".join(launcher.split())

    assert "scripts.dumb_session_baseline" in launcher
    assert "geen zoektocht" in flat
    assert "PLUK ER NIETS UIT" in flat


def test_it_finds_a_planted_effect_and_does_not_invent_one() -> None:
    """De enige test die zegt of dit gereedschap meet wat het beweert.

    Alle tests hierboven controleren onderdelen. Deze bouwt goud dat ALLEEN
    tussen 20:00 en 02:00 UTC oploopt en de rest van de dag terugzakt, en eist
    twee dingen tegelijk: het venster wordt gevonden, en de andere vensters
    worden NIET gevonden. Een meter die overal iets ziet is even nutteloos als
    een die nergens iets ziet, en alleen de tweede helft van deze test vangt
    het eerste geval.
    """

    from scripts.dumb_session_baseline import _atr

    index = pd.date_range("2025-01-01", periods=180 * 288, freq="5min", tz="UTC")
    rng = np.random.default_rng(7)
    step = rng.normal(0.0, 0.35, len(index))
    inside = (index.hour >= 20) | (index.hour < 2)
    step[inside] += 0.035
    step[~inside] -= 0.0117
    price = 2000.0 + np.cumsum(step)
    frame = pd.DataFrame(
        {
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 100.0,
            "spread": 1.0,
        },
        index=index,
    )
    atr = _atr(frame)

    def measured(start: int, end: int) -> dict:
        return _stats(_as_r(_sessions(frame, start, end), 0.8, atr), 0.05)

    planted = measured(20, 2)
    assert planted["t"] > 4.0, "het ingebouwde effect wordt niet gevonden"
    assert planted["totaal"] > 0.0

    for start, end in ((8, 12), (13, 19)):
        elsewhere = measured(start, end)
        assert elsewhere["t"] < 0.0, f"{start}-{end} zou geen positief effect mogen tonen"
