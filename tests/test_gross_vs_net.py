"""De brutokolom moet de netto-kolom van de replay precies terugdraaien."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.gross_vs_net import (
    _refused_but_resolved,
    _rows,
    _span,
    _taken,
    _tally,
    combined,
    report,
)

ROOT = Path(__file__).resolve().parents[1]

HEADER = (
    "when,symbol,module,outcome,clock,direction,result_r_fixed_stop,"
    "pnl_money_fixed_stop,managed_r_LIVE,managed_money_LIVE,cost_r_charged"
)


def _csv(tmp_path: Path, *lines: str, name: str = "replay.csv", header: str = HEADER) -> Path:
    path = tmp_path / name
    path.write_text("\n".join((header, *lines)) + "\n", encoding="utf-8")
    return path


def test_gross_is_net_plus_cost_and_the_beheerde_kolom_wins(tmp_path, capsys) -> None:
    """`managed_r_LIVE` is wat de rekening draait; de vaste kolom vult alleen aan.

    Dezelfde volgorde als `month_attribution`, en met opzet: twee scripts die
    dezelfde CSV lezen en een ander totaal noemen is erger dan geen van beide.
    """

    path = _csv(
        tmp_path,
        # beheerd +1,20 met 0,10 kosten -> bruto +1,30
        "2025-01-02T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,1.5,7,1.2,5.6,0.10",
        # geen beheerde kolom: de vaste route telt, -1,00 met 0,20 -> bruto -0,80
        "2025-01-03T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,-1.0,-5,,,0.20",
    )

    report(path)
    out = capsys.readouterr().out

    # netto 1,20 - 1,00 = +0,20 ; kosten 0,30 ; bruto +0,50
    assert "+0.50" in out
    assert "+0.20" in out
    assert "0.30" in out


def test_a_refused_setup_is_not_a_trade(tmp_path) -> None:
    """Alleen `outcome == TRADE` telt in de bruto-tegen-netto tabel.

    Een geweigerde setup is niet genomen en heeft dus geen kosten betaald. Zou
    hij meetellen, dan zou de kostenpoort de tabel verbeteren naarmate hij meer
    weigert -- precies omgekeerd aan wat er gemeten wordt. De weigeringen
    krijgen hun eigen tabel eronder.
    """

    path = _csv(
        tmp_path,
        "2025-01-02T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,1.0,5,,,0.10",
        "2025-01-03T10:00:00+00:00,XAUUSD,s6,SPREAD_EATS_THE_STOP,M5,LONG,,,,,0.0",
        "2025-01-04T10:00:00+00:00,XAUUSD,s6,UNDERCAPITALIZED,M5,LONG,,,,,0.0",
    )

    assert len(_rows(path)) == 3
    assert len(_taken(_rows(path))) == 1


def test_the_verdict_separates_a_dead_entry_from_an_expensive_one(tmp_path, capsys) -> None:
    """De hele reden dat dit script bestaat, en het moet niet te raden zijn.

    Bruto positief en netto negatief betekent "de instap ziet iets dat kleiner
    is dan de prijs". Bruto negatief betekent "de instap ziet niets". Die twee
    vragen om tegenovergestelde besluiten, dus het rapport moet ze met
    verschillende woorden uit elkaar houden en niet met twee getallen die de
    lezer zelf moet vergelijken.
    """

    duur = _csv(
        tmp_path,
        "2025-01-02T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,0.05,1,,,0.30",
        "2025-01-03T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,-0.20,-1,,,0.30",
        name="duur.csv",
    )

    # WITRUIMTE PLATGESLAGEN, want de tekst is op regels afgebroken en een
    # test die op de afbreking let gaat rood zodra iemand een woord toevoegt.
    # Wat vaststaat is het oordeel, niet waar de regel eindigt.
    def said(output: str) -> str:
        return " ".join(output.split())

    report(duur)
    assert "kleiner dan wat een rondje handelen kost" in said(capsys.readouterr().out)

    dood = _csv(
        tmp_path,
        "2025-01-02T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,-1.0,-5,,,0.05",
        "2025-01-03T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,-1.0,-5,,,0.05",
        name="dood.csv",
    )
    report(dood)
    assert "ziet geen richting, ook niet gratis" in said(capsys.readouterr().out)


def test_an_old_csv_without_the_cost_column_says_so(tmp_path) -> None:
    """Stil nul aannemen zou elke oude replay als gratis handelen rapporteren."""

    path = tmp_path / "oud.csv"
    path.write_text(
        "when,symbol,module,outcome,result_r_fixed_stop\n"
        "2025-01-02T10:00:00+00:00,XAUUSD,s6,TRADE,1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cost_r_charged"):
        _rows(path)


def test_a_windows_encoded_csv_still_loads(tmp_path) -> None:
    """Dezelfde CP-1252-tolerantie als `month_attribution`.

    Een export van de VPS draagt Nederlandse leestekens in de notitiekolom.
    Daarop struikelen zou een replay van uren weggooien om een aanhalingsteken.
    """

    path = tmp_path / "cp1252.csv"
    # Via chr() en niet als letterteken in de bron: ruff weigert dit soort
    # leestekens in broncode (RUF001), terwijl het juist het karakter is dat
    # deze test moet uitlokken.
    apostrophe = chr(0x2019)
    line = f"2025-01-02T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,1.0,5,,,0.10,druk{apostrophe}s"
    path.write_bytes(f"{HEADER},note\n{line}\n".encode("cp1252"))

    assert len(_rows(path)) == 1


def test_the_launcher_reads_csvs_and_never_starts_a_replay() -> None:
    """`kosten.cmd` mag geen uren kosten en geen markt aanraken.

    Het hele nut is dat het antwoord al op schijf staat. Zodra hier een
    replayvlag in sluipt is het geen snelle lezing meer en gaat niemand hem
    draaien.
    """

    launcher = (ROOT / "kosten.cmd").read_text(encoding="utf-8")

    assert "scripts.gross_vs_net" in launcher
    for replay_flag in ("--jarvis-replay", "--days", "--exit-grid", "dry_run_sections"):
        assert replay_flag not in launcher, replay_flag


def test_a_refused_setup_with_a_resolved_outcome_lands_in_the_gate_report(tmp_path, capsys) -> None:
    """De replay loopt geweigerde setups tóch uit, en dat antwoord telt.

    `dry_run_sections` schrijft voor elke weigering een volledig resultaat weg
    met de notitie "COUNTERFACTUAL if gate were off". Dat is de vraag "wat had
    die poort gekost", en die staat dus al in elk bestand dat er ligt. Zonder
    dit blok zou het rapport melden hoeveel setups een poort at en niets zeggen
    over of dat hielp -- de duurdere helft van de vraag.
    """

    path = _csv(
        tmp_path,
        "2025-01-02T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,1.0,5,,,0.10,",
        "2025-01-03T10:00:00+00:00,XAUUSD,s6,SPREAD_EATS_THE_STOP,M5,LONG,2.0,10,,,0.30,"
        "COUNTERFACTUAL if gate were off: spread is 30%",
        header=HEADER + ",note",
    )

    report(path)
    out = " ".join(capsys.readouterr().out.split())

    assert "SPREAD_EATS_THE_STOP" in out
    # nu +1,00 ; de poort weigerde +2,00 ; poorten uit zou +3,00 zijn
    assert "+1.00 +3.00 +2.00" in out


def test_a_refusal_without_an_outcome_is_not_counted_as_a_free_gate(tmp_path, capsys) -> None:
    """Niet elke weigering is uitgelopen, en nul aannemen liegt de goede kant op.

    Een setup die op UNDERCAPITALIZED strandde heeft geen resultaat. Zou die als
    0,00 R meetellen, dan lijkt elke poort gratis naarmate hij meer van dat soort
    weigeringen produceert -- precies omgekeerd aan wat er gevraagd wordt.
    """

    path = _csv(
        tmp_path,
        "2025-01-02T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,1.0,5,,,0.10,",
        "2025-01-03T10:00:00+00:00,XAUUSD,s6,UNDERCAPITALIZED,M5,LONG,,,,,0.0,",
        header=HEADER + ",note",
    )

    assert _refused_but_resolved(_rows(path)) == []
    report(path)
    assert "UNDERCAPITALIZED" not in capsys.readouterr().out


def test_the_gate_report_refuses_to_be_read_as_a_replay(tmp_path, capsys) -> None:
    """Optellen is geen herspelen, en dat moet er met zoveel woorden staan.

    Met de poorten echt uit hadden die setups slots bezet en andere trades
    verdrongen. Wie de kolom `poorten uit` als bedrag leest, leest een getal dat
    niemand had kunnen verdienen.
    """

    path = _csv(
        tmp_path,
        "2025-01-02T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,1.0,5,,,0.10,",
        "2025-01-03T10:00:00+00:00,XAUUSD,s6,SPREAD_EATS_THE_STOP,M5,LONG,2.0,10,,,0.30,"
        "COUNTERFACTUAL if gate were off: spread is 30%",
        header=HEADER + ",note",
    )

    report(path)
    out = " ".join(capsys.readouterr().out.split())

    # Kleingemaakt, want het kopje staat in kapitalen en of dat zo blijft is
    # opmaak. Wat vaststaat is dat het er staat.
    assert "geen replay met de poorten uit" in out.lower()
    assert "richting, niet als een bedrag" in out


def test_two_files_that_do_not_overlap_get_one_total(tmp_path, capsys) -> None:
    """De holdout plus de 360-daagse run is de vraag "2024 tot 2026"."""

    a = _csv(
        tmp_path,
        "2024-09-05T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,-1.0,-5,,,0.10,",
        name="holdout.csv",
        header=HEADER + ",note",
    )
    b = _csv(
        tmp_path,
        "2025-09-15T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,2.0,9,,,0.10,",
        name="goud360.csv",
        header=HEADER + ",note",
    )

    combined(
        [
            (a, _tally(_taken(_rows(a))), _span(_rows(a))),
            (b, _tally(_taken(_rows(b))), _span(_rows(b))),
        ]
    )
    out = " ".join(capsys.readouterr().out.split())

    assert "ALLES SAMEN — 05-09-2024 t/m 15-09-2025" in out
    # netto -1,00 + 2,00 = +1,00 ; kosten 0,20 ; bruto +1,20
    assert "2 +1.20 0.20 +1.00" in out


def test_overlapping_files_are_refused_instead_of_double_counted(tmp_path, capsys) -> None:
    """`september-2025-goud.csv` valt middenin `hoeveel-goud-360.csv`.

    Die twee optellen telt dezelfde trades twee keer en levert een groter getal
    op dat nergens op slaat. Stil optellen is precies de fout waar dit project
    aan lijdt, dus er komt geen totaal en er staat bij waarom.
    """

    a = _csv(
        tmp_path,
        "2025-09-01T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,1.0,5,,,0.10,",
        "2026-08-01T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,1.0,5,,,0.10,",
        name="goud360.csv",
        header=HEADER + ",note",
    )
    b = _csv(
        tmp_path,
        "2025-09-20T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,1.0,5,,,0.10,",
        name="september.csv",
        header=HEADER + ",note",
    )

    combined(
        [
            (a, _tally(_taken(_rows(a))), _span(_rows(a))),
            (b, _tally(_taken(_rows(b))), _span(_rows(b))),
        ]
    )
    out = " ".join(capsys.readouterr().out.split())

    assert "GEEN TOTAAL" in out
    assert "goud360.csv en september.csv" in out
    assert "ALLES SAMEN" not in out


def test_the_period_of_each_file_is_printed(tmp_path, capsys) -> None:
    """Zonder de periode is niet te zien WAT er samengeteld wordt.

    Twee bestanden met dezelfde naamgeving kunnen compleet andere vensters
    beslaan, en dan is een totaal een getal over een onbekende periode.
    """

    path = _csv(
        tmp_path,
        "2024-09-01T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,1.0,5,,,0.10,",
        "2025-08-31T10:00:00+00:00,XAUUSD,s6,TRADE,M5,LONG,1.0,5,,,0.10,",
        header=HEADER + ",note",
    )

    report(path)
    out = " ".join(capsys.readouterr().out.split())

    assert "01-09-2024 t/m 31-08-2025" in out


def test_the_launcher_defaults_to_the_two_files_that_span_2024_to_2026() -> None:
    """Zonder argument moet hij de holdout EN de 360-daagse run in EEN aanroep doen.

    Apart aanroepen geeft twee losse tabellen en geen som, en de som over beide
    vensters was de vraag. Een `for`-lus over runtime\\ kan dat per definitie
    niet: die start het script een keer per bestand.
    """

    launcher = (ROOT / "kosten.cmd").read_text(encoding="utf-8")

    assert "jarvis-holdout-2024-2025.csv" in launcher
    assert "hoeveel-goud-360.csv" in launcher
    # Een enkele aanroep met beide paden achter elkaar, niet twee aanroepen.
    assert "gross_vs_net %SAMEN%" in launcher
