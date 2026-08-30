"""Judging a run from the file it already wrote.

`_live_config_report` returns early when the section measured is not on
`live_enabled_modules`, so a SHADOW run prints its trades and none of the
judgement -- no sigma, no monthly table, no concentration check, no tick
boxes. The 180-day impulse_retest run came back with +20.60 R and EUR +126.73
and no way to tell whether either number meant anything.

Asking the owner to re-run twenty minutes to get arithmetic that could be done
on the CSV is not a reasonable thing to ask, and it is the third time this
session that a defect of mine has cost him a run.

THE RISK HERE IS A SECOND COPY OF THE BARS. If this file restated the
thresholds instead of importing them, the two would drift and the CSV verdict
would stop matching the live one. So it imports `_is_this_real` and these
tests check that it does.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.verdict_from_csv import rows_from

HEADER = [
    "when",
    "symbol",
    "module",
    "outcome",
    "direction",
    "entry",
    "stop",
    "target",
    "lots",
    "risk_money",
    "risk_pct",
    "result_r_fixed_stop",
    "pnl_money_fixed_stop",
    "managed_r_LIVE",
    "managed_money_LIVE",
    "note",
]


def _write(path: Path, rows: list[list], header: list[str] | None = None) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header or HEADER)
        writer.writerows(rows)
    return path


def _trade(when: datetime, fixed: float, managed: float | str = "") -> list:
    return [
        when.isoformat(),
        "US30",
        "impulse_retest",
        "TRADE",
        "LONG",
        1,
        1,
        1,
        0.1,
        7.24,
        3.36,
        fixed,
        fixed * 7.24,
        managed,
        (managed * 7.24) if managed != "" else "",
        "",
    ]


class TestItReadsWhatTheRunWrote:
    def test_it_takes_only_the_trade_rows(self, tmp_path: Path) -> None:
        """A CSV is mostly refusals -- 189,535 decisions against 358 trades on
        the run this was written for."""
        base = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
        refusal = _trade(base, 0.0)
        refusal[3] = "REFUSED_CONFLUENCE"
        path = _write(tmp_path / "x.csv", [_trade(base, 1.0, 0.1), refusal])

        trades, _managed = rows_from(path)

        assert len(trades) == 1
        assert trades[0].result_r == pytest.approx(1.0)

    def test_it_notices_the_managed_column(self, tmp_path: Path) -> None:
        base = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
        path = _write(tmp_path / "x.csv", [_trade(base, -1.0, 0.1)])

        trades, managed = rows_from(path)

        assert managed is True
        assert trades[0].managed_r == pytest.approx(0.1)

    def test_an_older_file_without_it_is_read_on_the_fixed_stop(self, tmp_path: Path) -> None:
        """And must not silently be judged as if it were the live exit."""
        base = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
        path = _write(tmp_path / "x.csv", [_trade(base, -1.0, "")])

        trades, managed = rows_from(path)

        assert managed is False
        assert trades[0].managed_r is None

    def test_a_blank_number_becomes_none_rather_than_zero(self, tmp_path: Path) -> None:
        """An unresolved trade scored as 0.0 would dilute every real outcome
        toward zero, which is the same mistake the resolver refuses to make."""
        base = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
        row = _trade(base, 1.0, 0.1)
        row[11] = ""
        path = _write(tmp_path / "x.csv", [row])

        trades, _m = rows_from(path)

        assert trades[0].result_r is None


class TestTheBarsAreImportedNotRestated:
    def test_it_uses_the_reports_own_verdict_function(self) -> None:
        """A second copy of a threshold is how the CSV verdict and the live
        one would eventually disagree."""
        script = Path("scripts/verdict_from_csv.py")
        source = script.read_text()

        assert "from scripts.dry_run_sections import Decision, _is_this_real" in source
        assert "sigma" not in source.split('"""')[2], "the bars are being restated here"
        assert script.exists()

    def test_it_produces_the_same_verdict_as_a_live_run_would(self, tmp_path: Path, capsys) -> None:
        """The property that matters: same trades, same answer, whichever path
        they arrive by."""
        from scripts.dry_run_sections import _is_this_real
        from scripts.verdict_from_csv import rows_from as read

        base = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
        rows = [
            _trade(base + timedelta(days=i % 100, minutes=i), 1.0 if i % 3 else -1.0, 0.5)
            for i in range(400)
        ]
        path = _write(tmp_path / "x.csv", rows)

        trades, managed = read(path)
        _is_this_real(trades, [("impulse_retest", "")], managed)
        from_csv = capsys.readouterr().out

        assert "sigma from zero" in from_csv
        assert "VERDICT" in from_csv
        assert "no single month carrying more than half the result" in from_csv


class TestTheLauncher:
    def test_it_names_a_file_and_survives_cmd(self) -> None:
        launcher = Path("verdict.cmd").read_text()

        for line in launcher.splitlines():
            if line.strip().startswith("set "):
                assert "," not in line, line
        assert "scripts.verdict_from_csv %FILE%" in launcher

    def test_a_missing_file_stops_rather_than_reporting_nothing(self) -> None:
        """An empty result reads as "no trades", which is this project's
        signature failure."""
        source = Path("scripts/verdict_from_csv.py").read_text()

        assert 'raise SystemExit(f"no such file: {path}")' in source
        assert 'raise SystemExit(f"{path} has no TRADE rows")' in source
