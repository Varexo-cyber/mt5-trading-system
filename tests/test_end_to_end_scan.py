"""A full runner cycle against a catalogue shaped like a real broker's.

Every unit test in this repo feeds the system spot FX: continuous bars, fresh
quotes, five symbols. A live Eightcap catalogue is 843 instruments, most of them
*not* FX — shares that trade eight hours a day, grain futures that halt for
several, indices with their own schedule — and at any given moment most of those
exchanges are shut.

Three production failures came from that gap, and all three would have been
caught here:

* WHEAT H1 rejected with "415 bars missing inside trading weeks (20.8%)". The
  gap check assumed 24-hour trading, so a daily session break read as a hole.
* Every symbol rejected as DATA_UNAVAILABLE on Monday morning, because bar age
  was counted from the bar's open and the weekend counted as missing data.
* The broker-time offset flapping many times a second, because a closed
  exchange's last quote looks exactly like a different timezone.

So this exercises `run_once` end to end — scan, data, filters, sizing — over a
mixed catalogue on a Monday morning, and asserts the system distinguishes "this
market is shut" from "this data is broken".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from advisory.providers import Advice, Reflection
from config.loader import DEFAULT_CONFIG_PATH, load_settings
from config.schema import MT5Config
from core.clock import SimulatedClock
from core.mt5_connector import MT5Connector
from risk.reasons import Reason
from runner.service import JarvisRunner, OperationMode
from tests.fakes.fake_mt5 import FakeMT5, eurusd_spec

# Monday, 07:49 UTC — exactly when the live run reported a dead universe.
# London is open, Chicago grains are not, and the weekend is just behind us.
MONDAY = datetime(2026, 8, 3, 7, 49, tzinfo=UTC)

#: name -> (session hours UTC or None for 24h, how stale its last tick is)
CATALOGUE: dict[str, tuple[tuple[int, int] | None, timedelta]] = {
    # Spot FX: continuous, always quoting.
    "EURUSD": (None, timedelta(0)),
    "GBPUSD": (None, timedelta(0)),
    "USDJPY": (None, timedelta(0)),
    # Grains: a daily halt. This is the WHEAT case.
    "WHEAT": ((1, 20), timedelta(0)),
    # London shares: open at 08:00 UTC, so at 07:49 they are shut and their last
    # quote is Friday's close. Whole hours stale, which is what made it look
    # like a timezone.
    "BRBY": ((8, 16), timedelta(hours=2)),
    "ENI": ((8, 16), timedelta(hours=3)),
    # An index that trades nearly around the clock.
    "SPX500": (None, timedelta(0)),
}


#: The scanner classifies instruments by the folder the broker files them under,
#: so the fake catalogue has to use the same shape a real one does.
FOLDERS = {
    "EURUSD": "Forex\\Majors",
    "GBPUSD": "Forex\\Majors",
    "USDJPY": "Forex\\Majors",
    "WHEAT": "Commodities\\Agriculture",
    "BRBY": "Shares\\UK",
    "ENI": "Shares\\Italy",
    "SPX500": "Indices\\US",
}


def _spec(name: str) -> SimpleNamespace:
    return eurusd_spec(name=name, path=f"{FOLDERS[name]}\\{name}", description=name)


@pytest.fixture
def fake() -> FakeMT5:
    return FakeMT5(
        equity=100.0,
        balance=100.0,
        currency="EUR",
        is_demo=True,
        now=MONDAY,
        specs={name: _spec(name) for name in CATALOGUE},
        quotes=dict.fromkeys(CATALOGUE, (1.085, 1.08512)),
        session_hours={n: s for n, (s, _) in CATALOGUE.items() if s is not None},
        tick_age={n: age for n, (_, age) in CATALOGUE.items() if age},
    )


@pytest.fixture
def settings(tmp_path: Path) -> Any:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["system"]["mode"] = "micro_live"
    raw["instruments"]["universe_mode"] = "affordable"
    raw["instruments"]["whitelist"]["micro_live"] = list(CATALOGUE)
    raw["journal"]["database_path"] = str(tmp_path / "journal.db")
    raw["ai"]["enabled"] = False
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_settings(path, env_overrides=False)


class _ApprovingAdvisor:
    """Stands in for Claude so the cycle runs without a network call."""

    def review(self, *_args: object, **_kwargs: object) -> Advice:
        return Advice(True, 1.0, "test", provider="fake")

    def reflect(self, *_args: object, **_kwargs: object) -> Reflection:
        return Reflection("test", provider="fake")


@pytest.fixture
def runner(fake: FakeMT5, settings: Any, tmp_path: Path) -> JarvisRunner:
    connector = MT5Connector(MT5Config(), mt5_module=fake)
    service = JarvisRunner(
        connector,
        settings,
        tmp_path,
        OperationMode.MONITOR,
        advisor=_ApprovingAdvisor(),
        clock=SimulatedClock(MONDAY),
    )
    service.connect()
    return service


def _decisions(runner: JarvisRunner) -> dict[str, tuple[str, str]]:
    """symbol -> (decision, reason) from the journal's own record of the cycle."""
    rows = runner.journal.conn.execute(
        "SELECT symbol, decision, reason FROM analysis_cycles ORDER BY id"
    ).fetchall()
    return {str(r[0]): (str(r[1]), str(r[2])) for r in rows}


class TestMixedCatalogue:
    def test_a_cycle_completes_over_a_mixed_catalogue(self, runner: JarvisRunner) -> None:
        summary = runner.run_once()

        assert summary.universe_size == len(CATALOGUE)
        assert summary.inspected > 0
        runner.close()

    def test_a_closed_exchange_is_not_reported_as_broken_data(self, runner: JarvisRunner) -> None:
        """Shut is not the same as broken, and the journal has to say which.

        WHEAT halts daily but is open at this hour, and the FX and index rows
        never close. None of them has a data fault, and labelling a session
        break one hid the real faults among hundreds of false ones.

        The London shares are excluded because at 07:49 they genuinely are shut:
        their newest intraday bars are Friday's, which is a real reason not to
        analyse them for an entry now. "Shut is not broken" is about markets
        that are trading, not about pretending a closed one is usable.
        """
        runner.run_once()
        decisions = _decisions(runner)
        open_now = {name for name in CATALOGUE if not FOLDERS[name].startswith("Shares")}

        broken = [
            s for s, (_, r) in decisions.items() if r == Reason.DATA_UNAVAILABLE and s in open_now
        ]
        assert broken == [], f"open markets rejected as broken data: {broken}"
        runner.close()

    def test_the_daily_session_break_is_not_counted_as_missing_bars(
        self, runner: JarvisRunner
    ) -> None:
        """The WHEAT case, isolated: a five-hour daily halt is not a gap."""
        from core.types import Timeframe

        series = runner.data.get_series("WHEAT", Timeframe.H1)

        assert len(series) > 0
        runner.close()

    def test_a_closed_exchange_does_not_move_the_broker_clock(
        self, runner: JarvisRunner, fake: FakeMT5
    ) -> None:
        """The offset must survive a scan that reads shut markets.

        BRBY and ENI serve quotes two and three whole hours stale. Read as
        timezones they are indistinguishable from the real offset, and taking
        the most recent reading made the value oscillate on every cycle.
        """
        runner.run_once()
        after_first = runner.broker.server_offset

        runner.run_once()

        # Stability is the property. The absolute value depends on the wall
        # clock the connector compares tick timestamps against, which is
        # deliberately not the simulated one — brokers stamp ticks in real time.
        assert runner.broker.server_offset == after_first
        runner.close()

    def test_the_cycle_reports_what_it_did(self, runner: JarvisRunner) -> None:
        """Silence and a hang have to look different in the console."""
        summary = runner.run_once()

        assert summary.finished_at >= summary.started_at
        assert summary.inspected + summary.rejected > 0
        assert (runner.root / "runtime" / "heartbeat.json").exists()
        runner.close()

    def test_a_hundred_euro_account_skips_rather_than_oversizes(self, runner: JarvisRunner) -> None:
        """Whatever else happens, the account must never exceed its risk.

        At EUR 100 most setups cannot be expressed at 1% risk. The correct
        outcome is a skip; the dangerous one is rounding up to the minimum lot.
        """
        runner.run_once()
        decisions = _decisions(runner)

        assert all(r != Reason.RISK_EXCEEDS_CAP for _, r in decisions.values())
        assert not runner.broker.positions(magic=runner.settings.system.magic_number)
        runner.close()
