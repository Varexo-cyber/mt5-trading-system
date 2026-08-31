"""The one unchecked number that decides whether FX may trade.

`stop_slippage_pips.forex` is 1.7, and on EURUSD that is $17 of a $24.50
round trip -- two thirds of the entire cost model. It is not a measurement. Its
own docstring names the single fill it came from.

Against `max_cost_share_of_risk = 0.12` it is the difference between FX being
refused on every clock and FX clearing on H1. So it gets measured from the
account's own fills rather than assumed, and these tests hold the measurement
to the things that would make it lie.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.measure_slippage import asset_class, pip_size, stop_outs


def journal_with(tmp_path: Path, rows: list[tuple]) -> sqlite3.Connection:
    path = tmp_path / "trading.db"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE trades (id INTEGER PRIMARY KEY, symbol TEXT, direction TEXT, "
        "entry_price REAL, sl REAL, tp REAL, exit_price REAL, sl_distance_pips REAL, "
        "closed_at TEXT, exit_reason TEXT, pnl_r REAL, opened_at TEXT)"
    )
    db.executemany(
        "INSERT INTO trades (symbol, direction, entry_price, sl, exit_price, "
        "sl_distance_pips, closed_at, exit_reason, pnl_r, opened_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    db.commit()
    db.row_factory = sqlite3.Row
    return db


class TestClassifyingASymbolWithoutABroker:
    """This runs against a journal, so there is no broker path to ask. A
    misfiled index would land in the number that decides FX."""

    def test_the_majors(self) -> None:
        for symbol in ("EURUSD.i", "GBPJPY.r", "USDCHF", "audcad.i", "NZDCAD"):
            assert asset_class(symbol) == "forex", symbol

    def test_metals_and_indices_and_crypto(self) -> None:
        assert asset_class("XAUUSD") == "metal"
        assert asset_class("XAGUSD.i") == "metal"
        assert asset_class("US30.i") == "index"
        assert asset_class("GER40") == "index"
        assert asset_class("JPN225") == "index"
        assert asset_class("SPX500.i") == "index"
        assert asset_class("BTCUSD") == "crypto"

    def test_a_dollar_index_is_not_a_currency_pair(self) -> None:
        """USDX is six characters and starts with USD. "X" is not a currency,
        and this symbol has already caused one matching bug on this account."""
        assert asset_class("USDX") != "forex"
        assert asset_class("USDX.i") != "forex"

    def test_an_unknown_symbol_is_named_rather_than_guessed(self) -> None:
        """Dropping it into forex would corrupt the only number this script
        exists to produce."""
        assert asset_class("COFFEE") == "unclassified"
        assert asset_class("TSLA") == "unclassified"


class TestFindingTheStopOuts:
    """Identified geometrically, not by `exit_reason`. Those strings vary by
    close path and a label filter would silently miss whole categories."""

    def test_a_long_filled_through_its_stop_is_adverse_slippage(self, tmp_path: Path) -> None:
        # entry 1.1000, stop 1.0990 = 10 pips, so one pip is 0.0001.
        # Filled at 1.09883: 1.7 pips past the stop.
        db = journal_with(
            tmp_path,
            [("EURUSD.i", "LONG", 1.1000, 1.0990, 1.09883, 10.0, "t", "SL", -1.0, "t")],
        )
        found, _ = stop_outs(db)

        assert len(found) == 1
        assert found[0]["slippage_pips"] == pytest.approx(1.7, abs=0.01)
        assert found[0]["asset_class"] == "forex"

    def test_a_short_filled_through_its_stop(self, tmp_path: Path) -> None:
        db = journal_with(
            tmp_path,
            [("EURUSD.i", "SHORT", 1.1000, 1.1010, 1.10117, 10.0, "t", "SL", -1.0, "t")],
        )
        assert stop_outs(db)[0][0]["slippage_pips"] == pytest.approx(1.7, abs=0.01)

    def test_a_better_than_asked_fill_reads_negative(self, tmp_path: Path) -> None:
        """A gap can fill a stop IN YOUR FAVOUR, and pretending otherwise would
        bias the average upward -- which is the direction that keeps FX
        refused. The sign is kept."""
        db = journal_with(
            tmp_path,
            [("EURUSD.i", "LONG", 1.1000, 1.0990, 1.09905, 10.0, "t", "SL", -1.0, "t")],
        )
        assert stop_outs(db)[0][0]["slippage_pips"] == pytest.approx(-0.5, abs=0.01)

    def test_a_trade_that_closed_at_target_is_not_a_stop_out(self, tmp_path: Path) -> None:
        db = journal_with(
            tmp_path,
            [("EURUSD.i", "LONG", 1.1000, 1.0990, 1.1010, 10.0, "t", "TP", 1.0, "t")],
        )
        assert stop_outs(db)[0] == []

    def test_a_manual_close_between_entry_and_stop_is_not_one_either(self, tmp_path: Path) -> None:
        """Half way to the stop is not a stop-out, and counting it would put a
        large negative "slippage" into the average."""
        db = journal_with(
            tmp_path,
            [("EURUSD.i", "LONG", 1.1000, 1.0990, 1.0995, 10.0, "t", "MANUAL", -0.5, "t")],
        )
        assert stop_outs(db)[0] == []

    def test_an_exact_fill_at_the_stop_still_counts(self, tmp_path: Path) -> None:
        """Zero slippage is the observation that most needs to be in the
        average. Floating-point equality would drop it."""
        db = journal_with(
            tmp_path,
            [("EURUSD.i", "LONG", 1.1000, 1.0990, 1.0990, 10.0, "t", "SL", -1.0, "t")],
        )
        found, _ = stop_outs(db)

        assert len(found) == 1
        assert found[0]["slippage_pips"] == pytest.approx(0.0, abs=0.01)

    def test_the_pip_comes_from_the_trade_s_own_row(self, tmp_path: Path) -> None:
        """No pip table and no broker lookup: `sl_distance_pips` sits next to
        `entry_price` and `sl`, so one pip is derivable per trade and cannot
        disagree with what the sizer wrote down. JPY and gold have different
        pip sizes and both fall out of this for free."""
        db = journal_with(
            tmp_path,
            [
                # USDJPY: pip is 0.01. 20-pip stop, filled 2 pips past.
                ("USDJPY.i", "LONG", 150.00, 149.80, 149.78, 20.0, "t", "SL", -1.0, "t"),
                # XAUUSD: pip is 0.1 here. 30-pip stop, filled 3 pips past.
                ("XAUUSD", "LONG", 4400.0, 4397.0, 4396.7, 30.0, "t", "SL", -1.0, "t"),
            ],
        )
        found = {row["symbol"]: row for row in stop_outs(db)[0]}

        assert found["USDJPY.i"]["slippage_pips"] == pytest.approx(2.0, abs=0.01)
        assert found["XAUUSD"]["slippage_pips"] == pytest.approx(3.0, abs=0.01)
        assert found["XAUUSD"]["asset_class"] == "metal"

    def test_open_trades_are_ignored(self, tmp_path: Path) -> None:
        db = journal_with(tmp_path, [])
        db.execute(
            "INSERT INTO trades (symbol, direction, entry_price, sl, exit_price, "
            "sl_distance_pips, closed_at) VALUES ('EURUSD.i','LONG',1.1,1.099,NULL,10.0,NULL)"
        )
        db.commit()

        assert stop_outs(db)[0] == []


class TestTheConfiguredNumberIsStillAnAssumption:
    def test_the_forex_row_is_the_one_that_matters(self) -> None:
        """1.7 pips on a 3.5-pip M15 stop is 49% of the stop from slippage
        alone, before commission or spread. Nothing else in the cost model is
        anywhere near that size."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        slippage = settings.risk.stop_slippage_pips

        assert slippage["forex"] == pytest.approx(1.7)
        # Commission is 5.50 round trip = 0.55 pip on a major. Slippage is
        # three times that, and it is the number nobody has checked.
        assert settings.risk.commission_per_lot("forex") / 10.0 < slippage["forex"]


class TestTheBugThatProducedEightThousandPips:
    """The first version reported a forex median of 8563 pips.

    It derived one pip from the trade row: `|entry - sl| / sl_distance_pips`.
    That looks exact -- both numbers are written by the sizer, so they cannot
    disagree -- and it is wrong for one reason. `trades.sl` is the CURRENT
    stop, not the one the trade opened with, and break-even management moves
    it to entry. `|entry - sl|` collapses toward zero, the derived pip with it,
    and every slippage divided by that pip explodes.

    On this account a moved stop is the normal case, so the derivation was
    broken for most rows rather than a few.
    """

    def test_a_moved_stop_is_measured_not_discarded(self, tmp_path: Path) -> None:
        """Entry 1.1000, opened with a 10-pip stop, break-even moved it to
        1.09995, filled at 1.09990 -- half a pip past the ACTIVE stop.

        The pip-from-the-row version called one pip 0.0000005 here and read
        this fill as thousands of pips. The fix for that was to derive the pip
        from the symbol; discarding the row as well was a second, wrong fix,
        and it threw away 222 of 349 rows on the first real run, leaving five.

        Break-even moves the stop on most trades on this account, so those 222
        ARE the data. A break-even stop is an ordinary stop at a different
        price, and how far past it a fill lands is exactly the question."""
        db = journal_with(
            tmp_path,
            [("EURUSD.i", "LONG", 1.1000, 1.09995, 1.09990, 10.0, "t", "SL", -0.1, "t")],
        )

        found, discarded = stop_outs(db)

        assert len(found) == 1
        assert found[0]["slippage_pips"] == pytest.approx(0.5, abs=0.01)
        assert discarded == {}

    def test_a_stop_exactly_at_entry_has_nothing_to_measure(self, tmp_path: Path) -> None:
        """The one case that really is unusable: no distance, so no order to
        slip against."""
        db = journal_with(
            tmp_path,
            [("EURUSD.i", "LONG", 1.1000, 1.1000, 1.0999, 10.0, "t", "SL", 0.0, "t")],
        )

        found, discarded = stop_outs(db)

        assert found == []
        assert any("exactly at entry" in why for why in discarded)

    def test_an_untouched_stop_still_measures(self, tmp_path: Path) -> None:
        db = journal_with(
            tmp_path,
            [("EURUSD.i", "LONG", 1.1000, 1.0990, 1.09883, 10.0, "t", "SL", -1.0, "t")],
        )

        found, discarded = stop_outs(db)

        assert len(found) == 1
        assert found[0]["slippage_pips"] == pytest.approx(1.7, abs=0.01)
        assert discarded == {}

    def test_slippage_wider_than_the_stop_is_refused(self, tmp_path: Path) -> None:
        """Not a fill, a broken row -- a spec change or a rename. Averaging it
        in is how one bad record moves a number that decides live trading."""
        db = journal_with(
            tmp_path,
            [("EURUSD.i", "LONG", 1.1000, 1.0990, 1.0960, 10.0, "t", "SL", -4.0, "t")],
        )

        found, discarded = stop_outs(db)

        assert found == []
        assert any("whole original stop" in why for why in discarded)


class TestOnePipWithoutABroker:
    """The journal has no `InstrumentSpec`, so the pip comes from the symbol's
    name and its price. It has to match what the terminal would say."""

    def test_the_conventions(self) -> None:
        assert pip_size("EURUSD.i", 1.10) == pytest.approx(0.0001)
        assert pip_size("USDJPY.i", 150.0) == pytest.approx(0.01)
        assert pip_size("GBPJPY", 195.0) == pytest.approx(0.01)
        assert pip_size("XAUUSD", 4400.0) == pytest.approx(0.1)
        assert pip_size("XAGUSD", 30.0) == pytest.approx(0.001)
        assert pip_size("GER40", 8300.0) == pytest.approx(1.0)
        assert pip_size("BTCUSD", 60000.0) == pytest.approx(1.0)

    def test_it_cannot_depend_on_the_moved_stop(self) -> None:
        """The whole point, asserted structurally rather than by grepping the
        source for the letters "sl" -- which matched its own docstring. The
        function takes a symbol and a price. There is no parameter through
        which a stop could reach it, moved or otherwise."""
        import inspect

        from scripts import measure_slippage

        parameters = list(inspect.signature(measure_slippage.pip_size).parameters)

        assert parameters == ["symbol", "price"]
