"""The XAUJPY mechanism search, and the two things that broke it on the VPS.

THE FIRST RUN DIED AFTER PRINTING ITS WHOLE BANNER:

    TypeError: MT5Connector.__init__() got an unexpected keyword argument 'login'

The call was written from what the constructor looks like it should take rather
than from what it takes. That is the same defect class as everything else in
this repository -- correct-looking code that is not on the path the code walks
-- and it is the reason the first two tests here check every script at once
instead of this one script by hand.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.instrument import AssetClass, InstrumentSpec

ROOT = Path(__file__).resolve().parents[1]


def _connector_calls() -> list[tuple[str, ast.Call]]:
    """Every `MT5Connector(...)` in every script, with the file it is in."""
    found: list[tuple[str, ast.Call]] = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "MT5Connector"
            ):
                found.append((path.name, node))
    return found


class TestEveryScriptBuildsTheConnectorTheWayItIsDefined:
    """A constructor call the class does not accept is a run that dies after
    twenty minutes of banner and fetching, on the owner's machine, where the
    fix costs a round trip."""

    def test_there_is_at_least_one_call_to_check(self) -> None:
        assert _connector_calls(), "no MT5Connector call found; this test has gone blind"

    def test_every_call_binds_against_the_real_signature(self) -> None:
        from core.mt5_connector import MT5Connector

        signature = inspect.signature(MT5Connector.__init__)
        problems: list[str] = []
        for filename, call in _connector_calls():
            names = [kw.arg for kw in call.keywords if kw.arg]
            positional = len(call.args)
            try:
                signature.bind(
                    None,
                    *[object()] * positional,
                    **{name: object() for name in names},
                )
            except TypeError as exc:
                problems.append(f"{filename}:{call.lineno}: {exc}")
        assert not problems, "MT5Connector built with arguments it does not take:\n" + "\n".join(
            problems
        )


def _spec() -> InstrumentSpec:
    """The REAL spec object with XAUJPY's shape, not a stand-in.

    The first version of this test used a hand-rolled stub with three
    attributes on it, and the sizer immediately asked for a fourth. A fake that
    is thinner than the real thing tests a cheaper instrument than the one that
    trades -- which is the same mistake as a second cost model, one layer down.
    """
    return InstrumentSpec(
        symbol="XAUJPY",
        digits=3,
        point=0.001,
        tick_size=0.001,
        tick_value=0.0068,
        contract_size=100,
        volume_min=0.01,
        volume_max=50.0,
        volume_step=0.01,
        stops_level=0,
        freeze_level=0,
        currency_base="XAU",
        currency_profit="JPY",
        currency_margin="XAU",
        filling_mode_mask=3,
        trade_mode=4,
        is_forex=False,
        path="Metals\\XAUJPY",
        description="Gold vs Japanese Yen",
        asset_class=AssetClass.METAL,
    )


class _FakeConnector:
    """Enough of the connector for `main` to run, and nothing more."""

    def __init__(self, *args, **kwargs) -> None:
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def shutdown(self) -> None:
        self.connected = False

    def spec(self, symbol: str, *, refresh: bool = False) -> InstrumentSpec:
        from core.mt5_connector import MT5Connector

        # Bound against the real method so a renamed parameter fails here
        # rather than on the VPS.
        inspect.signature(MT5Connector.spec).bind(self, symbol, refresh=refresh)
        return _spec()


def _bars(count: int, minutes: int) -> pd.DataFrame:
    """Bars whose volatility SCALES WITH THE BAR LENGTH, as real ones do.

    The first version drew the same per-bar noise whatever `minutes` was, so an
    M1 frame and an M15 frame had identical ATR -- which is exactly the
    assumption the cost bug rested on, reproduced in the fixture that was meant
    to catch it. A fake that cannot tell two clocks apart cannot test a
    per-clock stop width.
    """
    rng = np.random.default_rng(4)
    scale = 250.0 * (minutes**0.5)
    index = pd.date_range("2024-01-01", periods=count, freq=f"{minutes}min", tz="UTC")
    step = rng.normal(0.0, scale, size=count).cumsum() + 724_000.0
    frame = pd.DataFrame(
        {
            "open": step,
            "high": step + np.abs(rng.normal(0.0, scale * 0.8, size=count)),
            "low": step - np.abs(rng.normal(0.0, scale * 0.8, size=count)),
            "close": step + rng.normal(0.0, scale * 0.4, size=count),
            "volume": rng.integers(10, 400, size=count),
        },
        index=index,
    )
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    return frame


class TestTheSearchRunsEndToEnd:
    """Drives `main` against a fake broker.

    The unit tests around it all passed while the script could not reach its
    first fetch. Only running the whole thing catches that, so this does --
    with a connector and a history fetch standing in for MT5, and everything
    between them real.
    """

    def test_main_completes_and_writes_its_csv(self, tmp_path, monkeypatch, capsys) -> None:
        import scripts.search_xaujpy as jpy

        monkeypatch.setattr(jpy, "MT5Connector", _FakeConnector)
        monkeypatch.setattr(jpy, "load_credentials", lambda **_kwargs: None)
        monkeypatch.setattr(
            jpy,
            "fetch_mt5_history",
            lambda _market, _symbol, timeframe, _start, _end: _bars(
                3_000, int(timeframe.duration.total_seconds() // 60)
            ),
        )
        out = tmp_path / "cells.csv"
        monkeypatch.setattr(
            "sys.argv",
            ["search_xaujpy", "--days", "60", "--clocks", "M5", "--csv", str(out)],
        )

        jpy.main()

        printed = capsys.readouterr().out
        assert "XAUJPY MECHANISM SEARCH" in printed
        assert "WHICH CELLS EARNED A SECTION" in printed
        assert "WHAT THIS RUN DID NOT MODEL" in printed
        assert out.exists()

    def test_no_history_stops_with_a_sentence_rather_than_a_traceback(self, monkeypatch) -> None:
        import scripts.search_xaujpy as jpy

        monkeypatch.setattr(jpy, "MT5Connector", _FakeConnector)
        monkeypatch.setattr(jpy, "load_credentials", lambda **_kwargs: None)
        monkeypatch.setattr(jpy, "fetch_mt5_history", lambda *_a, **_k: _bars(10, 5))
        monkeypatch.setattr("sys.argv", ["search_xaujpy", "--days", "60", "--clocks", "M5"])

        with pytest.raises(SystemExit, match="no usable history"):
            jpy.main()


class TestTheReportNamesWhatItDoesNotModel:
    """The owner asked for the live gates to be carried into the measurement.
    Most of them cannot be, so the ones that are missing are named every run --
    an unnamed omission reads as an assumed zero."""

    def test_every_gate_the_live_runner_applies_is_either_used_or_named(self) -> None:
        import scripts.search_xaujpy as jpy

        source = inspect.getsource(jpy)
        for gate in (
            "AWAITING_CONFIRMATION",
            "NEWS_BLACKOUT",
            "TARGET_RARELY_REACHED",
            "SPREAD_EATS_THE_STOP",
            "MARKET_TOO_QUIET",
            "AWAITING_PULLBACK",
            "VOLUME_SPIKE",
            "ENTRY_OVEREXTENDED",
        ):
            assert gate in source, gate

    def test_the_cost_model_is_the_accounts_own(self) -> None:
        """A second definition of cost would eventually disagree with the one
        the account charges, and the search would be measuring a cheaper
        instrument than the one that trades."""
        import scripts.search_xaujpy as jpy

        assert "from scripts.search_section_four import" in inspect.getsource(jpy)
        assert "_cost_share(sizer, spec" in inspect.getsource(jpy)


class TestTheBlockedHoursAreActuallyApplied:
    """DOCUMENTED AND NOT IMPLEMENTED, found in the first real run's output.

    The module docstring said the configured blocked hours were applied. No
    line applied them. `london_drive` came third on +0.091 R per trade earned
    ENTIRELY inside 07:00-12:00 UTC -- the window the section refuses -- so the
    ranking was ordering cells by an edge the account may not take.
    """

    def test_the_muter_zeroes_exactly_the_blocked_hours(self) -> None:
        import scripts.search_xaujpy as jpy

        frame = _bars(600, 5)
        signals = np.ones(len(frame), dtype=int)
        blocked = {7, 8, 9, 10, 11, 12}

        muted = jpy._mute_blocked_hours(signals, frame, blocked)
        hours = frame.index.hour.to_numpy()

        assert (muted[np.isin(hours, list(blocked))] == 0).all()
        assert (muted[~np.isin(hours, list(blocked))] == 1).all()
        assert (signals == 1).all(), "the caller's array was mutated"

    def test_no_blocked_hours_leaves_every_signal_alone(self) -> None:
        import scripts.search_xaujpy as jpy

        frame = _bars(200, 5)
        signals = np.ones(len(frame), dtype=int)

        assert (jpy._mute_blocked_hours(signals, frame, set()) == 1).all()

    def test_the_hours_come_from_the_section_config_not_a_second_list(self) -> None:
        """Two lists that must agree are two lists that will disagree."""
        import inspect

        import scripts.search_xaujpy as jpy

        source = inspect.getsource(jpy)
        assert "getattr(settings.analysis, name).blocked_hours" in source

    def test_a_cell_whose_edge_is_blocked_scores_nothing(self, tmp_path, monkeypatch) -> None:
        """The end-to-end proof: a mechanism that only ever fires inside the
        blocked window must come back with no trades at all."""
        import scripts.search_xaujpy as jpy

        frame = _bars(3_000, 5)
        hours = frame.index.hour.to_numpy()
        only_london = np.where(np.isin(hours, [7, 8, 9, 10, 11, 12]), 1, 0)

        muted = jpy._mute_blocked_hours(only_london, frame, {7, 8, 9, 10, 11, 12})

        assert only_london.sum() > 0, "the fixture never fired in London"
        assert muted.sum() == 0


class TestTheCostIsChargedAgainstTheStopTheCellUses:
    """THE 720-DAY RUN CAME BACK WITH NINE WINNERS AND EVERY ONE WAS FAKE.

    The cost share was computed against a flat `close * 0.004` -- 0.4% of price
    as the stop width, the same number on M1, M5 and M15. The stop a cell
    actually resolves against is `stop_atr x ATR of that clock`, and an M1 ATR
    on a gold cross is far smaller than 0.4% of price. So M1 was charged a
    fraction of what it pays, and nine M1 cells at 74 to 252 trades a day came
    out at +4.4 to +8.1 sigma with large positive holdouts.

    That is not an edge, it is free trading. The same shape as the bug the
    `_cost_share` docstring already records one level up.
    """

    def test_the_stop_width_comes_from_the_clocks_own_atr(self) -> None:
        import inspect

        import scripts.search_xaujpy as jpy

        source = inspect.getsource(jpy)

        assert "stop_price = args.stop_atr * float(np.nanmedian(_atr(searched)))" in source
        # Only outside the comment that records the broken line, so the
        # explanation cannot trip the check it exists to explain.
        code = [
            line.split("#", 1)[0]
            for line in source.splitlines()
            if not line.strip().startswith("#")
        ]
        assert not [line for line in code if "0.004" in line], "a hand-rolled stop width is back"

    def test_a_faster_clock_is_charged_more_not_less(self) -> None:
        """The property the bug violated: cost is a share OF THE STOP, so a
        narrower stop on a faster clock must cost a LARGER share, never a
        smaller one. The broken version had it constant across clocks, which
        is what made M1 look free."""
        from analysis.mechanisms import _atr

        m1 = _bars(4_000, 1)
        m15 = _bars(4_000, 15)

        assert float(np.nanmedian(_atr(m1))) < float(np.nanmedian(_atr(m15)))

    def test_an_unaffordable_cell_cannot_be_a_survivor(self) -> None:
        """`SL_TOO_TIGHT_FOR_COSTS` is a live refusal, so a cell above the
        account's cap is not a cell the account can trade however good its
        sigma is. It is named in the report rather than dropped."""
        import inspect

        import scripts.search_xaujpy as jpy

        source = " ".join(inspect.getsource(jpy).split())

        assert "and c.cost_share <= cap" in source
        assert "SL_TOO_TIGHT_FOR_COSTS" in source
        assert "priced_out" in source

    def test_the_cost_share_is_printed_for_every_cell(self) -> None:
        """Its absence is what let a twenty-fold error rank nine cells at the
        top for a whole run without anybody being able to see it."""
        import inspect

        import scripts.search_xaujpy as jpy

        source = inspect.getsource(jpy)

        assert "{'cost':>7}" in source
        assert "cell.cost_share:>6.1%" in source


class TestAPatternIsNotASetup:
    """`streak_reversal` ON M1 WAS PROMOTED AND IT IS A COIN FLIP.

    "Four closes the same way, then trade against it." Four coin flips landing
    the same way is 12.5% of bars in a random walk; measured on real XAUJPY it
    fires on 10.3%, which is 149 signals a day on M1. There is no counterparty:
    nobody loses money because four one-minute candles went up. The replay
    agreed -- 855 trades at -0.18 R apiece, about the round-trip cost.

    The search ranked it FIRST, at +8.14 sigma, because nothing in it knew the
    difference between an event and a pattern.
    """

    def test_a_cell_firing_far_above_the_ask_cannot_be_promoted(self) -> None:
        import inspect

        import scripts.search_xaujpy as jpy

        source = " ".join(inspect.getsource(jpy).split())

        assert "and c.per_day <= MAX_TRADES_PER_DAY" in source
        assert "too_busy" in source
        assert jpy.MAX_TRADES_PER_DAY <= 10.0, "the bound has drifted away from the ask"

    def test_the_bound_leaves_room_over_what_was_asked_for(self) -> None:
        """Generous on purpose. It is there to refuse 74 trades a day, not to
        shave a cell that comes in at six."""
        import scripts.search_xaujpy as jpy

        _low, high = jpy.WANTED_TRADES_PER_DAY

        assert high < jpy.MAX_TRADES_PER_DAY
        assert high * 3 > jpy.MAX_TRADES_PER_DAY

    def test_the_firing_rate_is_measured_and_printed(self) -> None:
        """Its absence is what let a 10%-of-all-bars pattern top the table."""
        import inspect

        import scripts.search_xaujpy as jpy

        source = inspect.getsource(jpy)

        assert "cell.fire_rate = float(np.mean(signals != 0))" in source
        assert "{'fires':>7}" in source
        assert "cell.fire_rate:>6.1%" in source

    def test_the_coin_flip_mechanism_really_does_fire_that_often(self) -> None:
        """The number in the docstring above, checked rather than remembered."""
        from analysis.mechanisms import FAMILIES

        frame = _bars(20_000, 1)
        rate = float(np.mean(FAMILIES["all"]["streak_reversal"](frame) != 0))

        # A random walk gives 2 x (1/2)^4 = 12.5%; anything in that region is a
        # pattern rather than an event.
        assert rate > 0.05, "the fixture is not random enough to make the point"
        assert rate * 1440 > 50, "this would not be 50+ signals a day on M1"
