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
    rng = np.random.default_rng(4)
    index = pd.date_range("2024-01-01", periods=count, freq=f"{minutes}min", tz="UTC")
    step = rng.normal(0.0, 250.0, size=count).cumsum() + 724_000.0
    frame = pd.DataFrame(
        {
            "open": step,
            "high": step + np.abs(rng.normal(0.0, 200.0, size=count)),
            "low": step - np.abs(rng.normal(0.0, 200.0, size=count)),
            "close": step + rng.normal(0.0, 100.0, size=count),
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
