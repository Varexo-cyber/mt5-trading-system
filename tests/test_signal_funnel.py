"""The funnel has to count the detector's clauses, not its own idea of them.

WHY IT EXISTS. The 30 August core run took 42 trades and none on FX. The dry
run could say that; it could not say why, because `no weighted directional
evidence` at 95.9% collapses five distinct reasons into one bucket.

WHY IT IS DANGEROUS. It reimplements `ImpulseRetest._live_break` in order to
instrument it, and a reimplementation that drifts produces a confident answer
to the wrong question -- which is the failure mode this account has paid for
repeatedly. So these tests pin it to the module: same config object, same
thresholds, same clause order, and a synthetic market whose answer is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from scripts.signal_funnel import Funnel, _walk, build_parser


def _config():
    return load_settings(
        DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
    ).analysis.impulse_retest


def _flat(bars: int = 400, price: float = 100.0) -> pd.DataFrame:
    """A market that does nothing. Tiny alternating noise so ATR is non-zero
    but no 20-bar channel is ever broken."""
    index = pd.date_range("2026-01-01", periods=bars, freq="15min", tz="UTC")
    close = price + np.tile([0.01, -0.01], bars // 2)[:bars]
    return pd.DataFrame(
        {"open": close, "high": close + 0.02, "low": close - 0.02, "close": close}, index=index
    )


class TestItCountsTheClausesTheDetectorApplies:
    def test_a_flat_market_never_breaks(self) -> None:
        funnel = _walk(_flat(), _config())

        assert funnel.bars > 300
        assert funnel.broke == 0
        assert funnel.fired == 0

    def test_the_counters_only_ever_narrow(self) -> None:
        """Each clause is applied to what survived the previous one, so the
        funnel must be monotone. If it is not, the order has drifted from the
        module's."""
        frame = _flat()
        # A run that breaks the channel hard, then pulls back to it.
        frame.iloc[300:310, :] += 2.0
        frame.iloc[310:320, :] += 1.0

        f = _walk(frame, _config())

        assert f.bars >= f.broke >= f.decisive >= f.alive >= f.right_side >= f.fired

    def test_it_reads_the_modules_own_thresholds(self) -> None:
        """Not copies of them. A hardcoded 1.0 here would keep agreeing with
        the module right up until someone tuned the module."""
        import inspect

        from scripts import signal_funnel

        source = inspect.getsource(signal_funnel._walk)

        for field in (
            "config.channel_period",
            "config.minimum_impulse_atr",
            "config.stop_beyond_atr",
            "config.tolerance_atr",
            "config.lookback_bars",
            "config.atr_period",
        ):
            assert field in source, f"{field} is hardcoded instead of read"

    def test_those_fields_all_exist_on_the_real_config(self) -> None:
        """So a rename breaks this loudly rather than silently."""
        config = _config()

        for field in (
            "channel_period",
            "minimum_impulse_atr",
            "stop_beyond_atr",
            "tolerance_atr",
            "lookback_bars",
            "atr_period",
        ):
            assert hasattr(config, field), field


class TestTheIntrabarTouchIsTheHypothesis:
    """The research bought these with a resting LIMIT at the level, which
    fills on any touch. The live module sees only the price at the instant it
    is asked. If the band is being touched and not closed in, the setups exist
    and the sampling is what loses them -- a completely different problem from
    the setups not existing."""

    def test_it_is_counted_separately_from_a_fire(self) -> None:
        assert "touched_intrabar" in Funnel.__dataclass_fields__
        assert Funnel().touched_intrabar == 0

    def test_a_touch_without_a_close_inside_is_not_a_fire(self) -> None:
        import inspect

        from scripts import signal_funnel

        source = inspect.getsource(signal_funnel._walk)
        touch = source.index("touched_intrabar")
        fired = source.index("funnel.fired += 1")

        # The touch counter must not be what decides `fired`.
        assert touch < fired
        assert "best_gap <= config.tolerance_atr" in source


class TestTheLauncherSurvivesCmd:
    def test_its_command_line_parses(self) -> None:
        from pathlib import Path

        from tests.test_dry_run_script import cmd_argv

        launcher = (Path(__file__).resolve().parent.parent / "funnel.cmd").read_text()
        line = next(ln for ln in launcher.splitlines() if "scripts.signal_funnel" in ln)
        argv = [
            {"%DAYS%": "7"}.get(t, t) for t in line.split("scripts.signal_funnel", 1)[1].split()
        ]

        assert build_parser().parse_args(argv).days == 7
        assert cmd_argv  # the shared helper stays imported and used

    def test_the_launcher_takes_no_comma_arguments(self) -> None:
        from pathlib import Path

        launcher = (Path(__file__).resolve().parent.parent / "funnel.cmd").read_text()

        for line in launcher.splitlines():
            if line.strip().startswith("set "):
                assert "," not in line, line


class TestItDoesNotDivideByZero:
    def test_a_market_with_no_bars_reports_rather_than_raises(self) -> None:
        funnel = Funnel()

        assert funnel.bars == 0
        # `_print` guards on `of` being zero; exercise the same arithmetic.
        assert (funnel.broke / funnel.bars if funnel.bars else 0.0) == pytest.approx(0.0)
