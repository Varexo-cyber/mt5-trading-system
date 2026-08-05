"""Monitor mode: looks completely healthy, cannot trade, must not spend.

An hour of live running produced 79 adviser calls, 11 approvals and 0 trades.
Nothing was broken — monitor is the default operation and it returns before the
order every time. But it returned *after* paying for a verdict it could never
act on, and the only sign of the mode anywhere was a five-letter word in a
metric nobody reads as a warning.
"""

from __future__ import annotations

from dashboard.ai_exchange import NON_TRADING_OPERATIONS, cannot_trade, read_operation
from runner.service import OperationMode


def test_monitor_is_the_default_operation() -> None:
    """Which is exactly why it needs saying out loud: a session started without
    an explicit mode is in the one mode that cannot place an order."""
    import inspect

    signature = inspect.signature(type(OperationMode.MONITOR).__call__)
    del signature  # only the enum value matters below
    from runner.service import JarvisRunner

    default = inspect.signature(JarvisRunner.__init__).parameters["operation"].default
    assert default is OperationMode.MONITOR


def test_monitor_is_flagged_as_non_trading() -> None:
    assert cannot_trade(OperationMode.MONITOR.value)


def test_the_modes_that_do_trade_are_not_flagged() -> None:
    """Paper places real orders against the paper broker, so it exercises the
    full path — including the adviser — and must not be warned about."""
    for mode in (
        OperationMode.PAPER,
        OperationMode.DEMO,
        OperationMode.LIVE,
        OperationMode.EXPERIMENTAL_LIVE,
    ):
        assert not cannot_trade(mode.value), mode


def test_every_flagged_operation_is_a_real_one() -> None:
    """A typo here would silently disable the warning."""
    known = {mode.value for mode in OperationMode}
    assert known >= NON_TRADING_OPERATIONS


def test_the_operation_is_read_from_the_heartbeat(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "heartbeat.json"
    path.write_text('{"operation": "monitor", "inspected": 88}', encoding="utf-8")
    assert read_operation(path) == "monitor"
    assert cannot_trade(read_operation(path))


def test_a_missing_heartbeat_is_not_reported_as_monitor(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Jarvis not running at all is a different problem, and warning about the
    wrong one sends the operator to the wrong fix."""
    assert read_operation(tmp_path / "nothing.json") == ""
    assert not cannot_trade("")


def test_the_adviser_is_not_called_in_monitor_mode() -> None:
    """The money question. Monitor returns before the order regardless, so a
    paid verdict buys nothing — and 79 of them went out before anyone noticed.
    """
    import inspect

    from runner.service import JarvisRunner

    source = inspect.getsource(JarvisRunner._process_candidate)
    assert "self._reviewed(" in source, "the adviser is reached via _reviewed; update this test"
    monitor_guard = source.index("OperationMode.MONITOR")
    review_call = source.index("self._reviewed(")
    assert monitor_guard < review_call, (
        "the monitor check must come before the adviser call, or monitor mode "
        "pays for verdicts it can never act on"
    )


def test_monitor_leaves_rather_than_falling_through() -> None:
    """Ordering alone is not enough. A check that logs and carries on would sit
    in the right place and still spend the money."""
    import inspect

    from runner.service import JarvisRunner

    source = inspect.getsource(JarvisRunner._process_candidate)
    between = source[source.index("OperationMode.MONITOR") : source.index("self._reviewed(")]
    assert "return False" in between
