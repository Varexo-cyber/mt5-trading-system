"""`_record_skip` invites the scalp lane in; the scalp lane records its own
refusals through `_record_skip`.

WHAT THAT DID. Within an hour of section six being risk-sized, the VPS filled
with this, thousands of frames deep:

    service.py, line 4012, in _record_skip
        if signals and self._scalp_lane_took_it(cycle_id, symbol, ...)
    service.py, line 4461, in _scalp_lane_took_it
        return self._run_scalp_lane(cycle_id, symbol, signals, reason, extra)
    service.py, line 4633, in _run_scalp_lane
        self._record_skip(
    ...
    RecursionError: maximum recursion depth exceeded

and then a second RecursionError inside the logging formatter, trying to
render the traceback of the first.

THE LOOP WAS ALWAYS THERE. What changed is how often the lane refuses. It used
to send a FIXED 0.01 lot, so the refusal paths inside it -- a lot under the
broker minimum, a failed margin check, a full risk book -- practically never
fired. Sizing it at 3% of equity made all three ordinary, and a latent bug
became a live one the same day.

A guard, not a restructure: the lane may not be re-entered while it is already
running, and a refusal it records is a plain journal write like any other.
"""

from __future__ import annotations

from runner.service import JarvisRunner


def _runner():  # type: ignore[no-untyped-def]
    """A runner whose scalp lane always refuses, the way the live one did."""
    service = object.__new__(JarvisRunner)
    service.skips: list[str] = []  # type: ignore[attr-defined]
    service.entries = 0  # type: ignore[attr-defined]

    def record_skip(cycle_id, symbol, equity, reason, detail, **kwargs):  # type: ignore[no-untyped-def]
        service.skips.append(str(reason))  # type: ignore[attr-defined]
        # The real one invites the lane in from here. That is the loop.
        signals = kwargs.get("signals") or ["a signal"]
        if signals and not service._inside_scalp_lane:
            service._scalp_lane_took_it(cycle_id, symbol, signals, reason, None)
        return 1

    def run_lane(cycle_id, symbol, signals, reason, extra):  # type: ignore[no-untyped-def]
        service.entries += 1  # type: ignore[attr-defined]
        # Exactly what the live lane does when the sizer comes back too small
        # for the broker minimum: records a refusal and gives up.
        service._record_skip(cycle_id, symbol, 176.0, "TRADE_SKIPPED_UNDERCAPITALIZED", "too small")
        return False

    service._record_skip = record_skip  # type: ignore[assignment]
    service._run_scalp_lane = run_lane  # type: ignore[assignment]
    return service


def test_a_lane_that_refuses_does_not_invite_itself_back_in() -> None:
    """The crash, reduced. Without the guard this never returns."""
    service = _runner()

    service._record_skip("cycle-1", "XAUUSD", 176.0, "NO_SIGNAL", "nothing here")

    assert service.entries == 1, f"the lane ran {service.entries} times for one skip"
    assert len(service.skips) == 2, "one skip in, one refusal recorded by the lane"


def test_the_flag_is_cleared_so_the_next_cycle_still_gets_its_turn() -> None:
    """A guard that latches would silently switch section six off for the rest
    of the process, which is the same failure wearing a quieter face."""
    service = _runner()

    service._record_skip("cycle-1", "XAUUSD", 176.0, "NO_SIGNAL", "nothing here")
    service._record_skip("cycle-2", "XAUUSD", 176.0, "NO_SIGNAL", "nothing here")

    assert service.entries == 2
    assert service._inside_scalp_lane is False


def test_the_flag_is_cleared_even_when_the_lane_raises() -> None:
    """`_scalp_lane_took_it` catches everything the lane throws. The flag must
    come down on that path too, or one bad symbol disables the lane for good.
    """
    service = _runner()

    def explode(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("the broker dropped the connection")

    service._run_scalp_lane = explode  # type: ignore[assignment]

    assert service._scalp_lane_took_it("c", "XAUUSD", ["s"], "NO_SIGNAL", None) is False
    assert service._inside_scalp_lane is False


def test_a_half_built_runner_is_safe_before_init_has_run() -> None:
    """Every fixture in this suite builds the runner with `object.__new__`, and
    so does the recovery path in production. The flag is class-level for that
    reason."""
    bare = object.__new__(JarvisRunner)

    assert bare._inside_scalp_lane is False
