"""A big stake has to be corroborated, and for a month it was not.

The conviction ladder sizes on `min(engine_confidence, advice.confidence)`, and
the docstring on that line says exactly why:

    "engine conviction bought 5.74% of equity while the final adviser only
     reported 0.47 confidence. Taking the minimum preserves every approved
     trade and changes only its size."

It was a no-op. `local_history` returns `idea.confidence` -- the ENGINE'S own
number -- whenever it lacks enough comparable setups to form a view. That needs
five neighbours against a journal holding a few dozen trades, so it is the
normal path and not the edge case, and `min(x, x)` is `x`.

The second opinion that was supposed to restrain the stake was the first
opinion wearing a hat. One detector's reading could put 12% of the account on a
trade -- EUR 21 of EUR 176 -- with nothing agreeing with it.

Found by asking a mechanical question of the whole config: which settings does
nothing read? Six of the eight answers were harmless. This one was not.
"""

from __future__ import annotations

from advisory.providers import Advice


def advice(**overrides):  # type: ignore[no-untyped-def]
    base = {
        "approved": True,
        "confidence": 0.75,
        "thesis": "fine",
        "provider": "local_history",
        "said_yes": True,
    }
    base.update(overrides)
    return Advice(**base)  # type: ignore[arg-type]


class TestTheArchiveSaysWhenItHasNoView:
    def test_too_few_neighbours_is_flagged_as_not_independent(self) -> None:
        """The number it returns there is the engine's own, and a caller
        sizing on corroboration has to be able to tell."""
        import inspect

        from advisory import local_history

        source = inspect.getsource(local_history)

        # Both no-opinion paths carry the flag.
        assert source.count("independent=False") == 2

    def test_advice_defaults_to_independent(self) -> None:
        """A real adviser that never heard of this flag must keep counting as
        a second opinion, or adding the field would silently halve every
        stake."""
        assert advice().independent is True


class TestTheStakeFollowsIt:
    def _sizer(self, verdict):  # type: ignore[no-untyped-def]
        from types import SimpleNamespace

        from analysis.confluence import TradeIdea
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from core.types import Direction, Signal
        from runner.service import JarvisRunner

        service = object.__new__(JarvisRunner)
        service.settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        service.risk = SimpleNamespace(  # type: ignore[attr-defined]
            room_for_more_risk=lambda state, wanted, spec: 100.0
        )
        service.broker = SimpleNamespace(spec=lambda symbol: None)  # type: ignore[attr-defined]
        idea = TradeIdea(
            symbol="EURUSD",
            approved=True,
            direction=Direction.LONG,
            score=60.0,
            confidence=0.75,
            entry=1.08,
            stop_loss=1.078,
            take_profit=1.085,
            reason="test",
            signals=(Signal("impulse_break", 60.0, 0.75),),
        )
        return service, idea, verdict

    def test_without_a_second_opinion_it_takes_the_ordinary_stake(self) -> None:
        service, idea, verdict = self._sizer(advice(independent=False))

        stake, why = service._conviction_stake(None, idea, verdict)

        assert stake == service.settings.risk.conviction_risk.floor_pct
        assert "no independent second opinion" in why

    def test_with_one_it_may_climb_the_ladder(self) -> None:
        service, idea, verdict = self._sizer(advice(independent=True))

        stake, _ = service._conviction_stake(None, idea, verdict)

        assert stake > service.settings.risk.conviction_risk.floor_pct

    def test_an_independent_adviser_that_is_unsure_still_caps_it(self) -> None:
        """The behaviour the docstring always described, now that it can
        actually happen: a real second opinion at 0.47 sizes the trade down
        rather than leaving the engine's 0.75 unchallenged."""
        service, idea, verdict = self._sizer(advice(independent=True, confidence=0.47))

        capped, _ = service._conviction_stake(None, idea, verdict)
        free, _ = service._conviction_stake(None, idea, advice(independent=True))

        assert capped < free
