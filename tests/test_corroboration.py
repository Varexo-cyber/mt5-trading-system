"""The engine is called confluence and its score punished confluence.

The score is a weighted MEAN over the agreeing modules, so a second reader that
AGREES could only pull the average toward its own value. Against the live
threshold of 35:

    market_structure alone                        70.0
    market_structure + candle_momentum agreeing    51.9

Two readers pointing the same way scored eighteen points LOWER than one reader
alone. Corroboration was a penalty and it grew with how much of it there was.

WHAT THAT SELECTED FOR is the damage, not the undervaluation. At a fixed bar,
an engine whose score falls as agreement rises systematically prefers the
setups where exactly ONE detector is loud and every other is silent -- the
least corroborated readings available.

And that is the finding that dominated a day of measurement without being
explained: all eight detectors came back at 54-57% win, average win +0.68R
against average loss -1.04R. One shape, eight times. They did not resemble each
other because they saw the same thing; they resembled each other because the
engine picked the same KIND of setup out of each: the lonely one.

It was written down and the conclusion was never drawn. `candle_momentum`'s
docstring says "joining a strong reader makes matters worse rather than
better", and the response at the time was to give that module its own lane
AROUND the vote rather than to ask why agreement was being taxed.
"""

from __future__ import annotations

import pytest

from analysis.confluence import ConfluenceEngine
from core.types import Signal


def _score(engine, pairs) -> float:  # type: ignore[no-untyped-def]
    """The engine's OWN scoring method, not a restatement of it.

    This helper used to carry its own copy of the arithmetic, which is the
    exact mistake the code had: two definitions, one repaired. A test that
    reimplements what it checks passes against a broken engine.
    """
    return engine.score_of(list(pairs))


def _engine(**changes):  # type: ignore[no-untyped-def]
    from config.loader import DEFAULT_CONFIG_PATH, load_settings

    config = load_settings(
        DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
    ).analysis.confluence
    engine = ConfluenceEngine.__new__(ConfluenceEngine)
    engine.config = config.model_copy(update=changes) if changes else config
    return engine


STRUCTURE = (Signal("market_structure", 70.0, 1.0, reasoning="BOS"), 1.0)
MOMENTUM = (Signal("candle_momentum", 45.0, 0.75, reasoning="a decisive minute"), 1.0)


class TestAgreementCanNeverMakeASetupLookWorse:
    def test_a_second_agreeing_reader_no_longer_lowers_the_score(self) -> None:
        """The defect, stated as the property that must hold. Under the plain
        mean this was 70.0 against 51.9."""
        engine = _engine()

        alone = _score(engine, [STRUCTURE])
        corroborated = _score(engine, [STRUCTURE, MOMENTUM])

        assert corroborated > alone

    def test_zero_removes_the_premium_but_does_not_bring_the_mean_back(self) -> None:
        """STATED PLAINLY BECAUSE IT IS A REAL BEHAVIOUR CHANGE, not a
        backward-compatible switch.

        A premium on top of the mean was the first attempt and it failed its
        own test: at 15% a corroborated pair came out at 59.7 against 70.0
        alone, so the penalty survived the repair and merely shrank. The mean
        IS the defect, so the base is now the strongest agreeing reading and
        that is not configurable. Zero switches off the premium only.
        """
        off = _engine(corroboration_bonus_per_module=0.0, max_corroboration_bonus=0.0)

        # Not 51.875, which is what the mean gave this pair.
        assert _score(off, [STRUCTURE, MOMENTUM]) == pytest.approx(70.0)
        assert _score(off, [STRUCTURE]) == pytest.approx(70.0)

    def test_a_lone_reader_scores_exactly_what_it_always_scored(self) -> None:
        """The scale is preserved on purpose: the threshold has to keep
        meaning what it meant, or every other number calibrated against it
        moves at the same time and nothing can be attributed."""
        engine = _engine()

        assert _score(engine, [STRUCTURE]) == pytest.approx(70.0)


class TestThePremiumIsBounded:
    def test_it_stops_growing_so_a_crowd_cannot_outvote_quality(self) -> None:
        """Five weak modules nodding along must not outscore one strong,
        well-corroborated pair. Without a cap, agreement alone would become the
        strategy."""
        engine = _engine()
        weak = [(Signal(f"m{i}", 36.0, 0.5, reasoning="thin"), 1.0) for i in range(6)]

        crowd = _score(engine, weak)
        pair = _score(engine, [STRUCTURE, MOMENTUM])

        assert crowd < pair

    def test_the_premium_grows_with_the_strength_of_the_agreement(self) -> None:
        """Not with the COUNT of it. A module scraping past its floor must not
        buy the same premium as a second strong reader, or "find anything that
        agrees" becomes the strategy."""
        engine = _engine()
        token = (Signal("thin", 4.0, 0.5, reasoning="barely there"), 1.0)

        assert _score(engine, [STRUCTURE, MOMENTUM]) > _score(engine, [STRUCTURE, token])
        # And a token nod still cannot make the setup look worse than alone.
        assert _score(engine, [STRUCTURE, token]) >= _score(engine, [STRUCTURE])

    def test_the_cap_holds_once_enough_agreement_has_accumulated(self) -> None:
        engine = _engine()
        seven = [STRUCTURE, *[MOMENTUM] * 7]
        eight = [STRUCTURE, *[MOMENTUM] * 8]

        assert _score(engine, seven) == pytest.approx(70.0 * 1.45)
        assert _score(engine, eight) == pytest.approx(_score(engine, seven))


class TestTheShippedSettings:
    def test_the_premium_is_on_and_the_scale_is_unchanged(self) -> None:
        config = _engine().config

        assert config.corroboration_bonus_per_module == 0.15
        assert config.max_corroboration_bonus == 0.45
        # The threshold did not move with it. It could not: the whole point of
        # keeping the mean is that a one-module reading still faces the bar it
        # always faced, so nothing that trades today stops trading.
        assert config.score_threshold == 35.0

    def test_nothing_that_trades_today_stops_trading(self) -> None:
        """The premium is a multiplier at or above 1.0, so every setup that
        cleared the bar still clears it. This change can only ADD the
        corroborated setups that were being pushed under the bar by the fact
        that there was more evidence for them."""
        engine = _engine()
        off = _engine(corroboration_bonus_per_module=0.0, max_corroboration_bonus=0.0)

        for pairs in ([STRUCTURE], [STRUCTURE, MOMENTUM], [STRUCTURE, MOMENTUM, MOMENTUM]):
            assert _score(engine, pairs) >= _score(off, pairs)


class TestTheArithmeticExistsExactlyOnce:
    """It existed twice and only one copy was repaired.

    `evaluate` computed the final score and `readiness` computed a score of its
    own to decide which horizon owns the proposal -- quick, intraday or swing.
    Both were the same weighted mean. Fixing the first left the second still
    selecting on the defect: at a fixed bar, a score that falls as agreement
    rises prefers the group where one detector is loud and the rest are silent.

    So the corroborated quick group kept losing the horizon contest to the
    lonely swing one, after the final score had stopped punishing corroboration.
    """

    def test_no_second_copy_of_the_score_survives(self) -> None:
        """Asserted over the source, because the failure is invisible in
        behaviour: both copies returned plausible numbers and only one was
        right.

        The mean's signature is that exact numerator divided by the summed
        weights OVER THE AGREEING MODULES. The numerator alone is not enough to
        match on: `_vote` uses the same term to choose a DIRECTION by comparing
        the two sides, which is a different quantity that happens to share an
        expression. Matching it would have made this test fail on correct code,
        and the next person would have deleted the test rather than the copy.
        """
        import inspect

        from analysis import confluence

        source = " ".join(inspect.getsource(confluence).split())
        mean_over_agreeing = (
            "sum(abs(signal.score) * signal.confidence * weight "
            "for signal, weight in agreeing) / denominator"
        )

        assert mean_over_agreeing not in source, "a second copy of the score is back"

    def test_both_deciders_go_through_the_same_method(self) -> None:
        """`evaluate` produces the final score; `readiness` inside
        `_resolve_direction` decides which horizon owns the symbol. Those are
        the two that were separate."""
        import inspect

        from analysis.confluence import ConfluenceEngine

        assert "self.score_of(agreeing)" in inspect.getsource(ConfluenceEngine.evaluate)
        assert "self.score_of(agreeing)" in inspect.getsource(ConfluenceEngine._resolve_direction)
