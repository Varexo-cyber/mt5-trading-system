"""The A/B tool has to be trustworthy before its output is allowed to decide anything.

Its whole purpose is to answer "is this rule too strict" without putting money
behind the answer. Two things would quietly make it lie: a refusal grouping that
produces one bucket per decision and therefore no finding, and a `--set` that
turns `false` into the string "false" and silently measures the wrong config.
Both are pinned here.
"""

from __future__ import annotations

import pytest

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from scripts.why_no_setups import apply_overrides, shape_of


def settings():  # type: ignore[no-untyped-def]
    return load_settings(DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False)


class TestRefusalsGroupByShapeAndNotByNumber:
    def test_two_readings_of_one_rule_are_one_bucket(self) -> None:
        """Live refusal texts carry the number that caused them, so grouping on
        the raw string gives one bucket per decision and no finding at all."""
        first = "fast_ema_cross is the only detector pointing this way and reads 0.57"
        second = "fast_ema_cross is the only detector pointing this way and reads 0.61"

        assert shape_of(first) == shape_of(second)

    def test_two_different_rules_stay_apart(self) -> None:
        lone = "impulse_break is the only detector pointing this way and reads 0.50"
        target = "no target between 1.00R and 3.00R pays on this market"

        assert shape_of(lone) != shape_of(target)

    def test_percentages_collapse_too(self) -> None:
        assert shape_of("reached first 22% of the time") == shape_of(
            "reached first 41% of the time"
        )

    def test_a_long_refusal_is_truncated_rather_than_dropped(self) -> None:
        assert shape_of("x" * 400) == "x" * 110


class TestTheOverrideMeasuresWhatItSays:
    def test_a_float_field_is_a_float(self) -> None:
        changed = apply_overrides(settings(), ["lone_module_minimum_confidence=0.55"])

        assert changed.analysis.confluence.lone_module_minimum_confidence == 0.55

    def test_a_boolean_field_does_not_become_a_truthy_string(self) -> None:
        """`"false"` is truthy. Without the type read off the model this would
        have turned a gate ON while reporting that it was measuring it off, and
        nothing in the output would have shown it."""
        changed = apply_overrides(settings(), ["require_direction_advantage=false"])

        assert changed.analysis.confluence.require_direction_advantage is False

    def test_an_integer_field_stays_an_integer(self) -> None:
        changed = apply_overrides(settings(), ["target_horizon_bars=48"])

        assert changed.analysis.confluence.target_horizon_bars == 48

    def test_the_live_settings_are_not_mutated(self) -> None:
        """Both runs read the same bars; if the first run's config were mutated
        in place the comparison would be a variant against itself."""
        live = settings()
        before = live.analysis.confluence.lone_module_minimum_confidence

        apply_overrides(live, ["lone_module_minimum_confidence=0.10"])

        assert live.analysis.confluence.lone_module_minimum_confidence == before

    def test_several_overrides_apply_together(self) -> None:
        changed = apply_overrides(
            settings(),
            ["lone_module_minimum_confidence=0.55", "minimum_r_multiple=0.8"],
        )

        assert changed.analysis.confluence.lone_module_minimum_confidence == 0.55
        assert changed.analysis.confluence.minimum_r_multiple == 0.8

    def test_a_field_that_does_not_exist_stops_the_run(self) -> None:
        """Silently ignoring a typo would report "no difference" for a change
        that was never applied — the worst possible output from this tool."""
        with pytest.raises(SystemExit):
            apply_overrides(settings(), ["lone_module_minimum_confidenc=0.55"])

    def test_a_malformed_pair_stops_the_run(self) -> None:
        with pytest.raises(SystemExit):
            apply_overrides(settings(), ["lone_module_minimum_confidence"])
