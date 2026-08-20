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


class TestThePerDetectorFloor:
    """One floor for eight detectors was letting both the best and the worst
    through together, or holding both back. Dropping the global value from 0.65
    to 0.55 more than doubles setups (71 -> 153 over identical bars), and of the
    ~1,174 refusals it releases, 473 are `fast_ema_cross` and 414 are
    `liquidity_sweep` — the worst and the best records in the module backtest.
    """

    def test_an_unlisted_detector_still_takes_the_global_value(self) -> None:
        """An entry is a claim about ONE detector, and it must not leak.

        The table now carries `session_breakout` at 1.00 to hold it back. Every
        detector that has not earned an entry — in either direction — must still
        be judged by the single global floor.
        """
        confluence = settings().analysis.confluence

        assert "liquidity_sweep" not in confluence.lone_module_minimum_confidence_by_module
        assert confluence.lone_floor_for("liquidity_sweep") == (
            confluence.lone_module_minimum_confidence
        )

    def test_an_override_applies_to_that_detector_alone(self) -> None:
        changed = apply_overrides(
            settings(),
            ["lone_module_minimum_confidence_by_module=liquidity_sweep:0.55"],
        ).analysis.confluence

        assert changed.lone_floor_for("liquidity_sweep") == 0.55
        assert changed.lone_floor_for("fast_ema_cross") == 0.65

    def test_several_detectors_can_be_set_at_once(self) -> None:
        changed = apply_overrides(
            settings(),
            ["lone_module_minimum_confidence_by_module=liquidity_sweep:0.55,impulse_break:0.75"],
        ).analysis.confluence

        assert changed.lone_floor_for("liquidity_sweep") == 0.55
        assert changed.lone_floor_for("impulse_break") == 0.75
        assert changed.lone_floor_for("trend_momentum") == 0.65

    def test_a_misspelled_detector_stops_the_run(self) -> None:
        """The failure this whole tool exists to avoid: measuring a change that
        was never applied and reporting "no difference"."""
        with pytest.raises(SystemExit):
            apply_overrides(
                settings(), ["lone_module_minimum_confidence_by_module=liquidty_sweep:0.55"]
            )

    def test_a_malformed_entry_stops_the_run(self) -> None:
        with pytest.raises(SystemExit):
            apply_overrides(
                settings(), ["lone_module_minimum_confidence_by_module=liquidity_sweep"]
            )


class TestAListOfDetectorNamesCanBeOverridden:
    """`trend_continuation_modules` is a tuple, and a tuple had no branch.

    It fell through to the catch-all and set the field to a plain STRING —
    which pydantic either coerces to a tuple of single characters or rejects,
    so the run would have measured neither the current rule nor the proposed
    one while printing a difference. That is precisely the failure
    `apply_overrides` documents itself as existing to prevent.

    The question it blocks is a real one. The live overlay lists only
    `trend_momentum` as a continuation module while the config's own comments
    claim `drift_continuation` and `fast_ema_cross` are on it too, and
    `trend_momentum` carries no live weight — so the range/transition guard
    protects against nothing on this account. What adding the other two would
    cost in setups is measurable, and this is what measures it.
    """

    def test_a_comma_separated_list_becomes_a_tuple(self) -> None:
        changed = apply_overrides(
            settings(),
            ["trend_continuation_modules=trend_momentum,drift_continuation,fast_ema_cross"],
        ).analysis.confluence

        assert changed.trend_continuation_modules == (
            "trend_momentum",
            "drift_continuation",
            "fast_ema_cross",
        )

    def test_a_single_name_still_works(self) -> None:
        changed = apply_overrides(
            settings(), ["trend_continuation_modules=drift_continuation"]
        ).analysis.confluence

        assert changed.trend_continuation_modules == ("drift_continuation",)

    def test_a_misspelt_detector_stops_the_run(self) -> None:
        """A typo would silently measure nothing and report "no difference",
        which is worse than an error because it looks like an answer."""
        with pytest.raises(SystemExit, match="unknown detector"):
            apply_overrides(settings(), ["trend_continuation_modules=drift_contnuation"])

    def test_an_untypeable_field_is_refused_rather_than_guessed(self) -> None:
        """The catch-all used to pass the raw string through for ANY unhandled
        type. Refusing is the only honest answer: a run that sets a field to
        something the operator did not ask for measures neither version."""
        confluence = settings().analysis.confluence
        exotic = next(
            (
                name
                for name, value in confluence.model_dump().items()
                if not isinstance(value, (bool, int, float, str, dict, tuple, list))
                and value is not None
            ),
            "",
        )
        if not exotic:
            pytest.skip("no field of an unhandled type on this config")
        with pytest.raises(SystemExit, match="cannot yet type"):
            apply_overrides(settings(), [f"{exotic}=whatever"])
