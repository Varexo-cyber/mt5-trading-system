"""Profiles narrow what is traded; they must never widen what is risked."""

from __future__ import annotations

import pytest

from config.loader import load_settings
from runner.profiles import PROFILES, apply_profile


def test_fast_holds_one_position_in_continuously_traded_markets() -> None:
    """One slot turns over in about two days instead of six.

    Stop and target are 1.5 ATR and twice that, so in market hours every class
    reaches its target in roughly the same time. A share is open 42 hours a week
    against FX's 120, so the same setup takes about six calendar days — and a
    six-day trade in one of two slots is half the account held for a week.
    """
    settings = apply_profile(load_settings(env_overrides=False), "fast")

    assert settings.instruments.asset_classes == ("forex", "metal", "index")
    assert settings.effective_max_positions() == 1


def test_gold_looks_at_nothing_else() -> None:
    settings = apply_profile(load_settings(env_overrides=False), "gold")

    assert settings.instruments.symbols_only == ("XAUUSD",)
    assert settings.effective_max_positions() == 1


def test_a_profile_can_only_narrow_the_position_limit() -> None:
    """Selecting one must never raise risk above the validated configuration."""
    base = load_settings(env_overrides=False)
    broad = PROFILES["broad"]
    assert broad.max_positions >= base.effective_max_positions()

    settings = apply_profile(base, "broad")

    assert settings.effective_max_positions() == base.effective_max_positions()


def test_an_unknown_profile_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        apply_profile(load_settings(env_overrides=False), "moonshot")
