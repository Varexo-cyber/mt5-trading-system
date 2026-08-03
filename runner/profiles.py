"""Named presets for what to trade and how many at once.

Three settings decide how long money stays committed, and they only make sense
together: which markets are looked at, how many positions may be open, and how
many trades a day are allowed. Setting them one at a time invites incoherent
combinations — one slot with a six-day instrument means one trade a week.

The numbers behind the choice, measured rather than assumed. Stop and target are
1.5 ATR and twice that, so in *market hours* every asset class reaches its
target in about the same time, roughly a day and a half. What differs is how
many hours a week the market is open to travel it:

    FX, metals, indices    120 open hours/week  ->  ~2 calendar days
    LSE / Milan shares      42 open hours/week  ->  ~6 calendar days

So a share is not badly sized, it is slow, and slot turnover is the cost. With
two positions a six-day trade holds half the account's capacity for a week.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.schema import Settings


@dataclass(frozen=True, slots=True)
class TradingProfile:
    name: str
    description: str
    asset_classes: tuple[str, ...]
    max_positions: int
    symbols_only: tuple[str, ...] = ()


#: `fast` exists because concentration is the point of it: one position at a
#: time in continuously traded markets turns the slot over in about two days
#: instead of six, and forces the reviewer to hold out for the best single
#: candidate rather than filling two slots with whatever passed first.
PROFILES: dict[str, TradingProfile] = {
    "fast": TradingProfile(
        name="fast",
        description="Continuously traded markets only, one position at a time",
        asset_classes=("forex", "metal", "index"),
        max_positions=1,
    ),
    "gold": TradingProfile(
        name="gold",
        description="XAUUSD only, one position — nothing else is even looked at",
        asset_classes=("metal",),
        max_positions=1,
        symbols_only=("XAUUSD",),
    ),
    "broad": TradingProfile(
        name="broad",
        description="Every asset class, two positions — the widest search",
        asset_classes=(),
        max_positions=2,
    ),
}


def apply_profile(settings: Settings, name: str) -> Settings:
    """Return `settings` with the named profile's choices applied.

    Only narrows. A profile cannot raise the position limit above what the
    active mode already allows, so selecting one can never increase the risk
    the configuration was validated with — `min` rather than assignment.
    """
    profile = PROFILES.get(name)
    if profile is None:
        raise ValueError(f"unknown profile {name!r}; known: {sorted(PROFILES)}")

    instruments = settings.instruments.model_copy(
        update={
            "asset_classes": profile.asset_classes,
            "symbols_only": profile.symbols_only,
        }
    )
    risk = settings.risk.model_copy(
        update={
            "max_concurrent_positions": min(
                settings.risk.max_concurrent_positions, profile.max_positions
            )
        }
    )
    modes = dict(settings.modes)
    active = settings.mode.value
    modes[active] = modes[active].model_copy(
        update={
            "max_concurrent_positions": min(
                modes[active].max_concurrent_positions, profile.max_positions
            )
        }
    )
    return settings.model_copy(update={"instruments": instruments, "risk": risk, "modes": modes})
