"""A live ticket must say which measured strategy instance created it."""

from core.trade_origin import broker_comment, origin_for_setup_family


def test_every_live_setup_family_has_an_unambiguous_mt5_label() -> None:
    expected = {
        "section_ten_gold_m1": (10, "M1", "JARVIS-S10-AU-M1"),
        "section_eight_trend_day_h1": (8, "H1", "JARVIS-S8-TD-H1"),
        "section_nine_vwap_m30": (9, "M30", "JARVIS-S9-VW-M30"),
        "section_five_m5": (5, "M5", "JARVIS-S5-NL-M5"),
        "section_six_gold_m5": (6, "M5", "JARVIS-S6-AU-M5"),
        "section_six_spx_h1": (6, "H1", "JARVIS-S6-SP-H1"),
        "section_eleven_metals_m5": (11, "M5", "JARVIS-S11-XAU-M5"),
        "failed_session_breakout_m5": (7, "M5", "JARVIS-S7-FSB-M5"),
        "impulse_retest_m15": (2, "M15", "JARVIS-S2-IR-M15"),
        "impulse_retest_m30_m30": (2, "M30", "JARVIS-S2-IR-M30"),
        "order_block_fast_m1": (3, "M1", "JARVIS-S3-OB-M1"),
        "order_block_m15_m15": (3, "M15", "JARVIS-S3-OB-M15"),
        "order_block_m30": (3, "M30", "JARVIS-S3-OB-M30"),
        "order_block_h1_h1": (3, "H1", "JARVIS-S3-OB-H1"),
    }

    for family, (section, timeframe, comment) in expected.items():
        origin = origin_for_setup_family(family)
        assert origin is not None
        assert (origin.section, origin.timeframe, origin.comment) == (
            section,
            timeframe,
            comment,
        )
        assert broker_comment(family, is_addon=False, experimental_live=True) == comment
        assert len(comment) <= 31


def test_unrelated_and_addon_comments_keep_their_existing_identity() -> None:
    assert (
        broker_comment("trend_momentum_swing", is_addon=False, experimental_live=True)
        == "jarvis-exp-live"
    )
    assert (
        broker_comment("trend_momentum_swing", is_addon=False, experimental_live=False) == "jarvis"
    )
    assert (
        broker_comment("order_block_fast_m1", is_addon=True, experimental_live=True)
        == "jarvis-scalp"
    )


def test_every_live_section_carries_its_own_broker_label() -> None:
    """SECTION ELEVEN WENT LIVE WITHOUT ONE, and nothing anywhere complained.

    A section with no entry in `_ORIGINS` gets the generic `jarvis` comment,
    and three things then fail at once and all of them quietly:

      - MT5 mobile shows `jarvis`, so the owner cannot tell one section's
        ticket from another's on his phone.
      - The journal cannot attribute the trade, so its per-section report has
        no row for that section -- an absent row reading as a zero row, which
        is the confusion this project keeps shipping.
      - `section_of_comment` returns "", so the shared-symbol rule sees an
        UNIDENTIFIED holder and refuses to let any other section join that
        symbol. The rule the owner asked for on 4 September switches itself
        back off.

    This asserts the property for every module on the live allowlist, so the
    next promotion cannot reach the VPS in that state.
    """
    from config.loader import DEFAULT_CONFIG_PATH, load_settings
    from core.trade_origin import section_of_comment

    settings = load_settings(
        DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
    )

    for module in settings.analysis.confluence.live_enabled_modules:
        origin = origin_for_setup_family(module)
        assert origin is not None, f"{module} is live and has no broker label"
        assert len(origin.comment) <= 31, origin.comment
        # And the label has to survive the round trip the risk manager makes:
        # module name -> comment -> section. A label nothing can read back is
        # the same silence one layer down.
        assert section_of_comment(origin.comment) == origin.comment, module
