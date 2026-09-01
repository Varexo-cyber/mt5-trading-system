"""A live ticket must say which measured strategy instance created it."""

from core.trade_origin import broker_comment, origin_for_setup_family


def test_every_live_setup_family_has_an_unambiguous_mt5_label() -> None:
    expected = {
        "section_five_m5": (5, "M5", "JARVIS-S5-NL-M5"),
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
