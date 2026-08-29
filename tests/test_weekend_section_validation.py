import json
from datetime import UTC, datetime
from pathlib import Path

from config.loader import load_settings
from scripts.validate_weekend_sections import (
    STRATEGY_NAMES,
    Verdict,
    isolated_engine,
    passed,
    save_checkpoint,
)


def verdict(**changes: object) -> Verdict:
    values = {
        "section": "1",
        "strategy": "example",
        "segment": "validation",
        "setups": 150,
        "trades": 120,
        "total_r": 12.0,
        "expectancy_r": 0.10,
        "win_rate": 0.55,
        "profit_factor": 1.25,
        "max_drawdown_r": 5.0,
        "dsr": 0.90,
    }
    values.update(changes)
    return Verdict(**values)  # type: ignore[arg-type]


def test_promotion_needs_sample_edge_payoff_and_significance() -> None:
    assert passed(verdict())
    assert not passed(verdict(trades=99))
    assert not passed(verdict(expectancy_r=0.049))
    assert not passed(verdict(profit_factor=1.09))
    assert not passed(verdict(profit_factor=0.0))
    assert not passed(verdict(dsr=0.79))


def test_many_losing_trades_never_pass_the_volume_gate() -> None:
    assert not passed(
        verdict(
            setups=100_000,
            trades=50_000,
            total_r=-500.0,
            expectancy_r=-0.01,
            profit_factor=0.98,
            dsr=0.99,
        )
    )


def test_isolated_engine_executes_only_the_vote_and_regime_readers() -> None:
    settings = load_settings(overlay=Path("config/eightcap.yaml"))
    engine = isolated_engine(settings, "m1_micro_breakout")

    assert {reader.name for reader in engine.modules} == {
        "m1_micro_breakout",
        "market_regime",
        "volatility_regime",
    }


def test_checkpoint_keeps_completed_strategy_rows(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    now = datetime.now(UTC)
    row = verdict()

    save_checkpoint(path, signature={"run": 1}, start=now, end=now, rows=[row])

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["signature"] == {"run": 1}
    assert saved["rows"][0]["strategy"] == "example"
    assert not path.with_suffix(".tmp").exists()


def test_every_weekend_strategy_can_be_selected_individually() -> None:
    assert STRATEGY_NAMES == (
        "market_structure",
        "trend_momentum",
        "m1_micro_breakout",
        "basket_divergence",
        "candle_momentum",
        "vwap_reversion",
        "own_lane",
    )
