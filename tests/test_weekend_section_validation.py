from scripts.validate_weekend_sections import Verdict, passed


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
