"""Event-driven AI management reacts to evidence, not polling noise."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from analysis.position_health import PositionHealth
from core.types import Direction, Position, Tick
from runner.service import JarvisRunner, _SupervisionSnapshot

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)


class _Journal:
    def open_trade_by_ticket(self, _ticket: int) -> dict[str, float]:
        return {"sl": 90.0, "mfe_r": 0.2}


class _Broker:
    bid = 101.0

    def tick(self, _symbol: str) -> Tick:
        return Tick("TEST", NOW, self.bid, self.bid + 0.1)


def _runner() -> JarvisRunner:
    runner = object.__new__(JarvisRunner)
    runner.journal = _Journal()  # type: ignore[assignment]
    runner.broker = _Broker()  # type: ignore[assignment]
    # `equity` alongside `last_health`: the supervision trigger reads it to
    # express profit as a share of the account, which is the rung the R
    # ladder cannot see on a wide-stopped trade. Zero means "no equity
    # known", which switches that trigger off and leaves the R ladder as the
    # only one — the behaviour these tests were written against.
    runner.manager = SimpleNamespace(last_health={}, equity=0.0)  # type: ignore[assignment]
    runner.settings = SimpleNamespace(  # type: ignore[assignment]
        trade_management=SimpleNamespace(
            supervision_interval_minutes=15.0,
            supervision_event_driven=True,
            supervision_min_interval_minutes=2.0,
            supervision_profit_step_r=0.25,
            supervision_profit_step_equity_pct=0.5,
            supervision_giveback_trigger_fraction=0.25,
            giveback_arm_r=0.5,
            # The losing-side ladders, at their schema defaults. `equity` is
            # zero above, so the cash route reads nothing, and the R route sees
            # only the positive readings these tests were written with. Neither
            # changes an existing expectation.
            supervision_loss_step_r=0.25,
            supervision_loss_step_equity_pct=0.35,
        )
    )
    runner._supervised_at = {}
    runner._supervision_due_at = {}
    runner._supervision_snapshots = {}
    # No scan has run in these fixtures, so there is no fresh directional
    # read to compare against and the engine trigger stays silent — which
    # is the behaviour every test below was written against.
    runner._latest_direction = {}
    return runner


def _position() -> Position:
    return Position(
        ticket=1,
        symbol="TEST",
        direction=Direction.LONG,
        volume=0.01,
        price_open=100.0,
        sl=95.0,
        tp=120.0,
        profit=1.0,
        swap=0.0,
        opened_at=NOW - timedelta(hours=1),
    )


def test_a_new_position_is_reviewed_immediately() -> None:
    triggered = _runner()._supervision_trigger(_position(), NOW)

    assert triggered is not None
    assert triggered[0] == "position_opened"


def test_worsening_health_brings_the_review_forward() -> None:
    runner = _runner()
    runner._supervised_at[1] = NOW - timedelta(minutes=3)
    runner._supervision_due_at[1] = NOW + timedelta(minutes=12)
    runner._supervision_snapshots[1] = _SupervisionSnapshot(0.1, 0.2, 0.5, "healthy", 0.0)
    runner.manager.last_health[1] = PositionHealth(
        "deteriorating", 0.6, "tighten", (), "structure weakened"
    )

    triggered = runner._supervision_trigger(_position(), NOW)

    assert triggered is not None
    assert triggered[0] == "health_worsened:healthy->deteriorating"


def test_the_cost_cooldown_blocks_repeated_reconsideration() -> None:
    runner = _runner()
    runner._supervised_at[1] = NOW - timedelta(seconds=30)
    runner._supervision_due_at[1] = NOW + timedelta(minutes=14)
    runner._supervision_snapshots[1] = _SupervisionSnapshot(0.1, 0.2, 0.5, "healthy", 0.0)
    runner.manager.last_health[1] = PositionHealth(
        "broken", 1.0, "exit", (), "structure invalidated"
    )

    assert runner._supervision_trigger(_position(), NOW) is None


class TestMoneyMovesEvenWhenRDoesNot:
    """The live CADCHF long, and why nothing asked the reviewer about it.

    Entry 0.58542, stop 0.58422, price 0.58595: EUR 2.82 on a EUR 130 account,
    over two percent of everything, and only 0.44R because the stop was twelve
    pips wide. Every supervision trigger was written in R, so the 0.25R ladder
    had last spoken at EUR 1.60 and the next rung was EUR 3.20. The money moved
    and nobody was asked.
    """

    @staticmethod
    def _snapshot(r_now: float, pct: float):  # type: ignore[no-untyped-def]
        from runner.service import _SupervisionSnapshot

        return _SupervisionSnapshot(
            r_now=r_now,
            peak_r=r_now,
            giveback_fraction=0.0,
            health_verdict="healthy",
            health_severity=0.0,
            profit_pct_of_equity=pct,
        )

    def test_the_cash_ladder_fires_where_the_r_ladder_is_silent(self) -> None:
        """0.25R to 0.44R crosses no R rung; 1.2% to 2.2% crosses two cash ones."""
        from config.schema import TradeManagementConfig

        config = TradeManagementConfig()
        before = self._snapshot(0.25, 1.2)
        after = self._snapshot(0.44, 2.2)

        r_step = config.supervision_profit_step_r
        assert int(after.r_now / r_step) == int(before.r_now / r_step), "the R ladder is silent"

        cash_step = config.supervision_profit_step_equity_pct
        assert int(after.profit_pct_of_equity / cash_step) > int(
            before.profit_pct_of_equity / cash_step
        )

    def test_it_can_be_switched_off(self) -> None:
        from config.schema import TradeManagementConfig

        assert TradeManagementConfig(supervision_profit_step_equity_pct=0.0)


class TestTheReviewerIsToldWhetherTheProfitIsSafe:
    """It had every number needed to work this out and was never told it."""

    @staticmethod
    def _payload(stop: float):  # type: ignore[no-untyped-def]
        from datetime import UTC, datetime

        import pandas as pd

        from advisory.providers import build_supervision_payload
        from core.types import Direction, MarketContext, Position, Series, Tick, Timeframe

        now = datetime(2026, 8, 14, 19, 40, tzinfo=UTC)
        index = pd.date_range("2026-08-14", periods=60, freq="15min", tz=UTC)
        close = pd.Series([0.5850 + i * 0.00002 for i in range(60)], index=index)
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.0002,
                "low": close - 0.0002,
                "close": close,
                "tick_volume": 100,
                "spread": 10,
                "real_volume": 0,
            },
            index=index,
        )
        context = MarketContext(
            symbol="CADCHF.i",
            now=now,
            series={Timeframe.M15: Series("CADCHF.i", Timeframe.M15, frame, now)},
            tick=Tick("CADCHF.i", now, 0.58595, 0.58600),
        )
        position = Position(
            ticket=134663779,
            symbol="CADCHF.i",
            direction=Direction.LONG,
            volume=0.05,
            price_open=0.58542,
            sl=stop,
            tp=0.58691,
            profit=2.82,
            swap=0.0,
            opened_at=now,
        )
        return build_supervision_payload(position, context, {"account_equity": 130.0})

    def test_a_stop_below_entry_is_reported_as_unprotected(self) -> None:
        """The live shape: 2.2% of the account showing, none of it safe."""
        payload = self._payload(stop=0.58422)

        assert payload["profit_is_protected"] is False
        assert payload["unrealised_pct_of_account"] == pytest.approx(2.17, abs=0.05)
        assert payload["stop_distance_from_entry_in_r"] == pytest.approx(-1.0, abs=0.01)

    def test_a_stop_above_entry_is_reported_as_protected(self) -> None:
        payload = self._payload(stop=0.58580)

        assert payload["profit_is_protected"] is True
        assert payload["stop_distance_from_entry_in_r"] > 0

    def test_no_stop_at_all_is_never_called_protected(self) -> None:
        """Fails closed. A missing stop is the least protected state there is."""
        payload = self._payload(stop=0.0)

        assert payload["profit_is_protected"] is False


class TestAPeakThatDrainsWakesSomebody:
    """One euro, then ninety cents, then eighty. Who is watching?

    The operator asked it as a yes-or-no and the honest answer was no. On a
    EUR 133 account with 1R around EUR 2.35, a EUR 1.00 peak is 0.42R. The
    give-back closer arms at 0.50R, the peak-stall closer at 0.60R, and the
    give-back SUPERVISION trigger — the one that only asks — had inherited the
    closer's 0.50R floor. Both milestone ladders speak only on a new high. So
    between EUR 1.00 and EUR 0.75 there was no rule, no reviewer, and no call.

    Arming the question in money does not close anything. It buys one review at
    the moment there is still something left to decide, and the decision is the
    reviewer's: hold it because the pullback is a pullback, or bank it because
    it is not. That is the difference between judgement and a tripwire, and it
    is the whole reason the closers keep their R floors here.
    """

    #: Entry 100, original stop 97, so 1R is three points — the wide structural
    #: stop that makes EUR 1.00 read as 0.42R.
    RISK = 3.0
    PEAK_R = 0.42

    def _runner(self, *, profit: float, mfe_r: float, price: float) -> JarvisRunner:
        runner = _runner()
        runner.manager.equity = 133.0
        runner.settings.trade_management.supervision_profit_step_equity_pct = 0.5
        runner.settings.trade_management.supervision_giveback_trigger_fraction = 0.15
        runner.journal = SimpleNamespace(  # type: ignore[assignment]
            open_trade_by_ticket=lambda _t: {"sl": 97.0, "mfe_r": mfe_r}
        )
        runner.broker.bid = price
        runner._supervised_at[1] = NOW - timedelta(minutes=5)
        runner._supervision_due_at[1] = NOW + timedelta(minutes=10)
        self._profit = profit
        return runner

    def _position(self, profit: float) -> Position:
        return Position(
            ticket=1,
            symbol="TEST",
            direction=Direction.LONG,
            volume=0.01,
            price_open=100.0,
            sl=97.0,
            tp=120.0,
            profit=profit,
            swap=0.0,
            opened_at=NOW - timedelta(hours=1),
        )

    @staticmethod
    def _was(r_now: float, pct: float) -> _SupervisionSnapshot:
        return _SupervisionSnapshot(
            r_now=r_now,
            peak_r=r_now,
            giveback_fraction=0.0,
            health_verdict="healthy",
            health_severity=0.0,
            profit_pct_of_equity=pct,
            peak_pct_of_equity=pct,
        )

    def test_a_small_r_peak_draining_in_money_now_asks(self) -> None:
        """Peaked at EUR 1.00 (0.42R), now EUR 0.80. Twenty percent gone."""
        runner = self._runner(profit=0.80, mfe_r=self.PEAK_R, price=100.0 + 0.336 * self.RISK)
        runner._supervision_snapshots[1] = self._was(self.PEAK_R, 0.752)

        triggered = runner._supervision_trigger(self._position(0.80), NOW)

        assert triggered is not None
        assert triggered[0].startswith("profit_giveback:")

    def test_the_r_floor_alone_would_have_stayed_silent(self) -> None:
        """Identical trade, money route off: 0.42R never reaches the 0.50R arm."""
        runner = self._runner(profit=0.80, mfe_r=self.PEAK_R, price=100.0 + 0.336 * self.RISK)
        runner.settings.trade_management.supervision_profit_step_equity_pct = 0.0
        runner._supervision_snapshots[1] = self._was(self.PEAK_R, 0.752)

        assert runner._supervision_trigger(self._position(0.80), NOW) is None

    def test_a_peak_too_small_to_matter_is_still_ignored(self) -> None:
        """Not every wobble is worth paying for. EUR 0.20 on EUR 133 is not."""
        runner = self._runner(profit=0.16, mfe_r=0.084, price=100.0 + 0.0672 * self.RISK)
        runner._supervision_snapshots[1] = self._was(0.084, 0.150)

        assert runner._supervision_trigger(self._position(0.16), NOW) is None

    def test_the_reviewer_is_told_the_peak_in_money(self) -> None:
        """A shape is not a decision. "It was EUR 1.00, it is EUR 0.85" is."""
        payload = _payload_at_peak(0.60)

        assert payload["peak_unrealised_money"] == pytest.approx(3.85, abs=0.25)
        assert payload["money_handed_back_from_peak"] == pytest.approx(1.03, abs=0.25)

    def test_no_peak_means_no_invented_number(self) -> None:
        payload = _payload_at_peak(0.0)

        assert payload["peak_unrealised_money"] is None
        assert payload["money_handed_back_from_peak"] is None


def _payload_at_peak(peak_r: float):  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    import pandas as pd

    from advisory.providers import build_supervision_payload
    from core.types import Direction, MarketContext, Position, Series, Tick, Timeframe

    now = datetime(2026, 8, 14, 19, 40, tzinfo=UTC)
    index = pd.date_range("2026-08-14", periods=60, freq="15min", tz=UTC)
    close = pd.Series([0.5850 + i * 0.00002 for i in range(60)], index=index)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
            "tick_volume": 100,
            "spread": 10,
            "real_volume": 0,
        },
        index=index,
    )
    context = MarketContext(
        symbol="CADCHF.i",
        now=now,
        series={Timeframe.M15: Series("CADCHF.i", Timeframe.M15, frame, now)},
        tick=Tick("CADCHF.i", now, 0.58595, 0.58600),
    )
    position = Position(
        ticket=134663779,
        symbol="CADCHF.i",
        direction=Direction.LONG,
        volume=0.05,
        price_open=0.58542,
        sl=0.58422,
        tp=0.58691,
        profit=2.82,
        swap=0.0,
        opened_at=now,
    )
    return build_supervision_payload(position, context, {"account_equity": 130.0, "peak_r": peak_r})


class TestTheLosingHalfIsAskedAboutToo:
    """ "Ook als het -50 cent is: hey, gaat dit verder zakken?"

    Every trigger in this function clipped its reading with `max(x, 0.0)` — the
    R milestone, the cash milestone, and both halves of the give-back. Four
    separate places, the same clip, and between them the losing half of a trade
    was invisible. A position could walk from -0.1R to -0.8R, most of the risk
    budget, and cross nothing at all.

    What remained was the fifteen-minute scheduled review and a health reader
    that speaks about structure rather than about money. Neither is the question
    a person asks while watching a trade go against them, and that question is
    worth more here than on the winning side: a gain handed back can be taken
    again tomorrow, and a loss cannot be un-lost.
    """

    RISK = 3.0

    def _runner(self, *, mfe_r: float = 0.0) -> JarvisRunner:
        runner = _runner()
        runner.manager.equity = 133.0
        runner.settings.trade_management.supervision_loss_step_r = 0.25
        runner.settings.trade_management.supervision_loss_step_equity_pct = 0.35
        runner.journal = SimpleNamespace(  # type: ignore[assignment]
            open_trade_by_ticket=lambda _t: {"sl": 97.0, "mfe_r": mfe_r}
        )
        runner._supervised_at[1] = NOW - timedelta(minutes=5)
        runner._supervision_due_at[1] = NOW + timedelta(minutes=10)
        return runner

    @staticmethod
    def _position(profit: float) -> Position:
        return Position(
            ticket=1,
            symbol="TEST",
            direction=Direction.LONG,
            volume=0.01,
            price_open=100.0,
            sl=97.0,
            tp=120.0,
            profit=profit,
            swap=0.0,
            opened_at=NOW - timedelta(hours=1),
        )

    @staticmethod
    def _was(r_now: float, pct: float) -> _SupervisionSnapshot:
        return _SupervisionSnapshot(
            r_now=r_now,
            peak_r=max(r_now, 0.0),
            giveback_fraction=0.0,
            health_verdict="healthy",
            health_severity=0.0,
            profit_pct_of_equity=pct,
            peak_pct_of_equity=max(pct, 0.0),
        )

    def test_a_deepening_loss_in_r_now_asks(self) -> None:
        """-0.2R to -0.4R crosses the 0.25R rung on the losing side."""
        runner = self._runner()
        runner.broker.bid = 100.0 - 0.4 * self.RISK
        runner._supervision_snapshots[1] = self._was(-0.2, -0.30)

        triggered = runner._supervision_trigger(self._position(-0.94), NOW)

        assert triggered is not None
        assert triggered[0].startswith("loss_deepened")

    def test_fifty_cents_against_is_a_question_not_a_shrug(self) -> None:
        """EUR 0.47 is the first cash rung on a EUR 133 account. This is it."""
        runner = self._runner()
        # -0.16R: far short of the 0.25R rung, so only the money route can speak.
        runner.broker.bid = 100.0 - 0.16 * self.RISK
        runner._supervision_snapshots[1] = self._was(-0.05, -0.11)

        triggered = runner._supervision_trigger(self._position(-0.50), NOW)

        assert triggered is not None
        assert triggered[0].startswith("loss_deepened_in_cash")

    def test_a_loss_that_is_recovering_is_left_alone(self) -> None:
        """Coming back is not new evidence. Only a deeper loss is."""
        runner = self._runner()
        runner.broker.bid = 100.0 - 0.1 * self.RISK
        runner._supervision_snapshots[1] = self._was(-0.6, -1.20)

        assert runner._supervision_trigger(self._position(-0.30), NOW) is None

    def test_both_ladders_can_be_switched_off(self) -> None:
        runner = self._runner()
        runner.settings.trade_management.supervision_loss_step_r = 0.0
        runner.settings.trade_management.supervision_loss_step_equity_pct = 0.0
        runner.broker.bid = 100.0 - 0.9 * self.RISK
        runner._supervision_snapshots[1] = self._was(-0.05, -0.10)

        assert runner._supervision_trigger(self._position(-2.10), NOW) is None

    def test_the_overlay_asks_sooner_on_the_losing_side(self) -> None:
        """Not a symmetry slip. Losses are the half you cannot re-enter out of."""
        from config.loader import load_settings

        settings = load_settings("config/config.yaml", overlay="config/eightcap.yaml")
        management = settings.trade_management

        assert management.supervision_loss_step_equity_pct < (
            management.supervision_profit_step_equity_pct
        )


class TestTheManagementRecordReachesTheManagementDecision:
    """The account's own replay was being read by the wrong reviewer.

    `what_stepping_in_has_earned` replays every closed trade against its own
    untouched stop and target, so it is the only evidence in this system about
    MANAGEMENT rather than entries. Its own text in the briefing says "this is
    the question you are being asked".

    It was attached to the payload that asks whether to OPEN something — which
    cannot act on it — and the payload deciding whether to hold or bank a live
    position received the local memory and nothing else. So the ten-trade
    replay showing PEAK_STALL banking +0.43R where holding paid +1.92R was
    shown only to the half of the system that has no exits to make.
    """

    def _runner(self, *, brain_says: dict) -> tuple[JarvisRunner, list]:  # type: ignore[type-arg]
        from advisory.providers import Supervision
        from runner.service import OperationMode

        seen: list = []

        class _Advisor:
            supports_dynamic_management = True

            def supervise(self, payload):  # type: ignore[no-untyped-def]
                seen.append(payload)
                return Supervision(action="hold", confidence=0.6, reason="test", model="stub")

        runner = _runner()
        runner.advisor = _Advisor()  # type: ignore[assignment]
        runner.operation = OperationMode.EXPERIMENTAL_LIVE
        runner.clock = SimpleNamespace(now=lambda: NOW)  # type: ignore[assignment]
        runner.manager.equity = 133.0
        runner.manager.apply_supervision = lambda *_: None
        runner.brain = SimpleNamespace(  # type: ignore[assignment]
            briefing=lambda *_: brain_says,
            record_supervision=lambda **_: None,
        )
        runner.memory = SimpleNamespace(briefing=lambda *_: {})  # type: ignore[assignment]
        # The forward read. A stub that opens nothing keeps it at "no_setup",
        # which is neither agreement nor opposition and changes no assertion
        # in this class.
        runner.engine = SimpleNamespace(  # type: ignore[assignment]
            evaluate=lambda *_: SimpleNamespace(
                direction=None, reason="flat", score=0.0, confidence=0.0, approved=False
            )
        )
        runner.settings.mode = SimpleNamespace(is_live=False)
        runner.posture = SimpleNamespace(brief=lambda: {})  # type: ignore[assignment]
        runner.journal = SimpleNamespace(  # type: ignore[assignment]
            open_trade_by_ticket=lambda _t: {"sl": 97.0, "mfe_r": 0.0},
            supervision_context=lambda _t: {},
        )
        runner.broker.account = lambda: SimpleNamespace(currency="EUR", equity=133.0)
        runner.data = SimpleNamespace(get_context=lambda _s: _context())  # type: ignore[assignment]
        runner.ai_ledger = SimpleNamespace(append=lambda *_a, **_k: None)  # type: ignore[assignment]
        runner._headlines_for = lambda _s: []  # type: ignore[method-assign]
        runner._health_brief = lambda _t: {}  # type: ignore[method-assign]
        runner._managed_positions = lambda: ()  # type: ignore[method-assign]
        runner._brain_trades = {}
        return runner, seen

    def test_the_supervisor_is_handed_the_account_record(self) -> None:
        record = {
            "what_stepping_in_has_earned": {
                "records": ["PEAK_STALL: 3 trades, took +0.43R, holding paid +1.92R"],
                "weight": "the only evidence about management",
            }
        }
        runner, seen = self._runner(brain_says=record)

        runner._supervise_positions([_position()])

        assert seen, "the supervisor was never called"
        assert seen[0]["context"]["learned_over_the_account_lifetime"] == record

    def test_an_empty_record_is_left_out_rather_than_sent_blank(self) -> None:
        """A blank key reads as a consulted record. A missing one reads as none."""
        runner, seen = self._runner(brain_says={})

        runner._supervise_positions([_position()])

        assert seen
        assert "learned_over_the_account_lifetime" not in seen[0]["context"]


def _context():  # type: ignore[no-untyped-def]
    import pandas as pd

    from core.types import MarketContext, Series, Tick, Timeframe

    index = pd.date_range("2026-08-08", periods=60, freq="15min", tz=UTC)
    close = pd.Series([100.0 + i * 0.01 for i in range(60)], index=index)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "tick_volume": 100,
            "spread": 10,
            "real_volume": 0,
        },
        index=index,
    )
    return MarketContext(
        symbol="TEST",
        now=NOW,
        series={Timeframe.M15: Series("TEST", Timeframe.M15, frame, NOW)},
        tick=Tick("TEST", NOW, 101.0, 101.1),
    )


class TestTheOnlyReadingThatLooksForward:
    """ "Als het systeem zeker weet dit gaat nog meer zakken, laat me er nu uit."

    Every other reading about an open position is a post-mortem in progress.
    The health engine reports damage already done. The excursions report price
    already travelled. The account record reports what past exits earned. Not
    one of them separates the two states the decision actually turns on:

        down, and the thesis is intact        -> sit through it
        down, and the market has turned       -> leave, size of loss irrelevant

    In every field of the payload those two look identical, which is why the
    answer to both was hold. The engine that opened the trade is asked again on
    the live chart, and it answers exactly that question — in both directions,
    because a flip back into agreement is the moment a loser becomes worth
    holding rather than a moment to do nothing about.
    """

    @staticmethod
    def _runner(engine_says: str | None, score: float = 61.0):  # type: ignore[no-untyped-def]

        runner = _runner()
        runner.engine = SimpleNamespace(  # type: ignore[assignment]
            evaluate=lambda *_: SimpleNamespace(
                direction=None if engine_says is None else Direction[engine_says],
                score=score,
                confidence=0.72,
                approved=True,
                reason="live read",
            )
        )
        runner.settings.mode = SimpleNamespace(is_live=False)
        runner._supervised_at[1] = NOW - timedelta(minutes=5)
        runner._supervision_due_at[1] = NOW + timedelta(minutes=10)
        return runner

    @staticmethod
    def _was(*, engine_against: bool) -> _SupervisionSnapshot:
        return _SupervisionSnapshot(
            r_now=-0.30,
            peak_r=0.0,
            giveback_fraction=0.0,
            health_verdict="healthy",
            health_severity=0.0,
            profit_pct_of_equity=-0.30,
            peak_pct_of_equity=0.0,
            engine_against=engine_against,
        )

    def test_a_turn_against_an_open_position_wakes_the_reviewer(self) -> None:
        runner = self._runner("SHORT")
        runner._latest_direction["TEST"] = ("SHORT", 61.0)
        runner._supervision_snapshots[1] = self._was(engine_against=False)

        triggered = runner._supervision_trigger(_position(), NOW)

        assert triggered is not None
        assert triggered[0].startswith("engine_turned_against:SHORT")

    def test_a_turn_back_into_agreement_wakes_it_too(self) -> None:
        """The moment a losing trade becomes one worth sitting through."""
        runner = self._runner("LONG")
        runner._latest_direction["TEST"] = ("LONG", 58.0)
        runner._supervision_snapshots[1] = self._was(engine_against=True)

        triggered = runner._supervision_trigger(_position(), NOW)

        assert triggered is not None
        assert triggered[0].startswith("engine_back_in_agreement")

    def test_being_on_the_wrong_side_is_not_re_asked_every_two_minutes(self) -> None:
        """It fires on the change, not on the state. Otherwise a position on the
        wrong side would bill a review on every guard tick until it closed."""
        runner = self._runner("SHORT")
        runner._latest_direction["TEST"] = ("SHORT", 61.0)
        runner._supervision_snapshots[1] = self._was(engine_against=True)

        assert runner._supervision_trigger(_position(), NOW) is None

    def test_the_reviewer_is_told_which_way_the_engine_now_reads(self) -> None:
        runner = self._runner("SHORT")

        read = runner._conviction_now(_position(), context=object())

        assert read["verdict"] == "against"
        assert read["engine_would_take"] == "SHORT"
        assert read["position_holds"] == "LONG"
        assert "wrong side" in str(read["means"])

    def test_agreement_is_stated_as_a_reason_to_sit_through_a_loss(self) -> None:
        """Half the value is here. Without it, 'we are down' is the only fact
        on the table and every reading of it points one way."""
        runner = self._runner("LONG")

        read = runner._conviction_now(_position(), context=object())

        assert read["verdict"] == "agrees"
        assert "not by itself a reason to leave" in str(read["means"])

    def test_no_setup_is_not_counted_as_disagreement(self) -> None:
        """The absence of evidence, said plainly. Collapsing this into 'against'
        would turn every quiet hour into a reason to close."""
        runner = self._runner(None)

        read = runner._conviction_now(_position(), context=object())

        assert read["verdict"] == "no_setup"
        assert "absence of evidence" in str(read["means"])

    def test_an_unreadable_chart_says_unknown_rather_than_guessing(self) -> None:
        runner = self._runner("SHORT")
        runner.engine = SimpleNamespace(  # type: ignore[assignment]
            evaluate=lambda *_: (_ for _ in ()).throw(ValueError("no bars"))
        )

        read = runner._conviction_now(_position(), context=object())

        assert read["verdict"] == "unknown"
