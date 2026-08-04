"""Re-anchor today's loss baseline to the current equity. Deliberately narrow.

    python scripts/reanchor_day.py            # show what it would do
    python scripts/reanchor_day.py --apply

The daily loss limit measures equity against where the trading day started, and
`set_equity_mark` refuses to move that anchor once written — for a good reason.
Re-anchoring after a bad morning would hand back a fresh daily budget every
restart, and a limit that resets whenever you relaunch is not a limit.

The case this exists for is different and narrow: **the loss was not the
system's.** Trades placed by hand in the terminal move account equity, the
anchor was written before them, and the trader is then halted all day over a
decision it had no part in. That is not the limit doing its job, it is the limit
measuring the wrong thing.

So this refuses to run when the system did trade today. If there are journal
trades since the boundary the loss is at least partly its own, the halt is
correct, and moving the anchor would be exactly the reset the guard prevents.

**What this does not touch**, and none of it should be mistaken for a reset:

* The 4% daily limit still applies — it simply measures from here.
* The weekly loss limit is unchanged.
* The drawdown breaker is unchanged, and it is the one that matters. It runs
  from the all-time equity peak, so a bad morning has already spent part of it
  and no amount of re-anchoring gives that back.
* Risk per trade is unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.clock import LiveClock
from core.mt5_connector import MT5Connector
from journal.database import Journal, iso
from promotion.experimental import apply_experimental_live_limits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write it; otherwise report only")
    args = parser.parse_args(argv)

    settings = apply_experimental_live_limits(
        load_settings(overlay=ROOT / "config" / "eightcap.yaml")
    )
    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=False),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    try:
        account = connector.connect()
    except Exception as exc:  # noqa: BLE001 - the caller only needs the reason
        print(f"Could not connect to MT5: {type(exc).__name__}: {exc}")
        return 1
    finally:
        # A failing shutdown must not mask the account read that just succeeded.
        with contextlib.suppress(Exception):
            connector.shutdown()

    clock = LiveClock()
    journal = Journal(
        ROOT / settings.journal.database_path,
        clock,
        day_boundary_utc=settings.risk.day_boundary_utc,
    ).open()
    try:
        day_start = journal.day_start()
        anchor = journal.equity_mark("DAY", day_start)
        traded = journal.trades_since(day_start)
        limit = settings.effective_daily_loss_limit_pct()

        print(f"  trading day began   {iso(day_start)}")
        print(f"  anchored equity     {anchor:.2f}" if anchor else "  anchored equity     none")
        print(f"  equity now          {account.equity:.2f} {account.currency}")
        if anchor:
            pnl = (account.equity - anchor) / anchor * 100.0
            print(f"  day P&L             {pnl:+.2f}%  against a -{limit:.2f}% limit")
        print(f"  system trades today {traded}")

        if anchor is None:
            print("\n  No anchor for today yet. Nothing to move.")
            return 0

        if traded:
            print(
                f"\n  Refusing: the system opened {traded} trade(s) today, so this day's loss "
                "is at least partly its own.\n  Moving the anchor now would hand it a fresh "
                "budget after a bad run, which is the\n  reset the daily limit exists to "
                "prevent. Wait for the boundary."
            )
            return 1

        floor = settings.risk.max_drawdown_circuit_breaker_pct
        peak = journal.equity_peak() or account.equity
        hard_floor = peak * (1.0 - floor / 100.0)
        print(
            f"\n  The system has not traded today, so the {anchor - account.equity:.2f} "
            f"{account.currency} lost was not its doing."
        )
        risk_money = account.equity * settings.effective_risk_pct() / 100.0
        room = account.equity - hard_floor
        print(
            "\n  What re-anchoring changes:  the daily limit measures from "
            f"{account.equity:.2f} instead of {anchor:.2f}"
        )
        print(
            f"  What it does not change:    the {floor:.0f}% drawdown breaker still ends the "
            f"experiment at {hard_floor:.2f}"
        )
        print(
            f"                              {room:.2f} {account.currency} away, roughly "
            f"{room / risk_money:.1f} losing trades at the current risk"
        )

        if not args.apply:
            print("\n  Report only. Add --apply to write it.")
            return 0

        journal.conn.execute(
            "UPDATE equity_marks SET equity = ?, recorded_at = ? "
            "WHERE period = 'DAY' AND period_key = ?",
            (account.equity, iso(clock.now()), iso(day_start)),
        )
        journal.conn.commit()
        print(f"\n  Anchor moved to {account.equity:.2f}. Restart Jarvis for it to take effect.")
        return 0
    finally:
        journal.close()


if __name__ == "__main__":
    raise SystemExit(main())
