"""Leave an instrument alone for a while after it has taken money off us.

The live sequence, 6 August:

    18:51:29  GBPNZD short opened
    18:53:43  GBPNZD short closed, -0.71R
    18:54:58  GBPNZD short opened again
    18:57:36  GBPNZD short closed, -0.48R

Seventy-five seconds between them. Nothing in the system objected, because
nothing was watching for it: the position filters all ask what is open *now*,
and by 18:54:58 nothing was. The concentration filter saw no GBP exposure, the
slot limit saw a free slot, and the playbook read the same M15 range it had
read three minutes earlier and reached the same conclusion — because it was
looking at almost the same bars.

That is the whole mechanism. A playbook reads a window of history; a loss
consumes a few minutes of it; the window it reads next is mostly the window it
just read. The setup has not been re-evaluated so much as re-encountered.

So this filter has no opinion about the market at all. It asks one question of
the journal — did this instrument close a loser recently — and refuses on the
answer. Deliberately not a judgement about whether the new setup is good: the
first one looked good too.

**Direction is not considered, on purpose.** Taking the other side after being
wrong can be a legitimate reversal read, and blocking it is a stronger claim
than the evidence supports. But flipping within minutes is the same churn with
a minus sign, and the cost of being wrong here is one skipped setup out of the
twelve this system ranks every cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from config.schema import LossCooldownConfig
from filters.base import Filter, FilterContext, FilterVerdict
from risk.reasons import Reason

#: symbol -> when its last losing trade closed, or None. Injected rather than
#: taking a `Journal`, so this filter is a pure function of two inputs and can
#: be tested and replayed without a database.
LastLoss = Callable[[str], datetime | None]


class LossCooldownFilter(Filter):
    name = "loss_cooldown"

    def __init__(self, config: LossCooldownConfig, last_loss: LastLoss) -> None:
        self.config = config
        self.last_loss = last_loss

    def check(self, ctx: FilterContext) -> FilterVerdict:
        if not self.config.enabled or self.config.minutes <= 0:
            return FilterVerdict.allow(self.name, "cooldown disabled")

        try:
            closed_at = self.last_loss(ctx.symbol)
        except Exception as exc:  # noqa: BLE001 - a journal that cannot answer is not a pass
            return FilterVerdict.block(
                self.name,
                Reason.LOSS_COOLDOWN,
                f"could not read the trade history for {ctx.symbol}: {exc}",
            )
        if closed_at is None:
            return FilterVerdict.allow(self.name, "no recent loss on this instrument")

        # Both sides made tz-aware or both naive: a journal written before the
        # timestamps carried a zone would otherwise raise here, and a filter
        # that raises on old data blocks every symbol it has history for.
        now = ctx.now
        if closed_at.tzinfo is None and now.tzinfo is not None:
            closed_at = closed_at.replace(tzinfo=now.tzinfo)
        elif closed_at.tzinfo is not None and now.tzinfo is None:
            now = now.replace(tzinfo=closed_at.tzinfo)

        window = timedelta(minutes=self.config.minutes)
        elapsed = now - closed_at
        if elapsed >= window:
            return FilterVerdict.allow(
                self.name,
                f"last loss here was {elapsed.total_seconds() / 60:.0f} min ago",
                minutes_since_loss=round(elapsed.total_seconds() / 60, 1),
            )

        waited = elapsed.total_seconds() / 60
        return FilterVerdict.block(
            self.name,
            Reason.LOSS_COOLDOWN,
            f"{ctx.symbol} closed a loser {waited:.1f} min ago and the cooldown is "
            f"{self.config.minutes:g} min; the chart has not had time to become a "
            f"different chart",
            minutes_since_loss=round(waited, 1),
            cooldown_minutes=self.config.minutes,
        )
