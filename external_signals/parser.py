"""Conservative parser for the Rio/Gold signal formats seen by the owner."""

from __future__ import annotations

import re

from core.types import Direction
from external_signals.models import ExternalSignalEvent, ExternalSignalKind, NotificationEnvelope

_NUMBER = r"(?P<number>\d+(?:[.,]\d+)?)"
_SYMBOLS = (
    "XAUUSD",
    "GOLD",
    "BTCUSD",
    "ETHUSD",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
    "EURJPY",
    "GBPJPY",
    "EURGBP",
    "US30",
    "US500",
    "NAS100",
    "US2000",
)


def _price(value: str) -> float:
    return float(value.replace(",", "."))


class RioSignalParser:
    """Recognise instructions without treating marketing prose as an order."""

    def parse(self, envelope: NotificationEnvelope) -> ExternalSignalEvent:
        raw = envelope.combined_text
        text = self._normalise(raw)
        symbol = self._symbol(text)

        move = re.search(rf"\bMOVE\s+SL(?:\s+(?:TO|AT))?\s*[:=@-]?\s*{_NUMBER}", text)
        if move:
            return ExternalSignalEvent(
                envelope,
                ExternalSignalKind.MOVE_STOP,
                symbol_alias=symbol,
                move_stop_to=_price(move.group("number")),
                reason="provider requested a stop-loss change",
            )

        if re.search(r"\b(CLOSE|EXIT)\s+(?:NOW|TRADE|POSITION|GOLD|XAUUSD)\b", text):
            return ExternalSignalEvent(
                envelope,
                ExternalSignalKind.CLOSE,
                symbol_alias=symbol,
                reason="provider requested a full close",
            )
        if re.search(r"\bCANCEL(?:LED)?\b", text):
            return ExternalSignalEvent(
                envelope,
                ExternalSignalKind.CANCEL,
                symbol_alias=symbol,
                reason="provider cancelled the setup",
            )
        if re.search(r"\bBOOK\s+(?:SOME\s+)?PROFITS?\b|\bPARTIAL(?:LY)?\s+CLOSE\b", text):
            return ExternalSignalEvent(
                envelope,
                ExternalSignalKind.BOOK_PROFIT,
                symbol_alias=symbol,
                reason="provider requested partial profit taking",
            )
        if re.search(r"\bTP\s*\d*\s*(?:HIT|DONE|REACHED)\b", text):
            return ExternalSignalEvent(
                envelope,
                ExternalSignalKind.TP_HIT,
                symbol_alias=symbol,
                reason="provider reported a target hit",
            )

        direction_match = re.search(r"\b(BUY|LONG|SELL|SHORT)\b", text)
        if not direction_match:
            return ExternalSignalEvent(
                envelope,
                ExternalSignalKind.INFO,
                symbol_alias=symbol,
                reason="notification contains no entry or management instruction",
            )
        direction = (
            Direction.LONG if direction_match.group(1) in {"BUY", "LONG"} else Direction.SHORT
        )
        after = text[direction_match.end() :]
        entry_match = re.search(
            r"\s*[:=@-]?\s*(\d+(?:[.,]\d+)?)"
            r"(?:\s*(?:-|TO)\s*(\d+(?:[.,]\d+)?))?",
            after,
        )
        entry_low = _price(entry_match.group(1)) if entry_match else None
        entry_high = (
            _price(entry_match.group(2))
            if entry_match and entry_match.group(2)
            else entry_low
        )
        stop = self._labelled_price(text, ("SL", "STOP", "STOPLOSS", "STOP LOSS"))
        targets = self._targets(text)
        missing = [
            label
            for label, present in (
                ("symbol", symbol),
                ("entry", entry_low),
                ("stop loss", stop),
                ("take profit", targets),
            )
            if not present
        ]
        reason = (
            f"incomplete signal: missing {', '.join(missing)}"
            if missing
            else "complete provider entry signal"
        )
        return ExternalSignalEvent(
            envelope,
            ExternalSignalKind.NEW if not missing else ExternalSignalKind.INVALID,
            symbol_alias=symbol,
            direction=direction,
            entry_low=(
                min(entry_low, entry_high) if entry_low is not None and entry_high else entry_low
            ),
            entry_high=(
                max(entry_low, entry_high)
                if entry_low is not None and entry_high
                else entry_high
            ),
            stop_loss=stop,
            take_profits=targets,
            reason=reason,
        )

    @staticmethod
    def _normalise(value: str) -> str:
        text = value.upper().replace("\u2013", "-").replace("\u2014", "-")
        text = text.replace("/", " ").replace("_", " ")
        return re.sub(r"[ \t]+", " ", text)

    @staticmethod
    def _symbol(text: str) -> str | None:
        for symbol in _SYMBOLS:
            if re.search(rf"\b{re.escape(symbol)}\b", text):
                return symbol
        # Generic six-letter FX/metal pair, deliberately after known aliases.
        match = re.search(r"\b[A-Z]{6}\b", text)
        return match.group(0) if match else None

    @staticmethod
    def _labelled_price(text: str, labels: tuple[str, ...]) -> float | None:
        joined = "|".join(re.escape(label) for label in labels)
        match = re.search(rf"(?:{joined})\s*[:=@-]?\s*(\d+(?:[.,]\d+)?)", text)
        return _price(match.group(1)) if match else None

    @staticmethod
    def _targets(text: str) -> tuple[float, ...]:
        values: list[float] = []
        pattern = re.compile(
            r"\b(?:TP|TAKE\s*PROFIT)\s*\d*\s*[:=@-]?\s*(\d+(?:[.,]\d+)?)"
        )
        for match in pattern.finditer(text):
            value = _price(match.group(1))
            if value not in values:
                values.append(value)
        return tuple(values)
