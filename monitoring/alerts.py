"""Best-effort operator alerts; failures never relax trading gates."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from config.schema import MonitoringConfig
from infra.logging import get_logger

log = get_logger(__name__)


class AlertSender:
    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config

    def send(self, message: str) -> bool:
        if not self.config.alerts_enabled or self.config.alert_channel == "none":
            return False
        try:
            if self.config.alert_channel == "discord":
                url = os.environ["DISCORD_WEBHOOK_URL"]
                payload = json.dumps({"content": message[:1900]}).encode()
                content_type = "application/json"
            else:
                token = os.environ["TELEGRAM_BOT_TOKEN"]
                chat = os.environ["TELEGRAM_CHAT_ID"]
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                payload = urllib.parse.urlencode({"chat_id": chat, "text": message}).encode()
                content_type = "application/x-www-form-urlencoded"
            request = urllib.request.Request(
                url, data=payload, headers={"Content-Type": content_type}
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                return 200 <= response.status < 300
        except Exception as exc:
            log.exception(
                "alert delivery failed", extra={"event": "alert_failure", "error": str(exc)}
            )
            return False
