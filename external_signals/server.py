"""Small authenticated HTTP receiver; no broker calls happen on this thread."""

from __future__ import annotations

import hmac
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any
from urllib.parse import parse_qs

from external_signals.inbox import ExternalSignalInbox


class SignalReceiver:
    """Accept MacroDroid posts and put them on disk for the single MT5 thread."""

    def __init__(self, inbox: ExternalSignalInbox, host: str, port: int, token: str) -> None:
        if len(token) < 24:
            raise ValueError("external signal token must contain at least 24 characters")
        self.inbox = inbox
        self.host = host
        self.port = port
        self.token = token
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/health":
                    self._reply(HTTPStatus.OK, {"ok": True, "service": "external-signals"})
                else:
                    self._reply(HTTPStatus.NOT_FOUND, {"ok": False})

            def do_POST(self) -> None:
                if self.path.rstrip("/") != "/v1/rio":
                    self._reply(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                    return
                supplied = self.headers.get("X-Jarvis-Token", "")
                if not hmac.compare_digest(supplied, receiver.token):
                    self._reply(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
                    return
                try:
                    payload = self._payload()
                    envelope = receiver.inbox.enqueue(payload)
                except (ValueError, json.JSONDecodeError) as exc:
                    self._reply(
                        HTTPStatus.BAD_REQUEST,
                        {"ok": False, "error": "invalid_payload", "detail": str(exc)},
                    )
                    return
                self._reply(
                    HTTPStatus.ACCEPTED,
                    {"ok": True, "event_id": envelope.event_id, "status": "queued"},
                )

            def _payload(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 131_072:
                    raise ValueError("request body must contain 1 to 131072 bytes")
                body = self.rfile.read(length)
                content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
                if content_type == "application/json":
                    raw = json.loads(body.decode("utf-8"))
                    if not isinstance(raw, dict):
                        raise ValueError("JSON body must be an object")
                    return raw
                values = parse_qs(body.decode("utf-8"), keep_blank_values=True)
                return {key: items[-1] if items else "" for key, items in values.items()}

            def _reply(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
                data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = Thread(target=self._server.serve_forever, name="rio-receiver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
