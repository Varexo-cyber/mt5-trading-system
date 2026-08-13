# Rio Traders phone bridge

## What it does

MacroDroid forwards the actual notification fields to an authenticated HTTPS
endpoint. Jarvis stores the unedited notification first, parses it second, and
only then considers broker action on the single MT5 thread.

Recognised messages:

- `GOLD SELL 4385-4387 SL 4393 TP1 4378 TP2 4365`: new entry campaign.
- `Move SL at 4390`: tightens the matching open campaign; never widens a stop.
- `Book some profits`: closes the configured fraction when the broker lot step permits it.
- `Close now` / `Cancel`: closes the matching position.
- `TP1 hit`, profit-running messages and daily summaries: recorded, not treated as entries.

An entry without symbol, direction, entry, stop loss or at least one take profit
is recorded as incomplete and cannot place an order. A notification older than
three minutes cannot place or change an order. Duplicate content received within
fifteen minutes is stored once.

## One-time VPS setup

1. Run `setup_rio_bridge.cmd`. It writes a random `RIO_SIGNAL_TOKEN` to
   `config/.env` and shows it once for MacroDroid.
2. Restart `launch_jarvis_experimental_live.cmd`.
3. Keep the receiver on `127.0.0.1:8765`. Do **not** open port 8765 in Windows
   Firewall or expose it directly by IP. Publish it only through an HTTPS tunnel.
4. Fast test route: install Cloudflared once with
   `winget install --id Cloudflare.cloudflared`, then run `launch_rio_tunnel.cmd`.
   Keep that window open and copy its `https://...trycloudflare.com` address.
5. In MacroDroid replace the test notification action with **HTTP Request**:
   - method: `POST`
   - URL: `https://<your-tunnel-host>/v1/rio`
   - header: `X-Jarvis-Token` = the locally generated secret
   - body type: form/url-encoded
   - fields: `app`, `title`, `text`, `big_text`, `notification_timestamp`
   - insert each value from the Notification Trigger's Magic Text picker.

The fast route uses a random URL and therefore requires updating MacroDroid
after a tunnel restart. A named Cloudflare Tunnel is the later 24/7 route; it
needs a Cloudflare account and a domain choice and cannot be guessed by code.

## Audit

Dashboard tab **External signals** shows every receive, parse, refusal, broker
order and provider management update. Raw notifications and state live under
`runtime/external_signals/`; this runtime evidence is not committed to Git.
