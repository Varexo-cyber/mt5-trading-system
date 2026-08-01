# Execution architecture

## Decision

Python remains the authoritative analysis, risk, journal, and execution engine.
MT5 is now one broker adapter rather than the architecture. A REST broker can be
added without changing analysis modules or weakening the hard risk rules.

TradingView is optional and non-authoritative:

- useful for visual confirmation and operator monitoring;
- optionally able to emit authenticated alerts;
- never allowed to size a position or bypass the risk manager;
- never treated as proof that a real broker order filled.

## Why

TradingView strategies use a broker emulator and alerts. TradingView currently
does not directly automate Pine strategies against a brokerage account. A
webhook therefore still needs an external execution service and can fail in
transit. Putting TradingView in the critical execution path would add another
failure mode without removing the need for this Python system.

## Contracts

core/broker.py contains two structural protocols:

1. MarketDataProvider: specifications, quotes, and historical closed bars.
2. Broker: connection, account, positions, order execution, modification, and
   closing.

MT5Connector satisfies both contracts. DataManager, startup checks, filters,
and operator commands depend on the protocols instead of MT5Connector.

## Chosen non-MT5 path: cTrader Open API

cTrader Open API is the preferred next adapter. It supports demo and live
accounts, exposes market data and trading operations, has an official Python
SDK, and is available across cTrader-affiliated brokers. That makes it a safer
portable target than binding the engine to one broker's regional REST product.

The adapter cannot be live-verified until the owner has a cTID, a demo account
at a cTrader-affiliated broker, and approved Open API application credentials.
Until then MT5 remains the only executable adapter and cTrader remains a tested
implementation target, not a claimed working connection.

## Recommended first deployment

1. Forex only, because the existing pip maths, calendar filters, sessions, and
   hypotheses are pre-registered for FX.
2. cTrader demo account through Open API, using the broker's actual minimum
   volume and volume step rather than assuming 0.01 lots.
3. Thirty days shadow mode: calculate and journal decisions, send no orders.
4. Practice execution until reconciliation and execution reports contain zero
   unexplained discrepancies.
5. Micro-live only after the broker contract specifications and minimum size
   prove the structural stop fits the risk budget.

No cTrader broker is enabled yet. Availability, instrument names, account
currency, netting model, minimum trade size, and volume conversion must be
verified against the actual demo account before the adapter can become
selectable.
