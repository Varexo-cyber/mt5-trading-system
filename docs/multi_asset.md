# Multi-asset policy

The broker catalogue may be broad; the live trading whitelist must remain
narrow. Display access and order permission are deliberately separate.

| Asset class | Default context | Cost unit | Session model | Live status |
|---|---|---|---|---|
| Forex | D1, H4, H1, M15, M5 | pips + bps | London/New York, rollover blocked | micro-live candidate |
| Crypto | D1, H4, H1, M15, M5 | bps + money at min lot | continuous profile, broker tick required | research only |
| Stocks | W1, D1, H4, H1, M15 | bps + money at min lot | exchange-specific | research only |
| Indices | D1, H4, H1, M15, M5 | bps + ATR | cash/futures session-specific | research only |
| Metals | W1, D1, H4, H1, M15 | bps + ATR | London/New York/liquidity windows | blocked below equity gate |
| Commodities | W1, D1, H4, H1 | bps + ATR | contract-specific | research only |

## Spread decision

An entry clears the cost gate only when all available checks pass:

1. A valid non-zero quote exists.
2. Its age is below the asset-class limit.
3. The spread is below the symbol/hour median multiple.
4. Weekend and weekday observations are never mixed.
5. A configured absolute pip ceiling passes when one exists.
6. The cross-asset basis-point ceiling passes.
7. The money cost at minimum lot is recorded for the journal and report.

Raw means pricing can start at zero on selected markets; it does not promise a
zero or fixed spread. Crypto, stocks and CFDs can use different pricing from
Raw FX. The bot trusts executable MT5 quotes and learned symbol-specific costs,
not the word `Raw` in the account name.

## Timeframes

All 21 MT5 timeframes are available to the dashboard. The decision engine uses
a small hierarchy per asset instead of loading 847 × 21 charts every cycle.
Loading everything would increase latency and multiple-testing risk without
adding independent information. A timeframe may receive decision weight only
after its hypothesis is pre-registered and validated out of sample.
