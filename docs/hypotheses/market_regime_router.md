# Hypothesis: market regime router

> Pre-registered before implementation or backtesting. The values below are
> predictions, not measured results.

## Claim

On liquid FX majors, conditioning directional evidence on a closed-bar market
regime should improve net expectancy and reduce drawdown relative to the same
signals with static weights. Trend-following evidence should receive its base
weight in persistent markets and a reduced weight in ranges; reversal evidence
should receive its base weight in ranges and a reduced weight in persistent
markets. Extreme-volatility observations remain a veto.

## Economic rationale

The existing modules describe different mechanisms. Market structure and
trend momentum depend on persistence after information or positioning shocks.
Liquidity sweeps and level reactions depend on temporary price pressure being
absorbed and reverting. Treating these mechanisms as equally relevant in every
state mixes incompatible conditional distributions and creates false
confidence by adding scores that answer different questions.

Volatility and directional efficiency are not claimed to predict returns on
their own. They are context variables. The proposed router is therefore
non-directional and cannot create a trade without independently directional
evidence.

The volatility component is motivated by the documented difference between
expected returns and changing realised volatility in volatility-managed
portfolios. The persistence component is consistent with the documented
time-series-momentum effect across equity-index, currency, commodity and bond
futures. Neither paper validates this exact intraday implementation; this
hypothesis must still pass its own out-of-sample test.

Primary references:

- Moskowitz, Ooi and Pedersen (2012), *Time Series Momentum*, Journal of
  Financial Economics 104, 228-250.
- Moreira and Muir (2017), *Volatility-Managed Portfolios*, Journal of Finance
  72, 1611-1644.

## Operational definition

- Use only closed H1 and H4 bars.
- Directional efficiency is Kaufman's efficiency ratio:
  `abs(close[t] - close[t-L]) / sum(abs(delta close))`, bounded to 0..1.
- Direction is the sign of the close change over the same lookback.
- `trend_up` or `trend_down`: H1 and H4 efficiency are each at least 0.30 and
  their directions agree.
- `range`: H1 efficiency is at most 0.20 and H4 efficiency at most 0.25.
- `transition`: neither trend nor range.
- `extreme`: current H1 ATR(14) is at or above the 95th percentile of the last
  100 valid H1 ATR observations. Extreme takes precedence over every state.
- In a trend, `market_structure` and `trend_momentum` retain 1.00 of their base
  weight; `liquidity_sweep` and `level_reaction` receive 0.35.
- In a range, the reversal modules retain 1.00 and the trend modules receive
  0.35.
- In transition, all base weights are unchanged.
- The router cannot change signal direction, create evidence, relax filters,
  increase position size, or override a veto.
- Research/backtest and paper modes may apply routing. Live routing remains
  disabled until the validation criteria below pass.

## Predictions, stated in advance

- Expected change in net expectancy: **+0.03R to +0.15R per trade** relative
  to the identical static-weight engine.
- Expected change in maximum drawdown: **5% to 25% lower** on the validation
  split, measured in R.
- Expected change in qualified trade count: **10% to 45% lower**.
- Expected number of routed signals: at least **100 per evaluated regime**
  across the development instruments; no claim is made below that count.
- Conditions where this should **fail**: rapid regime changes; discontinuous
  markets; insufficient H4 history; instruments whose intraday behaviour is
  dominated by scheduled gaps; or when efficiency ratios contain no
  incremental information beyond the existing modules.

The hypothesis is rejected if net validation expectancy does not improve, if
drawdown worsens by more than 5%, if results depend on one narrow threshold, or
if held-back instruments reverse the sign of the effect.

## Test design

- Development instruments: **EURUSD, GBPUSD, USDJPY**.
- Cross-instrument holdback: **AUDUSD, USDCAD**.
- Primary timeframes: **H1 and H4**.
- Data split: oldest 60% train / next 20% validation / final 20% untouched.
- Costs: broker-realistic spread, commission, adverse slippage and swap; the
  same fills must be used for the routed and static engines.
- Primary comparison: paired routed versus static-weight decisions at every
  timestamp. Report expectancy, maximum drawdown, turnover, trade count and
  rejection reasons by regime.
- Sensitivity grid: efficiency lookback **20, 24, 30** and incompatible weight
  multiplier **0.20, 0.35, 0.50**. All other thresholds remain fixed.
- **Number of configurations tested: 9.** This count is included in the
  deflated-Sharpe correction before viewing the final holdout.

## Result

Not tested yet. Results will be recorded in
`docs/modules/market_regime_router.md`.

- [ ] At least 100 trades per evaluated regime
- [ ] Broad parameter plateau, not a spike
- [ ] Positive incremental result on held-back instruments
- [ ] Deflated Sharpe remains positive after 9 configurations
- [ ] Final 20% inspected exactly once

## Verdict

`live routing disabled; research/paper routing enabled for evidence collection`
