# Hypothesis — fresh entries and earlier thesis-break recognition

## Registered before implementation

Date: 2026-08-21

Four live losses exposed two separate defects. Directional evidence could remain
valid while the executable entry arrived after the move: EURAUD was bought at
83% of its recent M5 range on a 0.86 ATR bar, and FRA40 was sold from an impulse
that was already two M15 bars old without a fresh resumption bar. Separately,
`thesis_invalidation_at_r` did not control when the trajectory reader armed;
that reader retained its own fixed 0.35R threshold.

## Expected mechanism

Keep the directional setup alive, but turn a late one-bar thrust or a stale
impulse into `WAIT_RETEST`. A later closed bar that resumes in the setup's
direction may enter immediately through the durable setup lifecycle. This
changes timing, not the number of directional setups discovered.

For open positions, bind the trajectory reader to the configured thesis-break
threshold. A materially losing position still cannot close merely because it is
red: an independent chart family must corroborate the path. The change only
lets that existing two-family decision happen at the threshold the operator
configured.

## Pre-registered intervention

1. Add an asset-normalised body threshold specifically for entries already at
   the directional edge of their recent range.
2. When `impulse_break` is older than the configured freshness allowance,
   require a newly closed entry-timeframe bar to resume in the proposed
   direction before entering.
3. Persist both cases as lifecycle waits rather than deleting their setup.
4. Pass `thesis_invalidation_at_r` into the trajectory reader instead of using
   a second hidden constant.
5. Move the Eightcap threshold from 0.35R to 0.25R. This still requires a
   separate chart family and can only reduce an existing loss.

Risk percentages, total open-risk limits, lot rounding and stop requirements
are outside this intervention.

## Measurements and falsification

Compare waited setups with their immediate-entry counterfactuals on early MAE,
first-touch expectancy and net R after costs. Compare `HEALTH_EXIT` with the
untouched original SL/TP baseline. The hypothesis fails if the waits worsen
expectancy or if earlier health exits have negative lift on an untouched sample.
No profitability claim is permitted below 100 resolved examples per setup
family.
