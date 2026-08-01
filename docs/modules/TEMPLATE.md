# Module: <name>

## What it measures

Plain-language description of the signal, and the exact rule that produces it.

## Interface

Score range actually used (not all modules use the full -100..+100), what drives
`confidence`, and what `invalidation_price` means for this module.

## Measured edge

| Split | Trades | Win rate | Avg R | Expectancy | Max DD |
|---|---|---|---|---|---|
| Train (60%) | | | | | |
| Validation (20%) | | | | | |
| Holdout (20%) | | | | | |

Configurations tested: ___
Deflated Sharpe: ___

## Parameter stability

Plot or table across the swept range. An edge that exists at period 21 and
vanishes at 20 and 22 is noise, and this section is where that becomes visible.

## Correlation with other modules

Signal correlation against every module already carrying weight. SMC concepts
in particular overlap heavily — a liquidity sweep and a Wyckoff spring are
often the same event, and counting both is double-counting.

## Known limitations

Where it misfires, and what would break it.

## Weight

Current weight and the date/reasoning of the last change. Every change also goes
in `learning/changelog.md`.
