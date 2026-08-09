# Learning changelog

No parameter changes have been approved. Jarvis may propose changes after at
least 100 trades spanning 30 days, but never applies them automatically.

## 2026-08-08 - evidence controls strengthened

- No trading parameter or risk limit was changed.
- Shadow candidates are evaluated against the active champion on paired future
  price paths. Calendar age or decision count alone can no longer promote one.
- Promotion requires at least 100 resolved differences, 20 unique days, three
  symbols, a configured positive expectancy lift and a positive 95% lower
  confidence bound.
- Repeated validation files covering the same hypothesis, instrument and time
  window count once. Validation reports carry data, configuration and report
  content digests so changed evidence is detectable.
- The playbooks with negative operator-supplied 90-day evidence remain active
  for research but have no Experimental Live entry or veto authority. The
  independent swing/confluence route remains operational.

## 2026-08-09 - reproducibility and validation audit

- No trading parameter or risk limit was changed.
- Counterfactual and management comparisons exclude the decision candle and
  pessimistically score an intrabar stop-and-target collision as a stop.
- Evidence now binds itself to the effective configuration, historical input
  frames and the production implementation digest.
- Market structure, trend momentum and liquidity sweep each have an isolated,
  pre-registered parameter-stability validation path. This creates no live
  authority: the normal promotion audit still requires real OOS evidence.
- Read-only MT5 calls retry narrowly defined transient IPC failures; order sends
  remain non-retrying so an uncertain response cannot duplicate an order.
- Weekly reports compare resolved rejected plans by gate, including Claude
  vetoes, so claims that a filter adds value can be measured rather than assumed.
- The dashboard now displays the armed Experimental Live contract values
  directly instead of stale hard-coded risk text.
