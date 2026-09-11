# Retired trading sections

The Eightcap live book is intentionally limited to:

- `section_six_gold_m5`
- `section_ten_gold_m1`

Every numbered strategy other than S6 and S10 is retired from real-money
decision-making. The implementations and their negative measurements remain in
Git as research evidence. They are not candidates for the new human-context
decision process and cannot regain live permission through a weight, an
`enabled` flag, or a generic confluence vote: `live_enabled_modules` is the
single explicit spending allowlist.

`human_context_decision` is a new shadow-only experiment. It is constructed so
its decisions can be replayed and journalled, but it is deliberately absent
from `live_enabled_modules`. A 30-day result is diagnostic evidence, not live
promotion authority.

Deleting the retired source would erase failed hypotheses, break attribution
for historical journal rows and make it easier to repeat old experiments under
new names. Retirement therefore means no live vote, not destruction of the
audit trail.
