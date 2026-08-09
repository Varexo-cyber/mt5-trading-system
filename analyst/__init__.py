"""An advisory reading of the whole account, with no authority over it.

Deliberately not imported by the trading path. `analyst.review` returns prose;
there is no verdict, no `Reason`, and nothing downstream acts on it. That
separation is what makes an open-ended reasoner safe to point at a live
account — a gate has to be predictable, an analyst has to be free to say
something nobody anticipated, and one component cannot be both.
"""

from analyst.evidence import Evidence, TradeFact, dominant_refusal, gather
from analyst.review import Assessment, Finding, analyse

__all__ = [
    "Assessment",
    "Evidence",
    "Finding",
    "TradeFact",
    "analyse",
    "dominant_refusal",
    "gather",
]
