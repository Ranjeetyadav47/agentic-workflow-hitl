"""Explain agent: structured context -> a note a busy reviewer can act on in ten seconds.

Second and last place an LLM is used. It never decides anything; it summarises
decisions already made by the policy agent, and the reasons it is given are the
only facts it is allowed to use.
"""

from __future__ import annotations

import json

from ..llm.base import LLMClient, LLMResponse
from ..models import ExtractedInvoice, MatchResult, PolicyOutcome, ReviewerNote

SYSTEM = (
    "You write one-paragraph notes for an accounts-payable reviewer. Use only the facts in CONTEXT. "
    "State the invoice, vendor, amount, why it was routed, and the recommended next action. "
    "Do not invent numbers. Do not recommend approval or rejection beyond what the reasons support."
)


class ExplainAgent:
    name = "explain"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, inv: ExtractedInvoice, match: MatchResult, policy: PolicyOutcome) -> tuple[ReviewerNote, LLMResponse]:
        context = {
            "invoice_number": inv.invoice_number,
            "vendor_name": inv.vendor_name,
            "total_amount": inv.total_amount,
            "currency": inv.currency,
            "decision": policy.decision.value,
            "reasons": policy.reasons,
            "match_notes": match.notes,
        }
        resp = self.llm.complete(task="explain", system=SYSTEM, user="CONTEXT:\n" + json.dumps(context))
        summary = resp.content.strip() or "No summary generated."
        action = {
            "needs_human": "review",
            "reject": "confirm rejection",
            "auto_approve": "none",
        }[policy.decision.value]
        return ReviewerNote(summary=summary, recommended_action=action), resp
