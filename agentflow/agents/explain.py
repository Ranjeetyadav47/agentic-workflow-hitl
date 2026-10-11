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
    "You write one-paragraph notes for an accounts-payable reviewer working a queue. They read the "
    "first few words and decide whether to open the invoice now or later, so lead with the action.\n"
    "Write exactly one paragraph in three parts, in this order:\n"
    "1. The action and the identity: 'Review INV-1234 - Vendor Name, 12,345.00 INR.' Use 'Rejected' "
    "instead of 'Review' when the decision is reject.\n"
    "2. Why it was held: 'Held because: <the reasons, verbatim>.' Use 'Reason:' when the decision is reject.\n"
    "3. The single check that clears the first fired rule: 'Before approving: <check>.' Use 'Before closing:' "
    "when the decision is reject. FIRED_RULES is in policy order, so the first entry set the decision; "
    "name the check that addresses that rule specifically, not a generic instruction to compare against the PO.\n"
    "Use only the facts in CONTEXT. Do not invent numbers, vendors or reasons. Do not recommend approval or "
    "rejection beyond what the reasons support. No greeting, no sign-off, no bullet points."
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
            # Rule names in policy order: the first one set the decision, so it decides which
            # check the note tells the reviewer to do first.
            "fired_rules": policy.fired_rules(),
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
