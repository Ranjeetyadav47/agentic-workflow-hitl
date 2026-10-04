"""Policy agent: deterministic rules -> Decision.

Rules are evaluated in a fixed order and every evaluation is recorded in `rule_trace`,
so the answer to "why did this get approved?" is a list you can read, not a prompt.
The first rule that fires decides; later rules are still evaluated and logged so the
reviewer sees the full picture.
"""

from __future__ import annotations

from ..config import PolicyConfig
from ..models import Decision, ExtractedInvoice, MatchResult, PolicyOutcome


class PolicyAgent:
    name = "policy"

    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg

    def run(self, inv: ExtractedInvoice, match: MatchResult) -> PolicyOutcome:
        trace: list[str] = []
        reasons: list[str] = []
        decision: Decision | None = None

        def fire(rule: str, fired: bool, outcome: Decision, reason: str) -> None:
            nonlocal decision
            trace.append(f"{rule}: {'FIRED' if fired else 'pass'}")
            if fired:
                reasons.append(reason)
                if decision is None:
                    decision = outcome

        # 1. Hard stops first.
        fire("duplicate_invoice", match.duplicate_of is not None, Decision.REJECT,
             f"duplicate of ledger entry {match.duplicate_of}")
        fire("vendor_inactive", match.vendor_active is False, Decision.REJECT, "vendor is inactive")

        # 2. Can we trust the extraction at all?
        unparseable = inv.total_amount is None or inv.invoice_number is None or inv.confidence < 0.5
        fire("extraction_incomplete", unparseable, Decision.NEEDS_HUMAN,
             "invoice could not be read reliably: " + ", ".join(inv.issues or ["low confidence"]))

        # 3. Counterparty: a vendor we have never paid before gets a human look, however
        # clean the paperwork is. A clean first invoice is exactly what an invoice-fraud
        # attempt looks like, and the three-way match cannot see it.
        first_invoice = match.vendor_prior_invoices == 0
        fire("first_invoice_from_vendor", first_invoice, Decision.NEEDS_HUMAN,
             "first invoice from this vendor: no settled history on record")

        # 4. Three-way match.
        fire("po_missing", not match.po_found, Decision.NEEDS_HUMAN, "no matching purchase order")
        fire("vendor_mismatch", match.po_found and match.vendor_matches_po is False, Decision.NEEDS_HUMAN,
             "invoice vendor differs from PO vendor")
        fire("quantity_mismatch", match.po_found and match.quantity_ok is False, Decision.NEEDS_HUMAN,
             "invoiced quantity exceeds goods received or SKU not on PO")
        over_tol = match.max_price_variance_pct is not None and match.max_price_variance_pct > self.cfg.price_tolerance_pct
        fire("price_variance", match.po_found and over_tol, Decision.NEEDS_HUMAN,
             f"unit price variance {match.max_price_variance_pct}% exceeds {self.cfg.price_tolerance_pct}% tolerance")
        line_mismatch = any(i.startswith("line_sum_mismatch") for i in inv.issues)
        fire("total_vs_lines", line_mismatch, Decision.NEEDS_HUMAN, "invoice total does not equal sum of line items")

        # 5. Authority limit.
        over_limit = inv.total_amount is not None and inv.total_amount > self.cfg.auto_approve_limit
        fire("over_auto_approve_limit", over_limit, Decision.NEEDS_HUMAN,
             f"amount exceeds auto-approval limit of {self.cfg.auto_approve_limit:,.0f} {self.cfg.currency}")

        if decision is None:
            decision = Decision.AUTO_APPROVE
            reasons.append("three-way match clean and within authority limit")
            trace.append("default: AUTO_APPROVE")

        return PolicyOutcome(decision=decision, reasons=reasons, rule_trace=trace)
