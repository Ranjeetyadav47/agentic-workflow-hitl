import unittest

from agentflow.agents.policy import PolicyAgent
from agentflow.config import PolicyConfig
from agentflow.models import Decision, ExtractedInvoice, LineItem, MatchResult


def _inv(total=50_000.0, confidence=1.0, issues=None):
    return ExtractedInvoice(
        invoice_number="INV-1", vendor_name="Acme", po_number="PO-1", total_amount=total, currency="INR",
        line_items=[LineItem("SKU-1", "thing", 1, total)], confidence=confidence, issues=issues or [],
    )


def _match(**kw):
    base = dict(po_found=True, vendor_active=True, vendor_matches_po=True, quantity_ok=True,
                max_price_variance_pct=0.0, received_total=50_000.0, po_total=50_000.0, duplicate_of=None, notes=[])
    base.update(kw)
    return MatchResult(**base)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.agent = PolicyAgent(PolicyConfig(auto_approve_limit=250_000.0, price_tolerance_pct=2.0))

    def test_clean_invoice_auto_approves(self):
        out = self.agent.run(_inv(), _match())
        self.assertEqual(out.decision, Decision.AUTO_APPROVE)
        self.assertEqual(len(out.rule_trace), 10)   # every rule evaluated and logged

    def test_duplicate_rejects_even_if_everything_else_fails(self):
        out = self.agent.run(_inv(total=None, confidence=0.0), _match(po_found=False, duplicate_of="LED-1"))
        self.assertEqual(out.decision, Decision.REJECT)
        self.assertIn("duplicate_invoice: FIRED", out.rule_trace)

    def test_inactive_vendor_rejects(self):
        self.assertEqual(self.agent.run(_inv(), _match(vendor_active=False)).decision, Decision.REJECT)

    def test_unreadable_invoice_goes_to_human_not_dlq(self):
        out = self.agent.run(_inv(total=None, confidence=0.25, issues=["total_missing"]), _match(po_found=False))
        self.assertEqual(out.decision, Decision.NEEDS_HUMAN)
        self.assertTrue(out.reasons[0].startswith("invoice could not be read reliably"))

    def test_missing_po_goes_to_human(self):
        self.assertEqual(self.agent.run(_inv(), _match(po_found=False)).decision, Decision.NEEDS_HUMAN)

    def test_price_variance_boundary(self):
        self.assertEqual(self.agent.run(_inv(), _match(max_price_variance_pct=2.0)).decision, Decision.AUTO_APPROVE)
        self.assertEqual(self.agent.run(_inv(), _match(max_price_variance_pct=2.01)).decision, Decision.NEEDS_HUMAN)

    def test_authority_limit_boundary(self):
        self.assertEqual(self.agent.run(_inv(total=250_000.0), _match()).decision, Decision.AUTO_APPROVE)
        self.assertEqual(self.agent.run(_inv(total=250_000.01), _match()).decision, Decision.NEEDS_HUMAN)

    def test_all_reasons_collected_not_just_first(self):
        out = self.agent.run(_inv(total=300_000.0), _match(quantity_ok=False, max_price_variance_pct=9.0))
        self.assertEqual(out.decision, Decision.NEEDS_HUMAN)
        self.assertEqual(len(out.reasons), 3)


if __name__ == "__main__":
    unittest.main()
