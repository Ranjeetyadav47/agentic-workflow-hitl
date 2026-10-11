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
                max_price_variance_pct=0.0, received_total=50_000.0, po_total=50_000.0, duplicate_of=None,
                vendor_prior_invoices=24, notes=[])
    base.update(kw)
    return MatchResult(**base)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        # Test the thresholds we actually ship, so these tests document the real policy.
        self.cfg = PolicyConfig()
        self.agent = PolicyAgent(self.cfg)

    def test_clean_invoice_auto_approves(self):
        out = self.agent.run(_inv(), _match())
        self.assertEqual(out.decision, Decision.AUTO_APPROVE)
        self.assertEqual(len(out.rule_trace), 11)   # every rule evaluated and logged

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
        tol = self.cfg.price_tolerance_pct
        self.assertEqual(self.agent.run(_inv(), _match(max_price_variance_pct=tol)).decision, Decision.AUTO_APPROVE)
        self.assertEqual(self.agent.run(_inv(), _match(max_price_variance_pct=tol + 0.01)).decision,
                         Decision.NEEDS_HUMAN)

    def test_authority_limit_boundary(self):
        limit = self.cfg.auto_approve_limit
        self.assertEqual(self.agent.run(_inv(total=limit), _match()).decision, Decision.AUTO_APPROVE)
        self.assertEqual(self.agent.run(_inv(total=limit + 0.01), _match()).decision, Decision.NEEDS_HUMAN)

    def test_generator_thresholds_track_the_policy(self):
        """The labelled set is only ground truth if it was built against the shipped thresholds."""
        from agentflow.data.generate import AUTO_APPROVE_LIMIT, PRICE_TOLERANCE_PCT
        self.assertEqual(AUTO_APPROVE_LIMIT, self.cfg.auto_approve_limit)
        self.assertEqual(PRICE_TOLERANCE_PCT, self.cfg.price_tolerance_pct)

    def test_first_invoice_from_vendor_goes_to_human(self):
        """Clean paperwork is not enough on its own if we have never paid this vendor."""
        out = self.agent.run(_inv(), _match(vendor_prior_invoices=0))
        self.assertEqual(out.decision, Decision.NEEDS_HUMAN)
        self.assertIn("first_invoice_from_vendor: FIRED", out.rule_trace)
        self.assertIn("first invoice from this vendor: no settled history on record", out.reasons)

    def test_established_vendor_is_not_flagged_as_new(self):
        out = self.agent.run(_inv(), _match(vendor_prior_invoices=1))
        self.assertEqual(out.decision, Decision.AUTO_APPROVE)
        self.assertIn("first_invoice_from_vendor: pass", out.rule_trace)

    def test_unidentified_vendor_does_not_fire_the_new_vendor_rule(self):
        """None means 'could not identify the vendor', not 'brand new'; other rules catch it."""
        out = self.agent.run(_inv(), _match(vendor_prior_invoices=None))
        self.assertIn("first_invoice_from_vendor: pass", out.rule_trace)

    def test_inactive_new_vendor_still_rejects_not_reviews(self):
        out = self.agent.run(_inv(), _match(vendor_active=False, vendor_prior_invoices=0))
        self.assertEqual(out.decision, Decision.REJECT)

    def test_all_reasons_collected_not_just_first(self):
        out = self.agent.run(_inv(total=300_000.0), _match(quantity_ok=False, max_price_variance_pct=9.0))
        self.assertEqual(out.decision, Decision.NEEDS_HUMAN)
        self.assertEqual(len(out.reasons), 3)


if __name__ == "__main__":
    unittest.main()
