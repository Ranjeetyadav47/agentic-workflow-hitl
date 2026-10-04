"""Reviewer notes: shape, and whether the recommended check matches the rule that fired.

These go through the real PolicyAgent rather than a hand-built PolicyOutcome, so they
cover the whole chain: rule fires -> trace -> fired_rules() -> explain context -> note.
"""

import unittest

from agentflow.agents.explain import ExplainAgent
from agentflow.agents.policy import PolicyAgent
from agentflow.config import PolicyConfig
from agentflow.llm.mock import MockLLM
from agentflow.models import Decision, ExtractedInvoice, LineItem, MatchResult


def _inv(total=50_000.0, confidence=1.0, issues=None):
    return ExtractedInvoice(
        invoice_number="INV-2005", vendor_name="Indigo Valve Works", po_number="PO-5001",
        total_amount=total, currency="INR",
        line_items=[LineItem("SKU-1", "thing", 1, total)], confidence=confidence, issues=issues or [],
    )


def _match(**kw):
    base = dict(po_found=True, vendor_active=True, vendor_matches_po=True, quantity_ok=True,
                max_price_variance_pct=0.0, received_total=50_000.0, po_total=50_000.0, duplicate_of=None,
                vendor_prior_invoices=24, notes=[])
    base.update(kw)
    return MatchResult(**base)


class ExplainNoteTests(unittest.TestCase):
    def setUp(self):
        self.policy = PolicyAgent(PolicyConfig())
        self.explain = ExplainAgent(MockLLM())

    def _note(self, inv, match):
        outcome = self.policy.run(inv, match)
        note, _ = self.explain.run(inv, match, outcome)
        return outcome, note.summary

    def test_review_note_leads_with_the_action_then_identity(self):
        _, summary = self._note(_inv(), _match(po_found=False))
        self.assertTrue(summary.startswith("Review INV-2005 — Indigo Valve Works, 50,000.00 INR."), summary)
        self.assertIn("Held because:", summary)
        self.assertIn("Before approving:", summary)

    def test_rejection_note_says_rejected_not_review(self):
        _, summary = self._note(_inv(), _match(vendor_active=False))
        self.assertTrue(summary.startswith("Rejected INV-2005"), summary)
        self.assertIn("Reason:", summary)
        self.assertIn("Before closing:", summary)

    def test_check_is_specific_to_the_rule_that_fired(self):
        """A new-vendor flag must not send the reviewer to paperwork that already matches."""
        outcome, summary = self._note(_inv(), _match(vendor_prior_invoices=0))
        self.assertEqual(outcome.decision, Decision.NEEDS_HUMAN)
        self.assertIn("onboarding record and bank details", summary)
        self.assertNotIn("GRN", summary)

    def test_quantity_flag_points_at_the_po_and_grn(self):
        _, summary = self._note(_inv(), _match(quantity_ok=False))
        self.assertIn("match the flagged lines against the PO and the GRN", summary)

    def test_first_fired_rule_wins_when_several_fire(self):
        """Rules fire in policy order, so the check follows the rule that set the decision."""
        outcome, summary = self._note(_inv(total=900_000.0), _match(vendor_prior_invoices=0, quantity_ok=False))
        self.assertEqual(outcome.fired_rules()[0], "first_invoice_from_vendor")
        self.assertIn("onboarding record and bank details", summary)

    def test_every_policy_rule_has_a_check(self):
        """A rule without a mapped check silently degrades to generic advice."""
        from agentflow.llm.mock import _CHECKS
        all_rules = [t.split(":", 1)[0] for t in self.policy.run(_inv(), _match()).rule_trace
                     if not t.startswith("default")]
        self.assertEqual([r for r in all_rules if r not in _CHECKS], [])

    def test_unmapped_rules_fall_back_instead_of_crashing(self):
        mock = MockLLM()
        summary = mock._explain('{"invoice_number": "INV-1", "decision": "needs_human",'
                                ' "reasons": ["something new"], "fired_rules": ["a_rule_added_later"]}')
        self.assertIn("work through the flagged reasons", summary)

    def test_unparseable_context_does_not_pretend_to_summarise(self):
        summary = MockLLM()._explain("not json at all")
        self.assertTrue(summary.startswith("Review manually"), summary)


class FiredRulesTests(unittest.TestCase):
    def test_fired_rules_are_in_policy_order_and_exclude_passes(self):
        outcome = PolicyAgent(PolicyConfig()).run(_inv(), _match(po_found=False, vendor_prior_invoices=0))
        self.assertEqual(outcome.fired_rules(), ["first_invoice_from_vendor", "po_missing"])

    def test_clean_invoice_fires_nothing(self):
        self.assertEqual(PolicyAgent(PolicyConfig()).run(_inv(), _match()).fired_rules(), [])


if __name__ == "__main__":
    unittest.main()
