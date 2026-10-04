import json
import unittest

from agentflow.agents.intake import IntakeAgent
from agentflow.llm.mock import MockLLM
from agentflow.models import PermanentError

TEMPLATE_A = """TAX INVOICE
Invoice No: INV-2041
Date: 2026-07-14
Vendor: Meridian Chemicals Pvt Ltd
PO Number: PO-5123
Currency: INR
Items:
- SKU-114 | Sodium Hypochlorite 12% | qty 40 | unit 1,250.00
- SKU-209 | Drum 200L HDPE | qty 40 | unit 850.00
Total: 84,000.00
"""

TEMPLATE_C_LAKH = """INVOICE # INV-2043    PO # PO-5125    DT 2026-03-02
Supplier : Arjun Logistics
--------------------------------------------------------
SKU      DESC                        QTY          RATE
SKU-410  Freight Mumbai-Pune          12      9,500.00
--------------------------------------------------------
GRAND TOTAL (INR)  1,14,000.00
"""


class MockExtractionTests(unittest.TestCase):
    def test_parses_key_value_layout(self):
        inv, resp = IntakeAgent(MockLLM()).run(TEMPLATE_A)
        self.assertEqual(inv.invoice_number, "INV-2041")
        self.assertEqual(inv.po_number, "PO-5123")
        self.assertEqual(inv.vendor_name, "Meridian Chemicals Pvt Ltd")
        self.assertEqual(inv.total_amount, 84000.0)
        self.assertEqual(len(inv.line_items), 2)
        self.assertEqual(inv.issues, [])
        self.assertGreater(resp.prompt_tokens, 0)

    def test_indian_grouping_parsed_when_normaliser_on(self):
        inv, _ = IntakeAgent(MockLLM(normalise_indian_numbers=True)).run(TEMPLATE_C_LAKH)
        self.assertEqual(inv.total_amount, 114000.0)
        self.assertEqual(inv.issues, [])

    def test_indian_grouping_bug_reproduced_when_normaliser_off(self):
        """The regression the eval harness caught: a plausible but wrong total."""
        inv, _ = IntakeAgent(MockLLM(normalise_indian_numbers=False)).run(TEMPLATE_C_LAKH)
        self.assertEqual(inv.total_amount, 14000.0)
        self.assertTrue(any(i.startswith("line_sum_mismatch") for i in inv.issues))

    def test_garbled_text_yields_low_confidence_not_crash(self):
        inv, _ = IntakeAgent(MockLLM()).run("#%&*@~\nINVOlCE INV-<~\\2022\nT0tal due .... #%&\n")
        self.assertIsNone(inv.total_amount)
        self.assertLess(inv.confidence, 0.5)
        self.assertIn("total_missing", inv.issues)

    def test_non_json_model_output_is_permanent_error(self):
        class BadLLM:
            def complete(self, task, system, user, json_mode=False):
                from agentflow.llm.base import LLMResponse
                return LLMResponse(content="Sure! Here is the invoice: ...")

        with self.assertRaises(PermanentError):
            IntakeAgent(BadLLM()).run(TEMPLATE_A)

    def test_negative_total_rejected(self):
        class NegLLM:
            def complete(self, task, system, user, json_mode=False):
                from agentflow.llm.base import LLMResponse
                return LLMResponse(content=json.dumps({"invoice_number": "INV-1", "total_amount": -5}))

        with self.assertRaises(PermanentError):
            IntakeAgent(NegLLM()).run("x")


if __name__ == "__main__":
    unittest.main()
