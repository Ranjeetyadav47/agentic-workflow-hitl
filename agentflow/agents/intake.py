"""Intake agent: raw invoice text -> ExtractedInvoice.

This is the one place an LLM reads unstructured input. The output is validated
before anything downstream sees it; a model that returns garbage raises
PermanentError, and the orchestrator decides what that means for the run.
"""

from __future__ import annotations

import json

from ..llm.base import LLMClient, LLMResponse
from ..models import ExtractedInvoice, LineItem, PermanentError

SYSTEM = (
    "You extract structured fields from supplier invoices. Return only JSON with keys: "
    "invoice_number, vendor_name, po_number, total_amount (number), currency, "
    "line_items (list of {sku, description, quantity, unit_price}), confidence (0-1), issues (list of strings). "
    "Use null for anything you cannot find. Never guess an amount."
)


class IntakeAgent:
    name = "intake"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, raw_text: str) -> tuple[ExtractedInvoice, LLMResponse]:
        resp = self.llm.complete(
            task="extract", system=SYSTEM, user=f"INVOICE TEXT:\n{raw_text}", json_mode=True
        )
        try:
            data = json.loads(resp.content)
        except json.JSONDecodeError as e:
            raise PermanentError(f"intake: model returned non-JSON: {resp.content[:120]!r}") from e

        extracted = self._validate(data)
        return extracted, resp

    @staticmethod
    def _validate(d: dict) -> ExtractedInvoice:
        if not isinstance(d, dict):
            raise PermanentError("intake: JSON root is not an object")

        total = d.get("total_amount")
        if total is not None:
            try:
                total = float(total)
            except (TypeError, ValueError):
                raise PermanentError(f"intake: total_amount not numeric: {total!r}")
            if total < 0:
                raise PermanentError("intake: negative total")

        items = []
        for li in d.get("line_items") or []:
            try:
                items.append(
                    LineItem(
                        sku=str(li["sku"]).upper(),
                        description=str(li.get("description", "")),
                        quantity=float(li["quantity"]),
                        unit_price=float(li["unit_price"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                # A malformed line is an issue, not a crash: the policy layer will see it.
                continue

        issues = [str(i) for i in (d.get("issues") or [])]
        if items and total is not None:
            line_sum = round(sum(i.total for i in items), 2)
            if abs(line_sum - total) > 0.01:
                issues.append(f"line_sum_mismatch:{line_sum}!={total}")

        return ExtractedInvoice(
            invoice_number=(d.get("invoice_number") or None),
            vendor_name=(d.get("vendor_name") or None),
            po_number=(d.get("po_number") or None),
            total_amount=total,
            currency=(d.get("currency") or None),
            line_items=items,
            confidence=float(d.get("confidence") or 0.0),
            issues=issues,
        )
