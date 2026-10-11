"""Match agent: three-way match of invoice vs purchase order vs goods receipt.

No LLM here. Everything this agent reports is a fact looked up in the ERP, so a
reviewer can verify each line of the result by hand.
"""

from __future__ import annotations

from ..models import ExtractedInvoice, MatchResult
from ..store import Store
from ..tools.erp import ERP


class MatchAgent:
    name = "match"

    def __init__(self, erp: ERP, store: Store, price_tolerance_pct: float):
        self.erp = erp
        self.store = store
        self.price_tolerance_pct = price_tolerance_pct

    def run(self, inv: ExtractedInvoice) -> MatchResult:
        notes: list[str] = []
        po = self.erp.get_po(inv.po_number)
        vendor_by_name = self.erp.find_vendor_by_name(inv.vendor_name)

        # History of the vendor *named on the invoice*, not the PO's vendor: the question the
        # policy agent asks is "have we ever paid this counterparty before?".
        prior = vendor_by_name.prior_invoices if vendor_by_name else None
        if prior == 0:
            notes.append(f"no settled invoices on record for {inv.vendor_name!r}")

        duplicate_of = None
        if inv.vendor_name and inv.invoice_number:
            duplicate_of = self.store.find_ledger_duplicate(inv.vendor_name, inv.invoice_number)
            if duplicate_of:
                notes.append(f"invoice {inv.invoice_number} already posted as {duplicate_of}")

        if po is None:
            notes.append(f"no purchase order found for {inv.po_number!r}")
            return MatchResult(
                po_found=False,
                vendor_active=vendor_by_name.active if vendor_by_name else None,
                vendor_matches_po=None,
                quantity_ok=None,
                max_price_variance_pct=None,
                received_total=None,
                po_total=None,
                duplicate_of=duplicate_of,
                vendor_prior_invoices=prior,
                notes=notes,
            )

        po_vendor = self.erp.get_vendor(po.vendor_id)
        vendor_matches_po = bool(vendor_by_name and po_vendor and vendor_by_name.vendor_id == po_vendor.vendor_id)
        if not vendor_matches_po:
            notes.append(f"invoice vendor {inv.vendor_name!r} does not match PO vendor {po_vendor.name if po_vendor else None!r}")
        vendor_active = po_vendor.active if po_vendor else None
        if vendor_active is False:
            notes.append(f"vendor {po_vendor.name} is inactive")

        receipt = self.erp.get_receipt(po.po_number)
        received = receipt.received if receipt else {}
        po_lines = {l.sku: l for l in po.lines}

        quantity_ok = True
        max_var = 0.0
        received_total = 0.0
        for li in inv.line_items:
            po_line = po_lines.get(li.sku)
            if po_line is None:
                quantity_ok = False
                notes.append(f"{li.sku} not on PO")
                continue
            got = received.get(li.sku, 0.0)
            if li.quantity > got + 1e-9:
                quantity_ok = False
                notes.append(f"{li.sku}: invoiced {li.quantity:g}, received {got:g}")
            received_total += min(li.quantity, got) * po_line.unit_price
            if po_line.unit_price > 0:
                var = abs(li.unit_price - po_line.unit_price) / po_line.unit_price * 100.0
                if var > max_var:
                    max_var = var
                if var > self.price_tolerance_pct:
                    notes.append(f"{li.sku}: unit price {li.unit_price:,.2f} vs PO {po_line.unit_price:,.2f} ({var:.1f}%)")

        if not inv.line_items:
            quantity_ok = False
            notes.append("no line items to match")

        return MatchResult(
            po_found=True,
            vendor_active=vendor_active,
            vendor_matches_po=vendor_matches_po,
            quantity_ok=quantity_ok,
            max_price_variance_pct=round(max_var, 2),
            received_total=round(received_total, 2),
            po_total=po.total,
            duplicate_of=duplicate_of,
            vendor_prior_invoices=prior,
            notes=notes,
        )
