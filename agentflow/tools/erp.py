"""A tiny read-only ERP: vendors, purchase orders, goods receipts, loaded from JSON.

Agents call this instead of an LLM for anything that has a right answer. In a real
deployment this is the SAP/Oracle/NetSuite adapter; the interface would not change.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Vendor:
    vendor_id: str
    name: str
    active: bool
    prior_invoices: int = 0     # invoices already settled with this vendor; 0 = never traded with


@dataclass(frozen=True)
class POLine:
    sku: str
    quantity: float
    unit_price: float


@dataclass(frozen=True)
class PurchaseOrder:
    po_number: str
    vendor_id: str
    currency: str
    lines: tuple[POLine, ...]

    @property
    def total(self) -> float:
        return round(sum(l.quantity * l.unit_price for l in self.lines), 2)


@dataclass(frozen=True)
class Receipt:
    po_number: str
    received: dict[str, float]      # sku -> quantity received so far


class ERP:
    def __init__(self, data_dir: str):
        self.vendors: dict[str, Vendor] = {}
        self.vendors_by_name: dict[str, Vendor] = {}
        self.pos: dict[str, PurchaseOrder] = {}
        self.receipts: dict[str, Receipt] = {}
        self._load(data_dir)

    def _load(self, data_dir: str) -> None:
        with open(os.path.join(data_dir, "vendors.json")) as f:
            for v in json.load(f):
                # Missing prior_invoices means we cannot prove a trading history, so it reads
                # as 0 and the policy agent routes the invoice to a human. Fail safe, not open.
                vendor = Vendor(v["vendor_id"], v["name"], bool(v["active"]), int(v.get("prior_invoices", 0)))
                self.vendors[vendor.vendor_id] = vendor
                self.vendors_by_name[_norm(vendor.name)] = vendor
        with open(os.path.join(data_dir, "purchase_orders.json")) as f:
            for p in json.load(f):
                po = PurchaseOrder(
                    p["po_number"], p["vendor_id"], p["currency"],
                    tuple(POLine(l["sku"], float(l["quantity"]), float(l["unit_price"])) for l in p["lines"]),
                )
                self.pos[po.po_number] = po
        with open(os.path.join(data_dir, "receipts.json")) as f:
            for r in json.load(f):
                self.receipts[r["po_number"]] = Receipt(r["po_number"], {k: float(v) for k, v in r["received"].items()})

    def get_po(self, po_number: Optional[str]) -> Optional[PurchaseOrder]:
        return self.pos.get(po_number.upper()) if po_number else None

    def get_receipt(self, po_number: str) -> Optional[Receipt]:
        return self.receipts.get(po_number.upper())

    def get_vendor(self, vendor_id: str) -> Optional[Vendor]:
        return self.vendors.get(vendor_id)

    def find_vendor_by_name(self, name: Optional[str]) -> Optional[Vendor]:
        return self.vendors_by_name.get(_norm(name)) if name else None


def _norm(s: str) -> str:
    return " ".join(s.lower().replace(".", "").split())
