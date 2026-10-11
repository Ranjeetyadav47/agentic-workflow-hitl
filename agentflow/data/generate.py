"""Seeded synthetic data: vendors, purchase orders, goods receipts and invoice texts.

Every invoice is generated from a named scenario, so its correct decision is known
by construction. That label is the ground truth the eval harness scores against.
All names, SKUs and numbers are invented. Nothing here comes from any employer.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Iterator

from ..config import PolicyConfig

# scenario -> expected decision
SCENARIOS: dict[str, str] = {
    "clean": "auto_approve",
    "price_within_tolerance": "auto_approve",
    "over_limit": "needs_human",
    "price_variance": "needs_human",
    "quantity_mismatch": "needs_human",
    "missing_po": "needs_human",
    "vendor_mismatch": "needs_human",
    "garbled": "needs_human",
    "new_vendor": "needs_human",
    "vendor_inactive": "reject",
    "duplicate": "reject",
}

# How many of each in the default 120-invoice set.
MIX: dict[str, int] = {
    "clean": 32,
    "price_within_tolerance": 10,
    "over_limit": 10,
    "price_variance": 10,
    "quantity_mismatch": 12,
    "missing_po": 8,
    "vendor_mismatch": 8,
    "garbled": 6,
    "new_vendor": 8,
    "vendor_inactive": 8,
    "duplicate": 8,
}

VENDOR_NAMES = [
    "Meridian Chemicals Pvt Ltd", "Nova Packaging Ltd", "Arjun Logistics", "Kestrel Instruments",
    "Saffron Agro Supplies", "Blue Lotus Engineering", "Orion Lab Consumables", "Tulsi Transport Co",
    "Harbour Steel Works", "Pinnacle Office Systems", "Greenfield Seeds Ltd", "Vector Safety Gear",
    "Indigo Valve Works", "Marigold Agritech Pvt Ltd",
]
INACTIVE_VENDORS = {"Harbour Steel Works", "Vector Safety Gear"}
# Onboarded but never paid: these are the only vendors the new_vendor scenario draws from,
# so "first invoice from this vendor" stays a labelled scenario rather than a surprise.
NEW_VENDORS = {"Indigo Valve Works", "Marigold Agritech Pvt Ltd"}

CATALOGUE = [
    ("SKU-114", "Sodium Hypochlorite 12%", 1250.00), ("SKU-209", "Drum 200L HDPE", 850.00),
    ("SKU-301", "Corrugated box 5-ply", 42.50), ("SKU-302", "Tape roll 48mm", 65.00),
    ("SKU-410", "Freight Mumbai-Pune", 9500.00), ("SKU-512", "pH meter probe", 7800.00),
    ("SKU-613", "Lab gloves nitrile (box)", 540.00), ("SKU-704", "Pallet wrap 23mu", 310.00),
    ("SKU-815", "Safety goggles", 220.00), ("SKU-906", "Toner cartridge 85A", 3900.00),
    ("SKU-917", "A4 paper (ream)", 285.00), ("SKU-928", "Calibration service (hr)", 2400.00),
    ("SKU-939", "Steel bracket 40mm", 160.00), ("SKU-940", "Seed tray 104-cell", 48.00),
]

# Scenario design reads the real policy thresholds rather than copying them: a scenario
# labelled auto_approve must stay under the limit the policy agent actually enforces, and
# a duplicated constant is a mislabelled test set waiting to happen.
AUTO_APPROVE_LIMIT = PolicyConfig().auto_approve_limit
PRICE_TOLERANCE_PCT = PolicyConfig().price_tolerance_pct

# Template C renders totals with Indian digit grouping, and the parser bug it exposes only
# shows up at a lakh or more. If the authority limit ever drops below this, template C can
# no longer carry the bug, so fail loudly instead of quietly generating a weaker test set.
LAKH = 100_000.0
if AUTO_APPROVE_LIMIT * 0.95 <= LAKH * 1.1:
    raise ValueError(
        f"auto_approve_limit {AUTO_APPROVE_LIMIT:,.0f} is too low for the template C "
        f"lakh-format scenario; raise it above {LAKH * 1.1 / 0.95:,.0f} or redesign template C"
    )


@dataclass
class _Ctx:
    rng: random.Random
    vendors: list[dict]
    pos: list[dict]
    receipts: list[dict]
    invoices: list[dict]
    inv_seq: int = 2000
    po_seq: int = 5000


def generate(out_dir: str = "data", seed: int = 42, mix: dict[str, int] | None = None) -> dict[str, int]:
    rng = random.Random(seed)
    mix = dict(mix or MIX)
    ctx = _Ctx(rng=rng, vendors=[], pos=[], receipts=[], invoices=[])

    for i, name in enumerate(VENDOR_NAMES):
        ctx.vendors.append({
            "vendor_id": f"V{i+1:03d}",
            "name": name,
            "active": name not in INACTIVE_VENDORS,
            "prior_invoices": 0 if name in NEW_VENDORS else rng.randint(6, 180),
        })

    # Order matters for duplicates: originals first. Shuffle the rest for realism.
    order: list[str] = []
    for scenario, n in mix.items():
        order += [scenario] * n
    rng.shuffle(order)
    # Duplicates need an original that was auto-approved earlier in the stream.
    order = [s for s in order if s != "duplicate"]
    dup_n = mix.get("duplicate", 0)
    originals: list[dict] = []

    for scenario in order:
        inv = _make_invoice(ctx, scenario)
        ctx.invoices.append(inv)
        if scenario == "clean":
            originals.append(inv)

    # Append duplicates of random clean originals at the end of the stream.
    for _ in range(dup_n):
        src = rng.choice(originals)
        dup = dict(src)
        dup["invoice_id"] = f"inv-{len(ctx.invoices)+1:04d}"
        dup["scenario"] = "duplicate"
        dup["expected_decision"] = SCENARIOS["duplicate"]
        dup["duplicate_of"] = src["invoice_id"]
        ctx.invoices.append(dup)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vendors.json"), "w") as f:
        json.dump(ctx.vendors, f, indent=2)
    with open(os.path.join(out_dir, "purchase_orders.json"), "w") as f:
        json.dump(ctx.pos, f, indent=2)
    with open(os.path.join(out_dir, "receipts.json"), "w") as f:
        json.dump(ctx.receipts, f, indent=2)
    with open(os.path.join(out_dir, "invoices.jsonl"), "w") as f:
        for inv in ctx.invoices:
            f.write(json.dumps(inv) + "\n")

    counts: dict[str, int] = {}
    for inv in ctx.invoices:
        counts[inv["scenario"]] = counts.get(inv["scenario"], 0) + 1
    return counts


def load_invoices(data_dir: str = "data") -> Iterator[dict]:
    with open(os.path.join(data_dir, "invoices.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---- scenario construction ----------------------------------------------------------
def _make_invoice(ctx: _Ctx, scenario: str) -> dict:
    rng = ctx.rng
    # "Established" excludes the never-paid vendors so only the new_vendor scenario can draw them.
    established = [v for v in ctx.vendors if v["active"] and v["prior_invoices"] > 0]
    inactive = [v for v in ctx.vendors if not v["active"]]
    brand_new = [v for v in ctx.vendors if v["active"] and v["prior_invoices"] == 0]
    active = established
    if scenario == "vendor_inactive":
        vendor = rng.choice(inactive)
    elif scenario == "new_vendor":
        vendor = rng.choice(brand_new)
    else:
        vendor = rng.choice(established)

    # Template C uses Indian digit grouping; make a third of clean invoices land in
    # [1 lakh, limit) so the grouping bug is observable in the eval.
    template = rng.choice("ABC")
    if scenario == "over_limit":
        target = (AUTO_APPROVE_LIMIT * 1.1, AUTO_APPROVE_LIMIT * 3)
    elif scenario in ("clean", "price_within_tolerance") and template == "C":
        target = (110_000, AUTO_APPROVE_LIMIT * 0.95)
    else:
        target = (8_000, 95_000)

    lines = _po_lines(rng, target)
    ctx.po_seq += 1
    po_number = f"PO-{ctx.po_seq}"
    ctx.pos.append({
        "po_number": po_number, "vendor_id": vendor["vendor_id"], "currency": "INR",
        "lines": [{"sku": s, "quantity": q, "unit_price": p} for s, _, q, p in lines],
    })

    received = {s: q for s, _, q, _ in lines}
    inv_lines = [(s, d, q, p) for s, d, q, p in lines]
    inv_vendor_name = vendor["name"]
    inv_po = po_number

    if scenario == "quantity_mismatch":
        s, d, q, p = inv_lines[0]
        received[s] = max(0, q - rng.randint(1, max(1, int(q // 2))))
    elif scenario == "price_variance":
        # Well clear of the tolerance, so the label holds at any configured tolerance.
        s, d, q, p = inv_lines[0]
        over = rng.uniform(PRICE_TOLERANCE_PCT * 2, PRICE_TOLERANCE_PCT * 8) / 100.0
        inv_lines[0] = (s, d, q, round(p * (1 + over), 2))
    elif scenario == "price_within_tolerance":
        # Deliberately inside the tolerance band, never on its edge: these are labelled
        # auto_approve, so a rounding slip here would silently mislabel the test set.
        s, d, q, p = inv_lines[0]
        under = rng.uniform(PRICE_TOLERANCE_PCT * 0.1, PRICE_TOLERANCE_PCT * 0.8) / 100.0
        inv_lines[0] = (s, d, q, round(p * (1 + under), 2))
    elif scenario == "missing_po":
        inv_po = f"PO-{rng.randint(9000, 9999)}"
    elif scenario == "vendor_mismatch":
        other = rng.choice([v for v in active if v["vendor_id"] != vendor["vendor_id"]])
        inv_vendor_name = other["name"]

    ctx.receipts.append({"po_number": po_number, "received": received})

    ctx.inv_seq += 1
    invoice_number = f"INV-{ctx.inv_seq}"
    total = round(sum(q * p for _, _, q, p in inv_lines), 2)

    if scenario == "garbled":
        text = _garbled(rng, invoice_number)
        template = "D"
    else:
        text = _render(template, invoice_number, inv_po, inv_vendor_name, inv_lines, total, rng)

    return {
        "invoice_id": f"inv-{len(ctx.invoices)+1:04d}",
        "scenario": scenario,
        "expected_decision": SCENARIOS[scenario],
        "template": template,
        "invoice_number": invoice_number,
        "po_number": inv_po,
        "vendor_name": inv_vendor_name,
        "total": total,
        "text": text,
    }


def _po_lines(rng: random.Random, target: tuple[float, float]) -> list[tuple[str, str, int, float]]:
    """Pick 1-3 catalogue items and quantities so the total lands inside `target`."""
    lo, hi = target
    for _ in range(200):
        items = rng.sample(CATALOGUE, rng.randint(1, 3))
        lines = []
        for sku, desc, price in items:
            qty = rng.randint(1, 60)
            lines.append((sku, desc, qty, price))
        total = sum(q * p for _, _, q, p in lines)
        if lo <= total <= hi:
            return lines
    # Fallback: scale the first item to hit the range.
    sku, desc, price = rng.choice(CATALOGUE)
    qty = max(1, int(((lo + hi) / 2) // price))
    return [(sku, desc, qty, price)]


# ---- text rendering -----------------------------------------------------------------
def _fmt_western(x: float) -> str:
    return f"{x:,.2f}"


def _fmt_indian(x: float) -> str:
    """1234567.89 -> 12,34,567.89"""
    whole, frac = f"{x:.2f}".split(".")
    if len(whole) <= 3:
        return f"{whole}.{frac}"
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups) + "," + tail + "." + frac


def _render(template: str, inv_no: str, po: str, vendor: str, lines, total: float, rng: random.Random) -> str:
    date = f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}"
    if template == "A":
        body = "\n".join(f"- {s} | {d} | qty {q} | unit {_fmt_western(p)}" for s, d, q, p in lines)
        return (
            f"TAX INVOICE\nInvoice No: {inv_no}\nDate: {date}\nVendor: {vendor}\nPO Number: {po}\n"
            f"Currency: INR\nItems:\n{body}\nTotal: {_fmt_western(total)}\n"
        )
    if template == "B":
        body = "\n".join(f"{i+1}) {s} {d} x {q} @ {_fmt_western(p)}" for i, (s, d, q, p) in enumerate(lines))
        return (
            f"Hello, please find invoice {inv_no} dated {date} against your purchase order {po}.\n"
            f"From: {vendor}\nLine items:\n{body}\nAmount payable: INR {_fmt_western(total)}\nThanks.\n"
        )
    # template C: tabular, Indian digit grouping
    body = "\n".join(f"{s:<8} {d:<26} {q:>4}  {_fmt_indian(p):>12}" for s, d, q, p in lines)
    return (
        f"INVOICE # {inv_no}    PO # {po}    DT {date}\nSupplier : {vendor}\n"
        f"{'-'*56}\n{'SKU':<8} {'DESC':<26} {'QTY':>4}  {'RATE':>12}\n{body}\n{'-'*56}\n"
        f"GRAND TOTAL (INR)  {_fmt_indian(total)}\n"
    )


def _garbled(rng: random.Random, inv_no: str) -> str:
    """A scan that OCR mangled: fragments, no reliable total."""
    junk = "".join(rng.choice("#%&*@~^|/\\<>") for _ in range(rng.randint(8, 20)))
    frag = rng.choice(["Inv0ice", "INVOlCE", "lnvoice"])
    return f"{junk}\n{frag} {inv_no[:4]}{junk[:3]}{inv_no[4:]}\nT0tal due .... {junk}\n{junk}\n"
