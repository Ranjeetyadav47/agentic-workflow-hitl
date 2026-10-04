"""Deterministic stand-in for an LLM.

Why it exists: the workflow, retries, review queue and eval harness must be testable
without credentials or network, and the eval numbers must be reproducible by anyone
who clones the repo. The mock does the two jobs a real model does here:

  * "extract": read raw invoice text into structured fields (regex over known layouts)
  * "explain": write a short note for the human reviewer from structured context

It is deliberately imperfect in realistic ways (see `_parse_amount`), which is what
makes the evaluation harness worth having.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from ..models import TransientError
from .base import LLMResponse

_INVOICE_NO = re.compile(r"\b(INV-\d{3,})\b", re.I)
_PO_NO = re.compile(r"\b(PO-\d{3,})\b", re.I)
_VENDOR = re.compile(r"^(?:Vendor|From|Supplier)\s*:\s*(.+?)\s*$", re.I | re.M)
_CURRENCY = re.compile(r"\b(INR|USD|EUR)\b")
_TOTAL_LINE = re.compile(
    r"^(?:Total|Amount payable|GRAND TOTAL)\s*(?:\((?:INR|USD|EUR)\))?\s*:?\s*(?:INR|USD|EUR)?\s*([\d,]+(?:\.\d{1,2})?)\s*$",
    re.I | re.M,
)

# Three line-item layouts the synthetic generator produces.
_ITEM_A = re.compile(r"^- (SKU-\d+) \| (.+?) \| qty ([\d.]+) \| unit ([\d,]+(?:\.\d{1,2})?)\s*$", re.M)
_ITEM_B = re.compile(r"^\d+\) (SKU-\d+) (.+?) x ([\d.]+) @ ([\d,]+(?:\.\d{1,2})?)\s*$", re.M)
_ITEM_C = re.compile(r"^(SKU-\d+)\s{2,}(.+?)\s{2,}([\d.]+)\s{2,}([\d,]+(?:\.\d{1,2})?)\s*$", re.M)

_WESTERN = re.compile(r"^\d{1,3}(,\d{3})*(\.\d{1,2})?$")
_INDIAN = re.compile(r"^\d{1,2}(,\d{2})*,\d{3}(\.\d{1,2})?$")


class MockLLM:
    def __init__(self, normalise_indian_numbers: bool = True, fail_first_n_calls: int = 0):
        self.normalise_indian_numbers = normalise_indian_numbers
        self._fail_remaining = fail_first_n_calls
        self.calls = 0

    # ---- public contract -------------------------------------------------
    def complete(self, task: str, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        self.calls += 1
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise TransientError("mock: simulated rate limit (429)")

        if task == "extract":
            content = json.dumps(self._extract(_after_marker(user, "INVOICE TEXT:")))
        elif task == "explain":
            content = self._explain(_after_marker(user, "CONTEXT:"))
        else:
            raise ValueError(f"mock has no behaviour for task={task!r}")

        # Rough token accounting so cost shows up in the eval report even in mock mode.
        return LLMResponse(
            content=content,
            prompt_tokens=_approx_tokens(system) + _approx_tokens(user),
            completion_tokens=_approx_tokens(content),
            model="mock",
        )

    # ---- extraction --------------------------------------------------------
    def _extract(self, text: str) -> dict:
        inv = _first(_INVOICE_NO, text)
        po = _first(_PO_NO, text)
        vendor = _first(_VENDOR, text)
        currency = _first(_CURRENCY, text)
        total_raw = _first(_TOTAL_LINE, text)
        total = self._parse_amount(total_raw) if total_raw else None

        items = []
        for pat in (_ITEM_A, _ITEM_B, _ITEM_C):
            for sku, desc, qty, unit in pat.findall(text):
                price = self._parse_amount(unit)
                if price is None:
                    continue
                items.append(
                    {"sku": sku, "description": desc.strip(), "quantity": float(qty), "unit_price": price}
                )
            if items:
                break

        found = [inv, vendor, po, total]
        confidence = sum(1 for f in found if f is not None) / len(found)
        issues = []
        if inv is None:
            issues.append("invoice_number_missing")
        if total is None:
            issues.append("total_missing")
        if vendor is None:
            issues.append("vendor_missing")
        if not items:
            issues.append("no_line_items")

        return {
            "invoice_number": inv.upper() if inv else None,
            "vendor_name": vendor,
            "po_number": po.upper() if po else None,
            "total_amount": total,
            "currency": currency,
            "line_items": items,
            "confidence": round(confidence, 2),
            "issues": issues,
        }

    def _parse_amount(self, raw: str) -> Optional[float]:
        """Parse a grouped number.

        Western grouping (1,250.00 / 124,500.00) always works. Indian grouping
        (1,24,500.00) only works when `normalise_indian_numbers` is on. With it off,
        the regex still *matches part of* the string ("24,500.00") and returns a
        wrong-but-plausible number. That silent failure is the bug the eval harness
        caught; `--no-indian-numbers` reproduces it.
        """
        raw = raw.strip()
        if _WESTERN.match(raw):
            return float(raw.replace(",", ""))
        if self.normalise_indian_numbers and _INDIAN.match(raw):
            return float(raw.replace(",", ""))
        # Fallback: take the longest western-formatted suffix (this is the trap).
        m = re.search(r"\d{1,3}(,\d{3})*(\.\d{1,2})?$", raw)
        if m:
            return float(m.group(0).replace(",", ""))
        return None

    # ---- explanation -------------------------------------------------------
    def _explain(self, context_json: str) -> str:
        try:
            ctx = json.loads(context_json)
        except json.JSONDecodeError:
            return "Summary unavailable: context was not valid JSON. Recommended action: review manually."
        reasons = ctx.get("reasons") or ["no specific reason recorded"]
        inv = ctx.get("invoice_number") or "unknown invoice"
        vendor = ctx.get("vendor_name") or "unknown vendor"
        amount = ctx.get("total_amount")
        amount_s = f"{amount:,.2f} {ctx.get('currency') or ''}".strip() if amount is not None else "unknown amount"
        decision = ctx.get("decision", "needs_human")
        action = {
            "needs_human": "Check the flagged items against the PO and goods receipt, then approve or reject.",
            "reject": "Confirm the rejection reason with the vendor before closing.",
            "auto_approve": "No action needed.",
        }.get(decision, "Review manually.")
        return f"{inv} from {vendor} for {amount_s} was routed for review because: {'; '.join(reasons)}. {action}"


# ---- helpers -----------------------------------------------------------------
def _first(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _after_marker(text: str, marker: str) -> str:
    idx = text.find(marker)
    return text[idx + len(marker):].strip() if idx >= 0 else text


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)
