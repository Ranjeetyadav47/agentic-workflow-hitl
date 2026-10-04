"""Typed records that flow between agents. Plain dataclasses: no framework, easy to serialise."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class Decision(str, Enum):
    AUTO_APPROVE = "auto_approve"
    NEEDS_HUMAN = "needs_human"
    REJECT = "reject"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    APPROVED = "approved"       # terminal: posted to ledger
    REJECTED = "rejected"       # terminal
    FAILED = "failed"           # terminal: in dead-letter queue


class TransientError(Exception):
    """Retryable: timeouts, rate limits, flaky downstream. The orchestrator retries these."""


class PermanentError(Exception):
    """Not retryable: bad input, schema violation. Goes straight to the dead-letter queue."""


@dataclass
class LineItem:
    sku: str
    description: str
    quantity: float
    unit_price: float

    @property
    def total(self) -> float:
        return round(self.quantity * self.unit_price, 2)


@dataclass
class ExtractedInvoice:
    """What the intake agent pulls out of raw invoice text."""

    invoice_number: Optional[str]
    vendor_name: Optional[str]
    po_number: Optional[str]
    total_amount: Optional[float]
    currency: Optional[str]
    line_items: list[LineItem] = field(default_factory=list)
    confidence: float = 0.0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MatchResult:
    """Three-way match: invoice vs purchase order vs goods receipt."""

    po_found: bool
    vendor_active: Optional[bool]
    vendor_matches_po: Optional[bool]
    quantity_ok: Optional[bool]
    max_price_variance_pct: Optional[float]
    received_total: Optional[float]
    po_total: Optional[float]
    duplicate_of: Optional[str]
    vendor_prior_invoices: Optional[int] = None   # None = vendor on the invoice could not be identified
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RULE_FIRED = "FIRED"        # trace marker; `fired_rules` depends on it, so keep them together
RULE_PASSED = "pass"


@dataclass
class PolicyOutcome:
    decision: Decision
    reasons: list[str]
    rule_trace: list[str]          # every rule evaluated, in order, with its result

    def fired_rules(self) -> list[str]:
        """Names of the rules that fired, in policy order. The first one set the decision."""
        return [t.split(":", 1)[0] for t in self.rule_trace if t.endswith(f": {RULE_FIRED}")]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


@dataclass
class ReviewerNote:
    """Plain-language summary for the human reviewer, written by the explain agent."""

    summary: str
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowState:
    """Everything the orchestrator carries between steps. Serialised to SQLite after each step."""

    run_id: str
    invoice_id: str
    raw_text: str
    extracted: Optional[ExtractedInvoice] = None
    match: Optional[MatchResult] = None
    policy: Optional[PolicyOutcome] = None
    note: Optional[ReviewerNote] = None
    human_decision: Optional[str] = None
    human_note: Optional[str] = None
    ledger_entry_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "invoice_id": self.invoice_id,
            "raw_text": self.raw_text,
            "extracted": self.extracted.to_dict() if self.extracted else None,
            "match": self.match.to_dict() if self.match else None,
            "policy": self.policy.to_dict() if self.policy else None,
            "note": self.note.to_dict() if self.note else None,
            "human_decision": self.human_decision,
            "human_note": self.human_note,
            "ledger_entry_id": self.ledger_entry_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkflowState":
        ex = d.get("extracted")
        extracted = None
        if ex:
            extracted = ExtractedInvoice(
                invoice_number=ex["invoice_number"],
                vendor_name=ex["vendor_name"],
                po_number=ex["po_number"],
                total_amount=ex["total_amount"],
                currency=ex["currency"],
                line_items=[LineItem(**li) for li in ex.get("line_items", [])],
                confidence=ex.get("confidence", 0.0),
                issues=list(ex.get("issues", [])),
            )
        m = d.get("match")
        match = MatchResult(**m) if m else None
        p = d.get("policy")
        policy = (
            PolicyOutcome(decision=Decision(p["decision"]), reasons=p["reasons"], rule_trace=p["rule_trace"])
            if p
            else None
        )
        n = d.get("note")
        note = ReviewerNote(**n) if n else None
        return cls(
            run_id=d["run_id"],
            invoice_id=d["invoice_id"],
            raw_text=d["raw_text"],
            extracted=extracted,
            match=match,
            policy=policy,
            note=note,
            human_decision=d.get("human_decision"),
            human_note=d.get("human_note"),
            ledger_entry_id=d.get("ledger_entry_id"),
        )
