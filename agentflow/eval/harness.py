"""Evaluation harness: run the labelled invoice set through the workflow and score it.

What gets measured, and why:
  * accuracy             - headline, but not the number that matters most
  * false auto-approvals - invoices paid without a human that should not have been;
                           the only error class that costs real money
  * missed rejects       - duplicates / inactive vendors that slipped past the hard stops
  * human review rate    - the cost of being cautious; if this is 100% the system is useless
  * latency and tokens   - per invoice, so cost projections are grounded

A fresh database is used for every run so results are reproducible.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..config import RetryConfig, Settings
from ..data import load_invoices
from ..llm.mock import MockLLM
from ..logging_utils import configure_logging
from ..models import RunStatus
from ..orchestrator import Orchestrator
from ..store import Store
from ..tools.erp import ERP

CLASSES = ("auto_approve", "needs_human", "reject")

# Assumed list prices for projecting cost; stated in the report, easy to change.
PRICE_PER_M_INPUT = 0.15
PRICE_PER_M_OUTPUT = 0.60


@dataclass
class Mismatch:
    invoice_id: str
    scenario: str
    template: str
    expected: str
    predicted: str
    reasons: list[str]


@dataclass
class EvalResult:
    n: int
    accuracy: float
    confusion: dict[str, Counter]               # expected -> Counter(predicted)
    per_class: dict[str, dict[str, float]]      # class -> precision/recall/f1
    false_auto_approvals: int
    missed_rejects: int
    human_review_rate: float
    dead_lettered: int
    avg_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    retries: int
    wall_seconds: float
    mismatches: list[Mismatch] = field(default_factory=list)
    by_scenario: dict[str, tuple[int, int]] = field(default_factory=dict)   # scenario -> (correct, total)
    label: str = ""


def run_eval(data_dir: str = "data", normalise_indian_numbers: bool = True, label: str = "") -> EvalResult:
    configure_logging(quiet=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    settings = Settings.from_env(
        db_path=tmp.name, data_dir=data_dir, llm_mode="mock",
        normalise_indian_numbers=normalise_indian_numbers,
        retry=RetryConfig(sleep=False),
    )
    store = Store(settings.db_path)
    erp = ERP(data_dir)
    orch = Orchestrator(settings, store, erp, llm=MockLLM(normalise_indian_numbers=normalise_indian_numbers))

    confusion: dict[str, Counter] = defaultdict(Counter)
    mismatches: list[Mismatch] = []
    by_scenario: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    n = correct = human = dlq = 0
    t0 = time.perf_counter()

    for inv in load_invoices(data_dir):
        res = orch.process(inv["invoice_id"], inv["text"])
        expected = inv["expected_decision"]
        if res.status == RunStatus.FAILED:
            predicted = "dlq"
            dlq += 1
        else:
            predicted = res.decision.value
        n += 1
        confusion[expected][predicted] += 1
        by_scenario[inv["scenario"]][1] += 1
        if predicted == expected:
            correct += 1
            by_scenario[inv["scenario"]][0] += 1
        else:
            mismatches.append(Mismatch(inv["invoice_id"], inv["scenario"], inv["template"], expected, predicted, res.reasons))
        if predicted == "needs_human":
            human += 1

    wall = time.perf_counter() - t0
    stats = store.step_stats()
    store.close()
    os.unlink(tmp.name)

    per_class = {}
    for c in CLASSES:
        tp = confusion[c][c]
        fp = sum(confusion[e][c] for e in confusion if e != c)
        fn = sum(v for p, v in confusion[c].items() if p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}

    false_auto = sum(confusion[e]["auto_approve"] for e in confusion if e != "auto_approve")
    missed_rejects = sum(v for p, v in confusion["reject"].items() if p != "reject")

    return EvalResult(
        n=n,
        accuracy=correct / n if n else 0.0,
        confusion=dict(confusion),
        per_class=per_class,
        false_auto_approvals=false_auto,
        missed_rejects=missed_rejects,
        human_review_rate=human / n if n else 0.0,
        dead_lettered=dlq,
        avg_latency_ms=stats["avg_latency_ms"],
        prompt_tokens=stats["prompt_tokens"],
        completion_tokens=stats["completion_tokens"],
        retries=stats["retries"],
        wall_seconds=wall,
        mismatches=mismatches,
        by_scenario={k: (v[0], v[1]) for k, v in by_scenario.items()},
        label=label,
    )


def write_report(results: list[EvalResult], path: str) -> str:
    """Markdown report. Several results = several configurations compared side by side."""
    lines: list[str] = []
    lines.append("# Evaluation report\n")
    lines.append(f"Generated by `python -m agentflow eval`. Invoices: {results[0].n}. LLM: mock (deterministic). "
                 f"Fresh SQLite per run.\n")

    lines.append("## Headline\n")
    hdr = "| Metric | " + " | ".join(r.label or f"run {i+1}" for i, r in enumerate(results)) + " |"
    lines.append(hdr)
    lines.append("|" + " --- |" * (len(results) + 1))
    rows = [
        ("Decision accuracy", lambda r: f"{r.accuracy:.1%}"),
        ("False auto-approvals (paid without review, wrongly)", lambda r: str(r.false_auto_approvals)),
        ("Missed rejects", lambda r: str(r.missed_rejects)),
        ("Human review rate", lambda r: f"{r.human_review_rate:.1%}"),
        ("Dead-lettered runs", lambda r: str(r.dead_lettered)),
        ("Avg step latency (ms)", lambda r: f"{r.avg_latency_ms:.2f}"),
        ("Tokens per invoice (prompt / completion)", lambda r: f"{r.prompt_tokens / r.n:.0f} / {r.completion_tokens / r.n:.0f}"),
        ("Projected LLM cost per 1,000 invoices (USD)", lambda r: f"{_cost(r) * 1000 / r.n:.3f}"),
        ("Wall time (s)", lambda r: f"{r.wall_seconds:.2f}"),
    ]
    for name, fn in rows:
        lines.append(f"| {name} | " + " | ".join(fn(r) for r in results) + " |")
    lines.append("")
    lines.append(f"Cost assumes ${PRICE_PER_M_INPUT}/M input and ${PRICE_PER_M_OUTPUT}/M output tokens and the mock's "
                 "rough token counts (chars/4). Treat it as an order of magnitude, not a quote.\n")

    for r in results:
        title = r.label or "Result"
        lines.append(f"## {title}: per-class metrics\n")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("| --- | --- | --- | --- | --- |")
        for c in CLASSES:
            m = r.per_class[c]
            lines.append(f"| {c} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |")
        lines.append("")

        lines.append(f"## {title}: confusion matrix (rows = expected, columns = predicted)\n")
        cols = list(CLASSES) + ["dlq"]
        lines.append("| expected \\ predicted | " + " | ".join(cols) + " |")
        lines.append("|" + " --- |" * (len(cols) + 1))
        for e in CLASSES:
            lines.append(f"| {e} | " + " | ".join(str(r.confusion.get(e, Counter())[p]) for p in cols) + " |")
        lines.append("")

        lines.append(f"## {title}: by scenario\n")
        lines.append("| Scenario | Correct | Total |")
        lines.append("| --- | --- | --- |")
        for s, (c, t) in sorted(r.by_scenario.items()):
            lines.append(f"| {s} | {c} | {t} |")
        lines.append("")

        lines.append(f"## {title}: failure analysis\n")
        if not r.mismatches:
            lines.append("No mismatches.\n")
        else:
            lines.append("| Invoice | Scenario | Template | Expected | Predicted | Reasons given |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for m in r.mismatches:
                lines.append(f"| {m.invoice_id} | {m.scenario} | {m.template} | {m.expected} | {m.predicted} | "
                             f"{'; '.join(m.reasons)[:140]} |")
            lines.append("")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def _cost(r: EvalResult) -> float:
    return r.prompt_tokens / 1e6 * PRICE_PER_M_INPUT + r.completion_tokens / 1e6 * PRICE_PER_M_OUTPUT
