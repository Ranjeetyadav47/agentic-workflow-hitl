"""The orchestrator: a small, explicit state machine.

    intake -> match -> policy -> explain -> gate
                                            |-- auto_approve -> execute -> APPROVED
                                            |-- reject                  -> REJECTED
                                            '-- needs_human -> review queue -> (human) -> execute | REJECTED

Each step runs under `_run_step`, which handles retries with exponential backoff for
TransientError, sends PermanentError (and exhausted retries) to the dead-letter queue,
records an audit row per attempt, and persists state after every successful step so a
run can be resumed from where it stopped.

No framework. It is ~200 lines, which is the point: every transition is visible.
Porting to LangGraph is a mechanical mapping of steps to nodes and the gate to a
conditional edge; what you would gain is streaming and checkpoint plumbing, what
you would lose is the ability to read the whole control flow on one screen.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from .agents import ExecuteAgent, ExplainAgent, IntakeAgent, MatchAgent, PolicyAgent
from .config import Settings
from .llm import LLMClient, build_client
from .logging_utils import EventLogger
from .models import Decision, PermanentError, RunStatus, TransientError, WorkflowState
from .store import StepRecord, Store, stable_hash
from .tools.erp import ERP

STEPS = ("intake", "match", "policy", "explain", "gate")


@dataclass
class RunResult:
    run_id: str
    invoice_id: str
    status: RunStatus
    decision: Optional[Decision]
    reasons: list[str]
    ledger_entry_id: Optional[str]
    error: Optional[str] = None


class Orchestrator:
    def __init__(self, settings: Settings, store: Store, erp: ERP, llm: Optional[LLMClient] = None):
        self.settings = settings
        self.store = store
        self.log = EventLogger()
        self.llm = llm or build_client(settings)
        self.intake = IntakeAgent(self.llm)
        self.match = MatchAgent(erp, store, settings.policy.price_tolerance_pct)
        self.policy = PolicyAgent(settings.policy)
        self.explain = ExplainAgent(self.llm)
        self.execute = ExecuteAgent(store)

    # ---- public API ---------------------------------------------------------------
    def process(self, invoice_id: str, raw_text: str) -> RunResult:
        state = WorkflowState(run_id=uuid.uuid4().hex, invoice_id=invoice_id, raw_text=raw_text)
        self.store.create_run(state)
        self.log.info("run.start", run_id=state.run_id, invoice_id=invoice_id)
        return self._advance(state, start="intake")

    def resume(self, run_id: str, human_decision: str, note: str = "") -> RunResult:
        """Called after a human resolves a review-queue item."""
        if human_decision not in ("approve", "reject"):
            raise ValueError("human_decision must be 'approve' or 'reject'")
        state = self.store.load_state(run_id)
        row = self.store.get_run(run_id)
        if row["status"] != RunStatus.AWAITING_REVIEW.value:
            raise ValueError(f"run {run_id} is {row['status']}, not awaiting review")
        if human_decision == "approve" and not _postable(state):
            raise ValueError(
                f"run {run_id} cannot be approved: the invoice has no reliable number, vendor or total. "
                "Reject it, or fix the source document and re-submit."
            )
        self.store.resolve_review(run_id, human_decision, note)
        state.human_decision, state.human_note = human_decision, note
        self.log.info("review.resolved", run_id=run_id, decision=human_decision)

        if human_decision == "reject":
            self.store.save_state(state, RunStatus.REJECTED, "gate")
            return self._result(state, RunStatus.REJECTED)
        return self._do_execute(state, approved_by=_approved_by(state))

    def replay(self, run_id: str) -> RunResult:
        """Re-run a dead-lettered run from the step that failed (after the cause is fixed)."""
        rows = {r["run_id"]: r for r in self.store.pending_dead_letters()}
        if run_id not in rows:
            raise KeyError(f"run {run_id} is not in the dead-letter queue")
        state = self.store.load_state(run_id)
        step = rows[run_id]["step"]
        self.store.mark_replayed(run_id)
        self.log.info("run.replay", run_id=run_id, from_step=step)
        if step == "execute":
            # The decision was already made (by policy or by a human); do not re-run the gate.
            return self._do_execute(state, approved_by=_approved_by(state))
        return self._advance(state, start=step)

    # ---- state machine ----------------------------------------------------------------
    def _advance(self, state: WorkflowState, start: str) -> RunResult:
        try:
            for step in STEPS[STEPS.index(start):]:
                self.store.save_state(state, RunStatus.RUNNING, step)
                if step == "intake":
                    self._run_step(state, "intake", lambda: self._step_intake(state))
                elif step == "match":
                    self._run_step(state, "match", lambda: self._step_match(state))
                elif step == "policy":
                    self._run_step(state, "policy", lambda: self._step_policy(state))
                elif step == "explain":
                    if state.policy.decision == Decision.AUTO_APPROVE:
                        self.store.record_step(StepRecord(state.run_id, "explain", 1, "skipped"))
                        continue   # nobody will read it; do not pay for it
                    self._run_step(state, "explain", lambda: self._step_explain(state))
                elif step == "gate":
                    return self._gate(state)
            raise RuntimeError("unreachable: gate must return")   # pragma: no cover
        except (PermanentError, TransientError) as e:
            return self._dead_letter(state, str(e))

    def _gate(self, state: WorkflowState) -> RunResult:
        d = state.policy.decision
        if d == Decision.AUTO_APPROVE:
            return self._do_execute(state, approved_by="policy")
        if d == Decision.REJECT:
            self.store.save_state(state, RunStatus.REJECTED, "gate")
            self.log.info("run.rejected", run_id=state.run_id, reasons=state.policy.reasons)
            return self._result(state, RunStatus.REJECTED)
        # needs_human
        self.store.enqueue_review(state.run_id, state.invoice_id, state.note.summary, state.policy.reasons)
        self.store.save_state(state, RunStatus.AWAITING_REVIEW, "gate")
        self.log.info("run.awaiting_review", run_id=state.run_id, reasons=state.policy.reasons)
        return self._result(state, RunStatus.AWAITING_REVIEW)

    def _do_execute(self, state: WorkflowState, approved_by: str) -> RunResult:
        self.store.save_state(state, RunStatus.RUNNING, "execute")
        try:
            self._run_step(state, "execute", lambda: self._step_execute(state, approved_by))
        except (PermanentError, TransientError) as e:
            return self._dead_letter(state, str(e))
        self.store.save_state(state, RunStatus.APPROVED, "execute")
        self.log.info("run.approved", run_id=state.run_id, ledger_entry_id=state.ledger_entry_id, approved_by=approved_by)
        return self._result(state, RunStatus.APPROVED)

    def _dead_letter(self, state: WorkflowState, error: str) -> RunResult:
        row = self.store.get_run(state.run_id)
        step = row["current_step"] or "unknown"
        attempts = sum(1 for s in self.store.steps_for(state.run_id) if s["step"] == step)
        self.store.dead_letter(state, step, error, attempts)
        self.store.save_state(state, RunStatus.FAILED, step)
        self.log.error("run.dead_letter", run_id=state.run_id, step=step, error=error, attempts=attempts)
        return self._result(state, RunStatus.FAILED, error=error)

    # ---- retry wrapper ---------------------------------------------------------------
    def _run_step(self, state: WorkflowState, name: str, fn: Callable[[], tuple]) -> None:
        """Run `fn` with retries. `fn` returns (input_obj, output_obj, llm_response_or_None)."""
        cfg = self.settings.retry
        for attempt in range(1, cfg.max_attempts + 1):
            t0 = time.perf_counter()
            try:
                inp, out, resp = fn()
                ms = (time.perf_counter() - t0) * 1000
                self.store.record_step(StepRecord(
                    state.run_id, name, attempt, "ok", stable_hash(inp), stable_hash(out), round(ms, 3),
                    resp.prompt_tokens if resp else 0, resp.completion_tokens if resp else 0,
                ))
                self.log.info("step.ok", run_id=state.run_id, step=name, attempt=attempt, latency_ms=round(ms, 2))
                return
            except TransientError as e:
                ms = (time.perf_counter() - t0) * 1000
                self.store.record_step(StepRecord(state.run_id, name, attempt, "retry", latency_ms=round(ms, 3), error=str(e)))
                self.log.warning("step.retry", run_id=state.run_id, step=name, attempt=attempt, error=str(e))
                if attempt == cfg.max_attempts:
                    raise TransientError(f"{name}: gave up after {attempt} attempts: {e}") from e
                delay = min(cfg.base_delay_s * (2 ** (attempt - 1)), cfg.max_delay_s)
                if cfg.sleep:
                    time.sleep(delay)
            except PermanentError as e:
                ms = (time.perf_counter() - t0) * 1000
                self.store.record_step(StepRecord(state.run_id, name, attempt, "failed", latency_ms=round(ms, 3), error=str(e)))
                self.log.error("step.failed", run_id=state.run_id, step=name, attempt=attempt, error=str(e))
                raise

    # ---- steps (each returns input, output, llm_response) ----------------------------------
    def _step_intake(self, state: WorkflowState):
        extracted, resp = self.intake.run(state.raw_text)
        state.extracted = extracted
        return state.raw_text, extracted.to_dict(), resp

    def _step_match(self, state: WorkflowState):
        match = self.match.run(state.extracted)
        state.match = match
        return state.extracted.to_dict(), match.to_dict(), None

    def _step_policy(self, state: WorkflowState):
        outcome = self.policy.run(state.extracted, state.match)
        state.policy = outcome
        return {"extracted": state.extracted.to_dict(), "match": state.match.to_dict()}, outcome.to_dict(), None

    def _step_explain(self, state: WorkflowState):
        note, resp = self.explain.run(state.extracted, state.match, state.policy)
        state.note = note
        return state.policy.to_dict(), note.to_dict(), resp

    def _step_execute(self, state: WorkflowState, approved_by: str):
        entry_id = self.execute.run(state.run_id, state.extracted, approved_by)
        state.ledger_entry_id = entry_id
        return {"invoice": state.extracted.invoice_number, "approved_by": approved_by}, {"entry_id": entry_id}, None

    # ---- helpers ----------------------------------------------------------------------
    @staticmethod
    def _result(state: WorkflowState, status: RunStatus, error: Optional[str] = None) -> RunResult:
        return _result(state, status, error)


def _postable(state: WorkflowState) -> bool:
    ex = state.extracted
    return bool(ex and ex.invoice_number and ex.vendor_name and ex.total_amount is not None)


def _approved_by(state: WorkflowState) -> str:
    if state.human_decision == "approve":
        return f"human:{state.human_note or 'approved'}"
    return "policy"


def _result(state: WorkflowState, status: RunStatus, error: Optional[str] = None) -> RunResult:
    return RunResult(
        run_id=state.run_id,
        invoice_id=state.invoice_id,
        status=status,
        decision=state.policy.decision if state.policy else None,
        reasons=list(state.policy.reasons) if state.policy else [],
        ledger_entry_id=state.ledger_entry_id,
        error=error,
    )
