"""SQLite persistence: runs, per-step audit log, review queue, dead letters, ledger.

One file, no server. Every step writes a row with its input/output hashes, latency,
attempt number and token usage, which is what "auditable" means in practice: you can
answer "why was this invoice approved, by which rule, on which attempt, at what cost".
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from .models import RunStatus, WorkflowState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    invoice_id    TEXT NOT NULL,
    status        TEXT NOT NULL,
    current_step  TEXT,
    decision      TEXT,
    state_json    TEXT NOT NULL,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_invoice ON runs(invoice_id);
CREATE INDEX IF NOT EXISTS idx_runs_status  ON runs(status);

CREATE TABLE IF NOT EXISTS steps (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    step          TEXT NOT NULL,
    attempt       INTEGER NOT NULL,
    status        TEXT NOT NULL,            -- ok | retry | failed | skipped
    input_hash    TEXT,
    output_hash   TEXT,
    latency_ms    REAL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    error         TEXT,
    ts            REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id);

CREATE TABLE IF NOT EXISTS review_queue (
    run_id        TEXT PRIMARY KEY,
    invoice_id    TEXT NOT NULL,
    summary       TEXT NOT NULL,
    reasons_json  TEXT NOT NULL,
    enqueued_at   REAL NOT NULL,
    resolved_at   REAL,
    decision      TEXT,
    reviewer_note TEXT
);

CREATE TABLE IF NOT EXISTS dead_letters (
    run_id        TEXT PRIMARY KEY,
    invoice_id    TEXT NOT NULL,
    step          TEXT NOT NULL,
    error         TEXT NOT NULL,
    attempts      INTEGER NOT NULL,
    state_json    TEXT NOT NULL,
    ts            REAL NOT NULL,
    replayed_at   REAL
);

CREATE TABLE IF NOT EXISTS ledger (
    entry_id      TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    invoice_number TEXT NOT NULL,
    vendor_name   TEXT NOT NULL,
    amount        REAL NOT NULL,
    currency      TEXT NOT NULL,
    approved_by   TEXT NOT NULL,            -- policy | human:<note>
    posted_at     REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ledger_dup ON ledger(vendor_name, invoice_number);
"""


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class StepRecord:
    run_id: str
    step: str
    attempt: int
    status: str
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    latency_ms: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None


class Store:
    def __init__(self, path: str = "agentflow.db"):
        self.path = path
        self._conn = sqlite3.connect(path, isolation_level=None)   # autocommit; we manage txns
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ---- runs -------------------------------------------------------------
    def create_run(self, state: WorkflowState) -> None:
        now = time.time()
        self._conn.execute(
            "INSERT INTO runs(run_id, invoice_id, status, current_step, decision, state_json, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (state.run_id, state.invoice_id, RunStatus.PENDING.value, None, None,
             json.dumps(state.to_dict()), now, now),
        )

    def save_state(self, state: WorkflowState, status: RunStatus, current_step: Optional[str]) -> None:
        decision = state.policy.decision.value if state.policy else None
        self._conn.execute(
            "UPDATE runs SET status=?, current_step=?, decision=?, state_json=?, updated_at=? WHERE run_id=?",
            (status.value, current_step, decision, json.dumps(state.to_dict()), time.time(), state.run_id),
        )

    def get_run(self, run_id: str) -> Optional[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()

    def load_state(self, run_id: str) -> WorkflowState:
        row = self.get_run(run_id)
        if row is None:
            raise KeyError(f"run {run_id} not found")
        return WorkflowState.from_dict(json.loads(row["state_json"]))

    def list_runs(self, status: Optional[str] = None, limit: int = 200) -> list[sqlite3.Row]:
        if status:
            return self._conn.execute(
                "SELECT * FROM runs WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit)
            ).fetchall()
        return self._conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()

    # ---- steps --------------------------------------------------------------
    def record_step(self, rec: StepRecord) -> None:
        self._conn.execute(
            "INSERT INTO steps(run_id, step, attempt, status, input_hash, output_hash, latency_ms,"
            " prompt_tokens, completion_tokens, error, ts) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (rec.run_id, rec.step, rec.attempt, rec.status, rec.input_hash, rec.output_hash,
             rec.latency_ms, rec.prompt_tokens, rec.completion_tokens, rec.error, time.time()),
        )

    def steps_for(self, run_id: str) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM steps WHERE run_id=? ORDER BY id", (run_id,)).fetchall()

    # ---- review queue -------------------------------------------------------
    def enqueue_review(self, run_id: str, invoice_id: str, summary: str, reasons: list[str]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO review_queue(run_id, invoice_id, summary, reasons_json, enqueued_at)"
            " VALUES (?,?,?,?,?)",
            (run_id, invoice_id, summary, json.dumps(reasons), time.time()),
        )

    def pending_reviews(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM review_queue WHERE resolved_at IS NULL ORDER BY enqueued_at"
        ).fetchall()

    def resolve_review(self, run_id: str, decision: str, note: str) -> None:
        cur = self._conn.execute(
            "UPDATE review_queue SET resolved_at=?, decision=?, reviewer_note=? WHERE run_id=? AND resolved_at IS NULL",
            (time.time(), decision, note, run_id),
        )
        if cur.rowcount == 0:
            raise KeyError(f"no pending review for run {run_id}")

    # ---- dead letters ---------------------------------------------------------
    def dead_letter(self, state: WorkflowState, step: str, error: str, attempts: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO dead_letters(run_id, invoice_id, step, error, attempts, state_json, ts)"
            " VALUES (?,?,?,?,?,?,?)",
            (state.run_id, state.invoice_id, step, error, attempts, json.dumps(state.to_dict()), time.time()),
        )

    def pending_dead_letters(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM dead_letters WHERE replayed_at IS NULL ORDER BY ts").fetchall()

    def mark_replayed(self, run_id: str) -> None:
        self._conn.execute("UPDATE dead_letters SET replayed_at=? WHERE run_id=?", (time.time(), run_id))

    # ---- ledger ---------------------------------------------------------------
    def find_ledger_duplicate(self, vendor_name: str, invoice_number: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT entry_id FROM ledger WHERE vendor_name=? AND invoice_number=?", (vendor_name, invoice_number)
        ).fetchone()
        return row["entry_id"] if row else None

    def post_ledger(self, entry_id: str, run_id: str, invoice_number: str, vendor_name: str,
                    amount: float, currency: str, approved_by: str) -> None:
        self._conn.execute(
            "INSERT INTO ledger(entry_id, run_id, invoice_number, vendor_name, amount, currency, approved_by, posted_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (entry_id, run_id, invoice_number, vendor_name, amount, currency, approved_by, time.time()),
        )

    def ledger_entries(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM ledger ORDER BY posted_at").fetchall()

    # ---- metrics ----------------------------------------------------------------
    def step_stats(self) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n, AVG(latency_ms) AS avg_ms, SUM(prompt_tokens) AS pt,"
            " SUM(completion_tokens) AS ct, SUM(CASE WHEN status='retry' THEN 1 ELSE 0 END) AS retries"
            " FROM steps"
        ).fetchone()
        return {
            "steps": row["n"] or 0,
            "avg_latency_ms": round(row["avg_ms"] or 0.0, 2),
            "prompt_tokens": row["pt"] or 0,
            "completion_tokens": row["ct"] or 0,
            "retries": row["retries"] or 0,
        }
