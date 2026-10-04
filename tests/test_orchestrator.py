import unittest

from agentflow.llm.mock import MockLLM
from agentflow.models import Decision, RunStatus

from helpers import Workspace, invoices_by_scenario


class OrchestratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()
        cls.by = invoices_by_scenario(cls.ws.data_dir)

    @classmethod
    def tearDownClass(cls):
        cls.ws.cleanup()

    def _fresh(self, llm=None):
        import os
        if os.path.exists(self.ws.db_path):
            os.remove(self.ws.db_path)
        return self.ws.orchestrator(llm=llm)

    # ---- happy path ----------------------------------------------------------------
    def test_clean_invoice_is_posted_to_ledger_without_a_human(self):
        orch, store = self._fresh()
        inv = self.by["clean"][0]
        res = orch.process(inv["invoice_id"], inv["text"])
        self.assertEqual(res.status, RunStatus.APPROVED)
        self.assertEqual(res.decision, Decision.AUTO_APPROVE)
        self.assertIsNotNone(res.ledger_entry_id)
        entries = store.ledger_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["approved_by"], "policy")
        store.close()

    def test_explain_step_is_skipped_for_auto_approvals(self):
        """No human reads the note, so we do not pay for it."""
        orch, store = self._fresh()
        inv = self.by["clean"][0]
        res = orch.process(inv["invoice_id"], inv["text"])
        steps = {s["step"]: s["status"] for s in store.steps_for(res.run_id)}
        self.assertEqual(steps["explain"], "skipped")
        store.close()

    # ---- human in the loop ----------------------------------------------------------
    def test_needs_human_waits_then_resumes_on_approval(self):
        orch, store = self._fresh()
        inv = self.by["quantity_mismatch"][0]
        res = orch.process(inv["invoice_id"], inv["text"])
        self.assertEqual(res.status, RunStatus.AWAITING_REVIEW)
        self.assertIsNone(res.ledger_entry_id)
        self.assertEqual(len(store.pending_reviews()), 1)

        res2 = orch.resume(res.run_id, "approve", note="GRN updated")
        self.assertEqual(res2.status, RunStatus.APPROVED)
        self.assertEqual(store.ledger_entries()[0]["approved_by"], "human:GRN updated")
        self.assertEqual(len(store.pending_reviews()), 0)
        store.close()

    def test_human_reject_terminates_without_posting(self):
        orch, store = self._fresh()
        inv = self.by["missing_po"][0]
        res = orch.process(inv["invoice_id"], inv["text"])
        res2 = orch.resume(res.run_id, "reject", note="no PO raised")
        self.assertEqual(res2.status, RunStatus.REJECTED)
        self.assertEqual(len(store.ledger_entries()), 0)
        store.close()

    def test_cannot_resume_a_run_that_is_not_waiting(self):
        orch, store = self._fresh()
        inv = self.by["clean"][0]
        res = orch.process(inv["invoice_id"], inv["text"])
        with self.assertRaises(ValueError):
            orch.resume(res.run_id, "approve")
        store.close()

    # ---- hard stops -----------------------------------------------------------------
    def test_inactive_vendor_is_rejected(self):
        orch, store = self._fresh()
        inv = self.by["vendor_inactive"][0]
        res = orch.process(inv["invoice_id"], inv["text"])
        self.assertEqual(res.status, RunStatus.REJECTED)
        store.close()

    def test_duplicate_is_rejected_after_original_posts(self):
        orch, store = self._fresh()
        inv = self.by["clean"][0]
        first = orch.process(inv["invoice_id"], inv["text"])
        second = orch.process(inv["invoice_id"] + "-again", inv["text"])
        self.assertEqual(first.status, RunStatus.APPROVED)
        self.assertEqual(second.status, RunStatus.REJECTED)
        self.assertIn("duplicate", second.reasons[0])
        self.assertEqual(len(store.ledger_entries()), 1)
        store.close()

    # ---- resilience -----------------------------------------------------------------
    def test_transient_errors_are_retried_and_audited(self):
        orch, store = self._fresh(llm=MockLLM(fail_first_n_calls=2))
        inv = self.by["clean"][0]
        res = orch.process(inv["invoice_id"], inv["text"])
        self.assertEqual(res.status, RunStatus.APPROVED)
        intake = [s for s in store.steps_for(res.run_id) if s["step"] == "intake"]
        self.assertEqual([s["status"] for s in intake], ["retry", "retry", "ok"])
        store.close()

    def test_exhausted_retries_dead_letter_then_replay_succeeds(self):
        orch, store = self._fresh(llm=MockLLM(fail_first_n_calls=3))
        inv = self.by["clean"][0]
        res = orch.process(inv["invoice_id"], inv["text"])
        self.assertEqual(res.status, RunStatus.FAILED)
        self.assertEqual(len(store.pending_dead_letters()), 1)
        self.assertEqual(len(store.ledger_entries()), 0)

        res2 = orch.replay(res.run_id)        # the mock has stopped failing by now
        self.assertEqual(res2.status, RunStatus.APPROVED)
        self.assertEqual(len(store.pending_dead_letters()), 0)
        self.assertEqual(len(store.ledger_entries()), 1)
        store.close()

    def test_human_cannot_approve_an_unreadable_invoice(self):
        """A garbled scan has no total; approving it would post nothing sensible, so we refuse."""
        orch, store = self._fresh()
        inv = self.by["garbled"][0]
        res = orch.process(inv["invoice_id"], inv["text"])
        self.assertEqual(res.status, RunStatus.AWAITING_REVIEW)
        with self.assertRaises(ValueError):
            orch.resume(res.run_id, "approve")
        self.assertEqual(len(store.pending_reviews()), 1)       # still waiting; nothing was consumed
        res2 = orch.resume(res.run_id, "reject", note="unreadable, re-request from vendor")
        self.assertEqual(res2.status, RunStatus.REJECTED)
        store.close()

    def test_ledger_outage_at_execute_dead_letters_and_replays_without_rerunning_gate(self):
        """Human approved, then the ledger was locked: replay must post, not re-queue for review."""
        import sqlite3
        orch, store = self._fresh()
        inv = self.by["quantity_mismatch"][0]
        res = orch.process(inv["invoice_id"], inv["text"])

        real_post = store.post_ledger
        calls = {"n": 0}

        def flaky_post(*a, **kw):
            calls["n"] += 1
            raise sqlite3.OperationalError("database is locked")

        store.post_ledger = flaky_post
        res2 = orch.resume(res.run_id, "approve", note="ok")
        self.assertEqual(res2.status, RunStatus.FAILED)
        self.assertEqual(calls["n"], 3)                          # retried, then gave up
        self.assertEqual(store.pending_dead_letters()[0]["step"], "execute")

        store.post_ledger = real_post
        res3 = orch.replay(res.run_id)
        self.assertEqual(res3.status, RunStatus.APPROVED)
        self.assertEqual(len(store.pending_reviews()), 0)        # not re-queued
        self.assertEqual(store.ledger_entries()[0]["approved_by"], "human:ok")
        store.close()

    def test_state_round_trips_through_sqlite(self):
        orch, store = self._fresh()
        inv = self.by["price_variance"][0]
        res = orch.process(inv["invoice_id"], inv["text"])
        state = store.load_state(res.run_id)
        self.assertEqual(state.policy.decision, Decision.NEEDS_HUMAN)
        self.assertEqual(state.extracted.invoice_number, inv["invoice_number"])
        self.assertTrue(state.note.summary)
        store.close()


if __name__ == "__main__":
    unittest.main()
