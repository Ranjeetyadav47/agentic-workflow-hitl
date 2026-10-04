import os
import unittest

from agentflow.eval import run_eval, write_report

from helpers import Workspace


class EvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace(seed=11)

    @classmethod
    def tearDownClass(cls):
        cls.ws.cleanup()

    def test_workflow_matches_ground_truth_after_fix(self):
        r = run_eval(self.ws.data_dir, normalise_indian_numbers=True, label="after")
        self.assertEqual(r.n, 120)
        self.assertEqual(r.accuracy, 1.0)
        self.assertEqual(r.false_auto_approvals, 0)
        self.assertEqual(r.missed_rejects, 0)
        self.assertEqual(r.dead_lettered, 0)
        self.assertLess(r.human_review_rate, 0.6)

    def test_bug_is_reproducible_and_never_pays_wrongly(self):
        r = run_eval(self.ws.data_dir, normalise_indian_numbers=False, label="before")
        self.assertLess(r.accuracy, 1.0)
        # The bug makes the system over-cautious, never over-generous: that is the right failure mode.
        self.assertEqual(r.false_auto_approvals, 0)
        self.assertTrue(all(m.predicted == "needs_human" for m in r.mismatches))

    def test_report_is_written(self):
        r = run_eval(self.ws.data_dir, label="x")
        path = write_report([r], os.path.join(self.ws.dir, "reports", "r.md"))
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            text = f.read()
        self.assertIn("Decision accuracy", text)
        self.assertIn("confusion matrix", text)


if __name__ == "__main__":
    unittest.main()
