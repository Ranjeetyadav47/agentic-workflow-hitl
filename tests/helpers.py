"""Shared fixtures for the test suite (unittest-style so it runs with or without pytest)."""

from __future__ import annotations

import os
import shutil
import tempfile

from agentflow.config import RetryConfig, Settings
from agentflow.data import generate
from agentflow.llm.mock import MockLLM
from agentflow.logging_utils import configure_logging
from agentflow.orchestrator import Orchestrator
from agentflow.store import Store
from agentflow.tools.erp import ERP


class Workspace:
    """A temp directory with generated data and a fresh database."""

    def __init__(self, seed: int = 7):
        configure_logging(quiet=True)
        self.dir = tempfile.mkdtemp(prefix="agentflow-test-")
        self.data_dir = os.path.join(self.dir, "data")
        generate(out_dir=self.data_dir, seed=seed)
        self.db_path = os.path.join(self.dir, "test.db")

    def settings(self, **overrides) -> Settings:
        return Settings.from_env(
            db_path=self.db_path, data_dir=self.data_dir, llm_mode="mock",
            retry=RetryConfig(max_attempts=3, sleep=False), **overrides
        )

    def orchestrator(self, llm: MockLLM | None = None, **overrides) -> tuple[Orchestrator, Store]:
        s = self.settings(**overrides)
        store = Store(s.db_path)
        return Orchestrator(s, store, ERP(s.data_dir), llm=llm), store

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def invoices_by_scenario(data_dir: str) -> dict[str, list[dict]]:
    from agentflow.data import load_invoices
    out: dict[str, list[dict]] = {}
    for inv in load_invoices(data_dir):
        out.setdefault(inv["scenario"], []).append(inv)
    return out
