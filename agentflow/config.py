"""Runtime configuration. Everything a reviewer might ask "why this number?" about lives here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PolicyConfig:
    """Business rules applied by the policy agent. Deterministic by design."""

    auto_approve_limit: float = 250_000.0   # invoices above this always go to a human
    price_tolerance_pct: float = 2.0        # unit-price variance vs PO tolerated without review
    currency: str = "INR"


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    base_delay_s: float = 0.5               # exponential backoff: base * 2**attempt
    max_delay_s: float = 8.0
    sleep: bool = True                      # tests set False so retries are instant


@dataclass(frozen=True)
class Settings:
    db_path: str = "agentflow.db"
    data_dir: str = "data"
    llm_mode: str = "mock"                  # "mock" | "live"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    normalise_indian_numbers: bool = True   # the fix behind the before/after in README
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        """Build settings from environment variables, with explicit overrides winning."""
        base = dict(
            db_path=os.getenv("AGENTFLOW_DB", "agentflow.db"),
            data_dir=os.getenv("AGENTFLOW_DATA_DIR", "data"),
            llm_mode=os.getenv("AGENTFLOW_LLM_MODE", "mock"),
            llm_base_url=os.getenv("AGENTFLOW_LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_api_key=os.getenv("AGENTFLOW_LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            llm_model=os.getenv("AGENTFLOW_LLM_MODEL", "gpt-4o-mini"),
        )
        base.update(overrides)
        return cls(**base)
