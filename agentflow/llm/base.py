"""The only contract agents depend on. Swap providers without touching agent code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResponse:
    content: str                                   # raw text (JSON for extraction tasks)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "mock"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    def complete(self, task: str, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        """Run one completion.

        `task` is a short label ("extract", "explain") so a mock can route deterministically
        and a live client can tag usage by task.
        """
        ...
