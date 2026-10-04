from .base import LLMClient, LLMResponse
from .mock import MockLLM
from .openai_compat import OpenAICompatibleClient

__all__ = ["LLMClient", "LLMResponse", "MockLLM", "OpenAICompatibleClient", "build_client"]


def build_client(settings) -> LLMClient:
    """Pick the client from settings. Mock by default so the repo runs with no credentials."""
    if settings.llm_mode == "live":
        if not settings.llm_api_key:
            raise RuntimeError("AGENTFLOW_LLM_MODE=live but no API key set (AGENTFLOW_LLM_API_KEY)")
        return OpenAICompatibleClient(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key, model=settings.llm_model
        )
    # AGENTFLOW_MOCK_FAIL_FIRST=N makes the mock raise a retryable error on its first N
    # calls, so retries, backoff and the dead-letter queue can be demonstrated from the CLI.
    import os
    fail_first = int(os.getenv("AGENTFLOW_MOCK_FAIL_FIRST", "0"))
    return MockLLM(normalise_indian_numbers=settings.normalise_indian_numbers, fail_first_n_calls=fail_first)
