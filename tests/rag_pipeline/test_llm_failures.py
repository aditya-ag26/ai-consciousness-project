"""
Generation outages must surface as an explicit, actionable failure rather than
a hang or an opaque 500. Retrieval and the guardrail keep working regardless,
so the two are reported differently.
"""
import pytest

from src.rag_pipeline.llm_provider import (
    LLMUnavailableError,
    OllamaProvider,
    _describe_gemini_failure,
)


class Boom(Exception):
    pass


def test_quota_errors_explain_the_daily_limit():
    message = _describe_gemini_failure(Boom("429 ResourceExhausted: quota exceeded"))

    assert "quota" in message.lower()
    assert "resets" in message.lower()
    # The reader should know retrieval still works.
    assert "guardrail" in message.lower()


def test_auth_errors_point_at_the_api_key():
    message = _describe_gemini_failure(Boom("403 API key not valid"))

    assert "GOOGLE_API_KEY" in message


def test_unrecognised_errors_get_a_generic_message():
    message = _describe_gemini_failure(Boom("connection reset by peer"))

    assert "temporarily unavailable" in message.lower()


def test_ollama_failures_are_translated(monkeypatch):
    provider = OllamaProvider.__new__(OllamaProvider)

    class FailingLLM:
        num_predict = 0

        def invoke(self, prompt):
            raise Boom("connection refused")

    provider._llm = FailingLLM()

    with pytest.raises(LLMUnavailableError, match="ollama serve"):
        provider.generate("anything", max_tokens=10)
