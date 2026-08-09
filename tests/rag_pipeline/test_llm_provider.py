import pytest

from src.config import GeminiConfig, LLMConfig, OllamaConfig
from src.rag_pipeline import llm_provider
from src.rag_pipeline.llm_provider import build_provider

STOP = ["\nUser:"]


@pytest.fixture
def llm_config():
    return LLMConfig(
        provider="ollama",
        ollama=OllamaConfig(model_name="test-model", base_url="http://localhost:11434"),
        gemini=GeminiConfig(model_name="gemini-test"),
    )


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("LLM_PROVIDER", "OLLAMA_BASE_URL", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def recorded(monkeypatch):
    """Replaces the real backends so no client or network call is created."""
    calls = {}

    class FakeOllama:
        def __init__(self, model_name, base_url, stop_sequences):
            calls.update(kind="ollama", model_name=model_name, base_url=base_url)

    class FakeGemini:
        def __init__(self, model_name, api_key, stop_sequences):
            calls.update(kind="gemini", model_name=model_name, api_key=api_key)

    monkeypatch.setattr(llm_provider, "OllamaProvider", FakeOllama)
    monkeypatch.setattr(llm_provider, "GeminiProvider", FakeGemini)
    return calls


def test_defaults_to_the_configured_provider(llm_config, recorded):
    build_provider(llm_config, STOP)

    assert recorded["kind"] == "ollama"
    assert recorded["model_name"] == "test-model"


def test_env_var_overrides_the_configured_provider(llm_config, recorded, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    build_provider(llm_config, STOP)

    assert recorded["kind"] == "gemini"
    assert recorded["model_name"] == "gemini-test"
    assert recorded["api_key"] == "test-key"


def test_ollama_base_url_can_be_overridden(llm_config, recorded, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")

    build_provider(llm_config, STOP)

    assert recorded["base_url"] == "http://ollama:11434"


def test_provider_name_is_case_insensitive(llm_config, recorded, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "OLLAMA")

    build_provider(llm_config, STOP)

    assert recorded["kind"] == "ollama"


def test_gemini_without_an_api_key_fails_with_guidance(llm_config, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        build_provider(llm_config, STOP)


def test_unknown_provider_is_rejected(llm_config, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_provider(llm_config, STOP)
