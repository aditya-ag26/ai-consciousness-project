"""
Pluggable language-model backends.

The RAG pipeline treats the LLM as a swappable component: retrieval, the
relevance guardrail, and prompt construction are identical no matter which model
generates the text. Two backends ship with the project.

* ``ollama``  - runs a model locally. No API key, no data leaves the machine,
                and it is the default for development.
* ``gemini``  - calls a hosted model. Needs an API key but almost no memory,
                which is what makes free-tier container hosting viable.

Adding another backend means implementing ``generate`` and registering it in
``build_provider``; nothing else in the pipeline changes.
"""
import logging
import os
from typing import Protocol

from src.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """The single capability the RAG pipeline needs from a language model."""

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Returns the model's completion for a fully-rendered prompt."""
        ...


class OllamaProvider:
    """Generates with a model served by a local Ollama instance."""

    def __init__(self, model_name: str, base_url: str, stop_sequences: list[str]):
        from langchain_ollama import OllamaLLM

        self._llm = OllamaLLM(
            model=model_name, base_url=base_url, stop=stop_sequences
        )

    def generate(self, prompt: str, max_tokens: int) -> str:
        self._llm.num_predict = max_tokens
        return self._llm.invoke(prompt)


class GeminiProvider:
    """Generates with a hosted Google Gemini model."""

    def __init__(self, model_name: str, api_key: str, stop_sequences: list[str]):
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._llm = ChatGoogleGenerativeAI(
            model=model_name, google_api_key=api_key, stop=stop_sequences
        )

    def generate(self, prompt: str, max_tokens: int) -> str:
        # Chat models return a message object rather than a bare string.
        response = self._llm.bind(max_output_tokens=max_tokens).invoke(prompt)
        return getattr(response, "content", str(response))


def build_provider(config: LLMConfig, stop_sequences: list[str]) -> LLMProvider:
    """
    Builds the LLM backend named by configuration.

    The provider and its connection details can be overridden by environment
    variables so the same image runs locally and in a container without edits.
    """
    provider = os.getenv("LLM_PROVIDER", config.provider).lower()

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", config.ollama.base_url)
        logger.info(f"Using local Ollama model '{config.ollama.model_name}' at {base_url}")
        return OllamaProvider(config.ollama.model_name, base_url, stop_sequences)

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM provider 'gemini' requires GOOGLE_API_KEY to be set. "
                "Set it, or use LLM_PROVIDER=ollama to run a local model instead."
            )
        logger.info(f"Using hosted Gemini model '{config.gemini.model_name}'")
        return GeminiProvider(config.gemini.model_name, api_key, stop_sequences)

    raise ValueError(
        f"Unknown LLM provider '{provider}'. Supported providers: ollama, gemini."
    )
