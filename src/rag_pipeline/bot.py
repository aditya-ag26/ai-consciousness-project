"""
Core logic for the RAG (Retrieval-Augmented Generation) chatbot.

This module defines the QueryBot class, which encapsulates retrieval over the
FAISS knowledge base, a relevance guardrail that keeps answers grounded in the
ingested documents, and history-aware answer generation.
"""
import logging
from collections.abc import Sequence
from typing import Any

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from src.config import RAGApplicationConfig
from src.rag_pipeline.embeddings import build_embeddings
from src.rag_pipeline.llm_provider import build_provider

logger = logging.getLogger(__name__)

# A (role, content) pair, where role is "user" or "assistant".
Turn = tuple[str, str]
ScoredDocs = list[tuple[Document, float]]


class QueryBot:
    """Encapsulates the RAG flow for querying the consciousness knowledge base."""

    def __init__(self, config: RAGApplicationConfig):
        """
        Initializes the QueryBot with its dependencies and configuration.

        Args:
            config: A Pydantic model containing all necessary configuration.
        """
        self.config = config
        self.db = self._load_vector_store()
        self.llm = build_provider(config.llm, config.stop_sequences)
        self.answer_prompt = PromptTemplate(
            template=config.prompt_template,
            input_variables=["chat_history", "context", "question"],
        )
        self.condense_prompt = PromptTemplate(
            template=config.condense_prompt_template,
            input_variables=["chat_history", "question"],
        )

    def _load_vector_store(self) -> FAISS:
        """Loads the FAISS index built by src.data.build_vector_store."""
        logger.info(f"Loading FAISS index from: {self.config.faiss_index_path}")
        try:
            embeddings = build_embeddings(
                self.config.embedding_model, self.config.embedding_backend
            )
            return FAISS.load_local(
                folder_path=str(self.config.faiss_index_path),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            logger.error(f"Fatal error loading FAISS index: {e}")
            raise SystemExit(1) from e

    def retrieve(self, query: str) -> ScoredDocs:
        """Returns the top-k documents for a query with their L2 distances."""
        return self.db.similarity_search_with_score(query, k=self.config.retrieval_k)

    def is_in_scope(self, scored_docs: Sequence[tuple[Document, float]]) -> bool:
        """
        Decides whether retrieved documents are close enough to answer from.

        FAISS returns L2 distance, so lower means more similar.
        """
        if not scored_docs:
            return False
        return min(score for _, score in scored_docs) <= self.config.relevance_threshold

    def condense(self, query: str, chat_history: Sequence[Turn]) -> str:
        """
        Rewrites a context-dependent follow-up into a standalone question.

        A bare "why?" retrieves nothing useful on its own, so it must inherit
        the topic from the conversation before it can be scored or answered.
        """
        rewritten = self.llm.generate(
            self.condense_prompt.format(
                chat_history=format_history(chat_history, self.config.history_window),
                question=query,
            ),
            max_tokens=self.config.condense_num_predict,
        )
        # The model occasionally adds commentary; keep only the first line.
        rewritten = rewritten.strip().splitlines()[0].strip() if rewritten.strip() else ""
        return rewritten or query

    def ask(
        self,
        query: str,
        num_predict_tokens: int,
        chat_history: Sequence[Turn] | None = None,
    ) -> dict[str, Any]:
        """
        Answers a question using only the ingested knowledge base.

        Args:
            query: The user's question.
            num_predict_tokens: The max number of tokens for the LLM response.
            chat_history: Prior (role, content) turns in this conversation.

        Returns:
            A dict with the answer, its source documents, and whether the
            relevance guardrail refused the question.
        """
        chat_history = list(chat_history or [])
        logger.info(f"Received query: '{query}'")

        search_query = query
        scored_docs = self.retrieve(search_query)

        # A follow-up that fails the gate may just be missing its context, so
        # give it one chance to be resolved against the conversation.
        if chat_history and not self.is_in_scope(scored_docs):
            search_query = self.condense(query, chat_history)
            logger.info(f"Condensed follow-up to: '{search_query}'")
            scored_docs = self.retrieve(search_query)

        if not self.is_in_scope(scored_docs):
            logger.info("Query rejected by relevance guardrail.")
            return {
                "result": self.config.out_of_scope_message.strip(),
                "source_documents": [],
                "refused": True,
            }

        documents = [doc for doc, _ in scored_docs]
        answer = self.llm.generate(
            self.answer_prompt.format(
                chat_history=format_history(chat_history, self.config.history_window),
                context="\n\n".join(doc.page_content for doc in documents),
                question=query,
            ),
            max_tokens=num_predict_tokens,
        )

        return {
            "result": answer.strip(),
            "source_documents": documents,
            "refused": False,
        }


def format_history(chat_history: Sequence[Turn], window: int) -> str:
    """Renders the most recent turns as transcript lines for a prompt."""
    if not chat_history:
        return "(no previous messages)"

    # A window counts exchanges, and each exchange is a user and assistant turn.
    recent = chat_history[-(window * 2):]
    labels = {"user": "User", "assistant": "Assistant"}
    return "\n".join(f"{labels.get(role, role)}: {content}" for role, content in recent)


def format_response(result: dict[str, Any]) -> str:
    """
    Formats the raw RAG output into a user-friendly string.

    Args:
        result: The raw dictionary response from the RAG flow.

    Returns:
        A formatted string containing the answer and its sources.
    """
    answer = result.get("result", "No answer found.")
    sources = result.get("source_documents", [])

    response = f"\nAnswer:\n{answer}"

    if sources:
        response += "\n\nSources:"
        unique_sources = { # Remove duplicate sources
            format_source_doc(doc) for doc in sources
        }
        for source_str in sorted(unique_sources):
            response += f"\n - {source_str}"

    return response


def format_source_doc(doc: Document) -> str:
    """Helper function to format a single source document."""
    title = doc.metadata.get("title", "Unknown Title")
    if doc.metadata.get("source_type") == "arxiv_paper":
        category = doc.metadata.get("primary_category", "N/A")
        return f"{title} ({category})"
    return title
