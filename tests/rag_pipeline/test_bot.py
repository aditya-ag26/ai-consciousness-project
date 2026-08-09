from types import SimpleNamespace

import pytest
from langchain.schema import Document

from src.rag_pipeline.bot import QueryBot, format_history, format_response


def test_format_response_with_sources():
    """
    Tests if the format_response function correctly formats an answer
    with both arXiv and transcript sources.
    """
    # 1. Arrange: Create a fake RAG result dictionary
    fake_result = {
        "result": "Consciousness is a complex phenomenon.",
        "source_documents": [
            Document(
                page_content="...",
                metadata={
                    "title": "A Paper on AI",
                    "source_type": "arxiv_paper",
                    "primary_category": "cs.AI",
                },
            ),
            Document(
                page_content="...",
                metadata={
                    "title": "An Expert Transcript",
                    "source_type": "transcript",
                },
            ),
        ],
    }

    # 2. Act: Call the function we are testing
    formatted_string = format_response(fake_result)

    # 3. Assert: Check if the output is what we expect
    assert "Answer:" in formatted_string
    assert "Consciousness is a complex phenomenon." in formatted_string
    assert "Sources:" in formatted_string
    assert "A Paper on AI (cs.AI)" in formatted_string
    assert "An Expert Transcript" in formatted_string

def test_format_response_no_sources():
    """
    Tests if the format_response function works correctly when no sources
    are returned.
    """
    # 1. Arrange
    fake_result = {
        "result": "I don't know.",
        "source_documents": [],
    }

    # 2. Act
    formatted_string = format_response(fake_result)

    # 3. Assert
    assert "Answer:" in formatted_string
    assert "I don't know." in formatted_string
    assert "Sources:" not in formatted_string


@pytest.fixture
def gated_bot():
    """A QueryBot with only the config the relevance gate needs."""
    bot = QueryBot.__new__(QueryBot)
    bot.config = SimpleNamespace(relevance_threshold=1.15)
    return bot


def scored(*distances):
    return [(Document(page_content="...", metadata={}), d) for d in distances]


def test_in_scope_when_best_match_is_close_enough(gated_bot):
    assert gated_bot.is_in_scope(scored(0.61, 1.4, 1.9)) is True


def test_out_of_scope_when_every_match_is_distant(gated_bot):
    assert gated_bot.is_in_scope(scored(1.5, 1.7, 1.9)) is False


def test_threshold_is_inclusive(gated_bot):
    assert gated_bot.is_in_scope(scored(1.15)) is True


def test_out_of_scope_when_nothing_retrieved(gated_bot):
    assert gated_bot.is_in_scope([]) is False


def test_format_history_labels_speakers():
    formatted = format_history([("user", "What is qualia?"), ("assistant", "Subjective experience.")], window=3)

    assert formatted == "User: What is qualia?\nAssistant: Subjective experience."


def test_format_history_keeps_only_the_most_recent_window():
    turns = [("user", f"q{i}") for i in range(10)]

    formatted = format_history(turns, window=2)

    assert formatted.count("\n") == 3  # window of 2 exchanges == 4 turns
    assert "q9" in formatted
    assert "q0" not in formatted


def test_format_history_with_no_turns():
    assert format_history([], window=3) == "(no previous messages)"