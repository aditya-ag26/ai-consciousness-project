"""
Guards against an index that looks correct but carries no usable content.

The committed index once contained 151 chunks holding nothing but a paper's
title. Retrieval metrics stayed perfect - the right document was being found -
while the model received headings instead of abstracts and answered "there is
not enough information" to on-topic questions.

The cause was chunk size interacting with document layout: papers are stored as
"Title: ...\n\nAbstract: ...", the splitter breaks on the blank line first, and
a title could not merge with an abstract that alone exceeded the chunk size. So
these assertions are about the *interaction*, which no unit test of the splitter
would have caught.
"""
import pytest
from langchain_community.vectorstores import FAISS

from src.config import config
from src.rag_pipeline.embeddings import build_embeddings

# Below this a chunk holds a heading at best, not an argument.
THIN_CHUNK_CHARS = 200


@pytest.fixture(scope="module")
def chunks():
    embeddings = build_embeddings(config.rag_application.embedding_model, "onnx")
    store = FAISS.load_local(
        str(config.rag_application.faiss_index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return list(store.docstore._dict.values())


def test_the_index_is_not_empty(chunks):
    assert len(chunks) > 100


def test_no_chunk_is_only_a_title(chunks):
    title_only = [
        c.page_content
        for c in chunks
        if c.page_content.strip().startswith("Title:") and "Abstract:" not in c.page_content
    ]

    assert title_only == [], (
        f"{len(title_only)} chunks contain a title and no content. "
        "Retrieval will match them on the title and hand the model nothing to "
        "answer from. Check chunk_size against max_abstract_len."
    )


def test_chunks_carry_enough_text_to_answer_from(chunks):
    thin = [c for c in chunks if len(c.page_content) < THIN_CHUNK_CHARS]

    assert len(thin) / len(chunks) < 0.05, (
        f"{len(thin)} of {len(chunks)} chunks are under {THIN_CHUNK_CHARS} characters"
    )


def test_every_paper_keeps_its_abstract(chunks):
    papers = [c for c in chunks if c.metadata.get("source_type") == "arxiv_paper"]
    with_abstract = [c for c in papers if "Abstract:" in c.page_content]

    # One chunk per paper, each holding the abstract it belongs to.
    assert len(with_abstract) == len(papers)


def test_chunks_carry_the_metadata_citations_need(chunks):
    assert all(c.metadata.get("title") for c in chunks)
    assert all(
        c.metadata.get("source_type") in {"arxiv_paper", "transcript"} for c in chunks
    )
