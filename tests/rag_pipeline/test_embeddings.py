import pytest

from src.config import config
from src.rag_pipeline.embeddings import build_embeddings

MODEL = config.rag_application.embedding_model


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown embedding backend"):
        build_embeddings(MODEL, "not-a-backend")


def test_backend_name_is_case_insensitive():
    assert build_embeddings(MODEL, "ONNX") is not None


def test_env_var_overrides_the_configured_backend(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "onnx")

    embeddings = build_embeddings(MODEL, "torch")

    assert type(embeddings).__name__ == "OnnxEmbeddings"


def test_onnx_embeddings_are_unit_normalised():
    vector = build_embeddings(MODEL, "onnx").embed_query("what is consciousness?")

    magnitude = sum(component**2 for component in vector) ** 0.5
    assert len(vector) == 384
    assert magnitude == pytest.approx(1.0, abs=1e-5)


@pytest.mark.slow
def test_onnx_and_torch_backends_agree():
    """
    The ONNX and PyTorch backends must produce the same vectors, because the
    committed FAISS index was built with one and is queried with the other.
    """
    texts = ["the hard problem of consciousness", "how do I cook pasta?"]

    onnx = build_embeddings(MODEL, "onnx").embed_documents(texts)
    torch = build_embeddings(MODEL, "torch").embed_documents(texts)

    for onnx_vector, torch_vector in zip(onnx, torch):
        cosine = sum(a * b for a, b in zip(onnx_vector, torch_vector))
        assert cosine == pytest.approx(1.0, abs=1e-4)
