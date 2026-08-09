"""
Pluggable embedding backends.

Both backends run the same `all-MiniLM-L6-v2` weights and produce identical,
unit-normalised vectors, so a FAISS index built with one is readable by the
other. They differ only in the runtime they need:

* ``torch`` - sentence-transformers on PyTorch. Convenient for development and
              for rebuilding the index, but PyTorch alone is ~700 MB.
* ``onnx``  - the same model executed by ONNX Runtime. No PyTorch, which cuts
              the container image by roughly 2 GB and the cold start from about
              110 s to a few seconds.

`tests/rag_pipeline/test_embeddings.py` pins the equivalence that makes the two
interchangeable.
"""
import logging
import os

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class OnnxEmbeddings(Embeddings):
    """LangChain embedding backed by ONNX Runtime via fastembed."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        # Keeping the model outside the system temp directory means a container
        # can bake it in at build time and never download at runtime.
        cache_dir = os.getenv("FASTEMBED_CACHE_DIR") or None
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings(model_name: str, backend: str = "onnx") -> Embeddings:
    """
    Builds the embedding backend named by configuration.

    EMBEDDING_BACKEND overrides the configured value, so one image can be run
    either way without a rebuild.
    """
    backend = os.getenv("EMBEDDING_BACKEND", backend).lower()

    if backend == "onnx":
        logger.info(f"Embedding with ONNX Runtime: {model_name}")
        return OnnxEmbeddings(model_name)

    if backend == "torch":
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info(f"Embedding with PyTorch: {model_name}")
        return HuggingFaceEmbeddings(model_name=model_name)

    raise ValueError(
        f"Unknown embedding backend '{backend}'. Supported backends: onnx, torch."
    )
