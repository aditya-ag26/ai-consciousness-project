"""
Builds a FAISS vector store from multiple data sources.

This script orchestrates the following steps:
1. Loads documents from raw text files (transcripts).
2. Loads documents from a processed Parquet file (papers).
3. Combines all documents.
4. Splits documents into manageable chunks.
5. Initializes a sentence-transformer embedding model.
6. Creates a new FAISS index or updates an existing one with the document embeddings.
"""
import logging
from pathlib import Path

import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

from src.config import config  # Import our validated config object
from src.rag_pipeline.embeddings import build_embeddings

# Set up professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_from_parquet(file_path: Path) -> list[Document]:
    """Loads documents from a Parquet file."""
    if not file_path.exists():
        logging.warning(f"Parquet file not found at: {file_path}. Skipping.")
        return []
        
    logging.info(f"Loading papers from {file_path}...")
    df = pd.read_parquet(file_path)
    
    docs = []
    # Use the efficient df.to_dict('records') instead of iterrows()
    for row in df.to_dict('records'):
        content = f"Title: {row.get('title', '')}\n\nAbstract: {row.get('abstract', '')}"
        metadata = {
            "title": row.get('title', ''),
            "primary_category": row.get('categories', '').split(" ")[0],
            "authors": row.get('authors', ''),
            "source_type": "arxiv_paper",
        }
        docs.append(Document(page_content=content, metadata=metadata))
        
    logging.info(f"Loaded {len(docs)} documents from Parquet.")
    return docs


def load_from_text_files(source_paths: list[Path]) -> list[Document]:
    """Loads documents from a list of .txt files and directories."""
    all_docs = []
    
    for path in source_paths:
        if not path.exists():
            logging.warning(f"Source not found: {path}. Skipping.")
            continue
            
        if path.is_dir():
            logging.info(f"Loading all transcripts from directory: {path}...")
            # Use rglob to find all .txt files in the directory and subdirectories
            files_to_load = list(path.rglob("*.txt"))
        elif path.is_file() and path.suffix == ".txt":
            logging.info(f"Loading transcript from file: {path}...")
            files_to_load = [path]
        else:
            logging.warning(f"Skipping unsupported source: {path}")
            continue

        for file_path in files_to_load:
            loader = TextLoader(str(file_path), encoding="utf-8")
            raw_docs = loader.load()
            
            title = file_path.stem.replace("_", " ").title()
            for doc in raw_docs:
                doc.metadata = {"title": title, "source_type": "transcript"}
                all_docs.append(doc)

    logging.info(f"Loaded {len(all_docs)} documents from text files.")
    return all_docs


def main() -> None:
    """Main function to orchestrate the vector store creation."""
    logging.info("Starting vector store build process...")
    
    # Use the dedicated config section
    pipeline_config = config.embedding_pipeline
    
    # 1. Load documents from all sources (SRP)
    paper_docs = load_from_parquet(pipeline_config.parquet_source)
    transcript_docs = load_from_text_files(pipeline_config.transcript_sources)
    all_documents = paper_docs + transcript_docs

    if not all_documents:
        logging.error("No documents found from any source. Exiting.")
        return

    # 2. Split documents (SRP)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=pipeline_config.text_splitter.chunk_size,
        chunk_overlap=pipeline_config.text_splitter.chunk_overlap,
    )
    split_docs = splitter.split_documents(all_documents)
    logging.info(f"Split {len(all_documents)} documents into {len(split_docs)} chunks.")

    # 3. Initialize Embeddings (SRP)
    logging.info(f"Initializing embedding model: {pipeline_config.embedding_model}")
    embeddings = build_embeddings(
        pipeline_config.embedding_model, config.rag_application.embedding_backend
    )

    # 4. Build or Update Vector Store (SRP)
    index_path = pipeline_config.faiss_index_path
    index_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dir exists

    if index_path.exists():
        logging.info(f"Loading existing FAISS index from {index_path}.")
        vector_store = FAISS.load_local(
            folder_path=str(index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        logging.info(f"Adding {len(split_docs)} new document chunks to the index.")
        vector_store.add_documents(split_docs)
    else:
        logging.info("Creating new FAISS index.")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        
    # 5. Save the final index
    vector_store.save_local(str(index_path))
    logging.info(f"FAISS index successfully saved to {index_path}")


if __name__ == "__main__":
    main()