"""
Processes a local arXiv metadata JSONL file.

This script reads a large JSON Lines file containing arXiv paper metadata,
filters each paper based on keywords and categories defined in the central
configuration, transforms the data into a clean schema, and saves the result
as a compressed Parquet file.
"""
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from src.config import config  # Import our validated config object

# Set up professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def stream_papers(jsonl_path: Path) -> Iterator[dict]:
    """
    Lazily loads papers from a JSON Lines file using a generator.

    Args:
        jsonl_path: The path to the JSONL file.

    Yields:
        A dictionary representing a single paper's metadata.
    """
    logging.info(f"Streaming papers from {jsonl_path}...")
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed JSON line {i+1}: {line[:100]}...")
                continue


def paper_matches_criteria(paper: dict, keywords: list[str], categories: list[str]) -> bool:
    """
    Checks if a paper matches keyword and category criteria.

    Args:
        paper: A dictionary representing a single paper.
        keywords: Keywords to search for in title and abstract.
        categories: Target categories to match.

    Returns:
        True if the paper matches the criteria, False otherwise.
    """
    title = paper.get("title", "").lower()
    abstract = paper.get("abstract", "").lower()
    paper_categories = paper.get("categories", "").split()
    
    text_content = title + " " + abstract
    
    keyword_match = any(kw.lower() in text_content for kw in keywords)
    category_match = any(cat in categories for cat in paper_categories)
    
    return keyword_match and category_match


def transform_paper(paper: dict, max_title_len: int, max_abstract_len: int) -> dict:
    """
    Transforms a raw paper dictionary into a structured format.

    Args:
        paper: The raw paper dictionary.
        max_title_len: Max character length for the title.
        max_abstract_len: Max character length for the abstract.

    Returns:
        A dictionary with a clean, defined schema.
    """
    authors_parsed = paper.get("authors_parsed", [])
    author_names = [f"{first} {last}" for last, first, *_ in authors_parsed]

    return {
        "title": paper.get("title", "")[:max_title_len],
        "abstract": paper.get("abstract", "")[:max_abstract_len],
        "categories": paper.get("categories", ""),
        "authors": ", ".join(author_names),
        "update_date": paper.get("update_date", ""),
    }


def main() -> None:
    """Main function to orchestrate the data processing pipeline."""
    logging.info("Starting local JSON processing pipeline...")
    
    # Use the dedicated config section for this script
    proc_config = config.local_json_processing

    # 1. Filter and Transform Data (SRP)
    filtered_papers = []
    paper_iterator = stream_papers(proc_config.input_path)
    
    for paper in paper_iterator:
        if paper_matches_criteria(paper, proc_config.filter_keywords, proc_config.target_categories):
            transformed = transform_paper(
                paper, proc_config.max_title_len, proc_config.max_abstract_len
            )
            filtered_papers.append(transformed)

    if not filtered_papers:
        logging.warning("No papers matched the filter criteria. No output file will be generated.")
        return

    logging.info(f"Found {len(filtered_papers)} matching papers.")
    
    # 2. Save Data (SRP)
    output_path = proc_config.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(filtered_papers)
    df.to_parquet(output_path, index=False)
    
    logging.info(f"Filtered data successfully saved to: {output_path}")


if __name__ == "__main__":
    main()