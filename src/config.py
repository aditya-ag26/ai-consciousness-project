"""
Configuration loader for the project.

Uses Pydantic to load and validate settings from a YAML file,
ensuring that the configuration is type-safe and structured correctly.
"""
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

# Define the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env so local settings and secrets reach os.getenv without being exported
# by hand. Real environment variables win, which is how hosting platforms and
# container orchestrators are expected to supply these values.
load_dotenv(PROJECT_ROOT / ".env", override=False)

class DataSourceConfig(BaseModel):
    """Configuration for the data source."""
    kaggle_dataset: str

class ProcessingConfig(BaseModel):
    """Configuration for data processing parameters."""
    keywords: list[str]

class PathConfig(BaseModel):
    """Configuration for project paths."""
    download_dir: Path
    output_path: Path

class LocalJsonProcessingConfig(BaseModel):
    """Configuration for processing the local arXiv JSON snapshot."""
    input_path: Path
    output_path: Path
    filter_keywords: list[str]
    target_categories: list[str]
    max_title_len: int
    max_abstract_len: int

class TextSplitterConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int

class EmbeddingPipelineConfig(BaseModel):
    transcript_sources: list[Path]
    parquet_source: Path
    faiss_index_path: Path
    embedding_model: str
    text_splitter: TextSplitterConfig

class OllamaConfig(BaseModel):
    model_name: str
    base_url: str

class GeminiConfig(BaseModel):
    model_name: str

class LLMConfig(BaseModel):
    """Selects and configures the swappable language-model backend."""
    provider: str
    ollama: OllamaConfig
    gemini: GeminiConfig

class RAGApplicationConfig(BaseModel):
    faiss_index_path: Path
    log_path: Path
    embedding_model: str
    embedding_backend: str
    llm: LLMConfig
    retrieval_k: int
    relevance_threshold: float
    history_window: int
    out_of_scope_message: str
    prompt_template: str
    condense_prompt_template: str
    condense_num_predict: int
    stop_sequences: list[str]
    answer_length_map: dict[str, int]

class AppConfig(BaseModel):
    """Main application configuration model."""
    data_source: DataSourceConfig
    processing: ProcessingConfig
    paths: PathConfig
    local_json_processing: LocalJsonProcessingConfig
    embedding_pipeline: EmbeddingPipelineConfig
    rag_application: RAGApplicationConfig

def load_config() -> AppConfig:
    """
    Loads configuration from the YAML file and returns a validated AppConfig object.

    Returns:
        AppConfig: The validated application configuration.
    """
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)

    # Resolve relative paths to absolute paths for all path-containing sections.
    # Covers every section so the app works regardless of working directory,
    # which matters once it runs in a container.
    path_sections = [
        "paths",
        "local_json_processing",
        "embedding_pipeline",
        "rag_application",
    ]
    for section_name in path_sections:
        if section_name in config_yaml:
            for key, value in config_yaml[section_name].items():
                if ("path" in key or "dir" in key) and isinstance(value, str):
                    config_yaml[section_name][key] = PROJECT_ROOT / value

    return AppConfig.model_validate(config_yaml)

# Load the configuration globally on import
config = load_config()