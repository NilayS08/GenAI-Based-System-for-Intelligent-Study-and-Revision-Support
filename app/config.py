"""
Configuration module for Smart Revision Generator.
Loads environment variables and provides centralized settings.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "Smart Revision Generator"
    app_env: str = "development"
    debug: bool = True

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- OpenAI ---
    openai_api_key: str

    # --- Supabase ---
    supabase_url: str
    supabase_key: str

    # --- Vector Store ---
    faiss_index_path: str = "app/vector_store/faiss_index"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- LLM ---
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4000

    # --- Retrieval ---
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    top_k_results: int = 5

    # --- Chunking ---
    chunk_size: int = 400
    chunk_overlap: int = 50

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env.lower() == "production"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()


# Export settings instance for easy import
settings = get_settings()
