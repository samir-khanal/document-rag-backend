from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    gemini_api_key: str = ""

    qdrant_url: str
    qdrant_api_key: str | None = None

    postgres_url: str

    redis_url: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Load settings once and reuse the same object throughout the application.

    Keeping configuration in one place prevents database URLs and API keys
    from being scattered across the codebase.
    """
    return Settings()