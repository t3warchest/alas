"""
alas/agent_service/utils/config.py

Centralised settings loaded once at import time.
All components import from here — no scattered os.getenv() calls.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    eval_model: str = Field(default="gpt-4o-mini", alias="EVAL_MODEL")

    # Memory / vector store
    chroma_persist_dir: str = Field(default="./data/chromadb", alias="CHROMA_PERSIST_DIR")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    # Session behaviour
    session_window: int = Field(default=8, alias="SESSION_WINDOW")
    max_episode_summaries: int = Field(default=5, alias="MAX_EPISODE_SUMMARIES")
    eval_every_n_turns: int = Field(default=1, alias="EVAL_EVERY_N_TURNS")

    # API server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    cors_origins: list[str] = Field(default=["*"], alias="CORS_ORIGINS")

    model_config = {"env_file": ".env", "populate_by_name": True}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
