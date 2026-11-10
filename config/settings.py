import os
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()


class Provider(str, Enum):
    GOOGLE = "google"
    OPENAI = "openai"
    GROQ = "groq"

    @classmethod
    def from_str(cls, value: str) -> "Provider":
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        valid = ", ".join([p.value for p in cls])
        raise ValueError(
            f"Unsupported provider '{value}'. Supported providers are: {valid}."
        )


class Settings:
    """Centralized configuration reading from environment variables."""

    # Default providers
    DEFAULT_CHAT_PROVIDER: Provider = Provider.from_str(
        os.getenv("DEFAULT_CHAT_PROVIDER", "google")
    )
    DEFAULT_EMBEDDING_PROVIDER: Provider = Provider.from_str(
        os.getenv("DEFAULT_EMBEDDING_PROVIDER", "google")
    )

    # API Keys
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

    # Chat model defaults
    GOOGLE_CHAT_MODEL: str = os.getenv("GOOGLE_CHAT_MODEL", "gemini-3.5-flash")
    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    GROQ_CHAT_MODEL: str = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")

    # Embedding model defaults
    GOOGLE_EMBEDDING_MODEL: str = os.getenv(
        "GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"
    )
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    # Versioning
    EMBEDDING_CONFIG_VERSION: str = os.getenv("EMBEDDING_CONFIG_VERSION", "v1.0.0")


settings = Settings()

