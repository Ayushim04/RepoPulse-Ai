"""
RepoPulse AI - Configuration
==============================
Centralizes all application configuration.
Secrets are loaded exclusively from environment variables — never hardcoded.
"""

import os
from dataclasses import dataclass


@dataclass
class BaseConfig:
    """Shared base configuration for all environments."""

    # Flask core
    SECRET_KEY: str = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me-in-prod")
    DEBUG: bool = False
    TESTING: bool = False

    # GitHub Webhook
    # Set this in your .env — GitHub uses it to sign webhook payloads (HMAC-SHA256)
    GITHUB_WEBHOOK_SECRET: str | None = os.environ.get("GITHUB_WEBHOOK_SECRET")

    # Gemini API
    GEMINI_API_KEY: str | None = os.environ.get("GEMINI_API_KEY")

    # GitHub REST API — Personal Access Token for posting comments, fetching diffs
    # Needs `repo` scope for private repos, `public_repo` for public
    GITHUB_TOKEN: str | None = os.environ.get("GITHUB_TOKEN")

    # RAG / Doc-Bot — local path to the repository to index
    # e.g. /home/user/my-repo  or  . (current directory)
    REPO_PATH: str | None = os.environ.get("REPO_PATH")

    # Logging
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")


@dataclass
class DevelopmentConfig(BaseConfig):
    """Development-specific overrides."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"


@dataclass
class ProductionConfig(BaseConfig):
    """Production-specific overrides. Enforce stricter settings here."""
    DEBUG: bool = False

    @classmethod
    def validate(cls) -> None:
        """Raise an error if required production secrets are missing."""
        required = ["FLASK_SECRET_KEY", "GITHUB_WEBHOOK_SECRET", "GEMINI_API_KEY", "GITHUB_TOKEN"]
        missing = [key for key in required if not os.environ.get(key)]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables for production: {missing}"
            )
