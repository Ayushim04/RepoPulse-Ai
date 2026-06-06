import os
from dataclasses import dataclass


@dataclass
class BaseConfig:
    SECRET_KEY: str = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me-in-prod")
    DEBUG: bool = False
    TESTING: bool = False

    # GitHub signs webhook payloads with this secret via HMAC-SHA256
    GITHUB_WEBHOOK_SECRET: str | None = os.environ.get("GITHUB_WEBHOOK_SECRET")

    GEMINI_API_KEY: str | None = os.environ.get("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro-latest")

    # PAT needs `repo` scope (private) or `public_repo` (public)
    GITHUB_TOKEN: str | None = os.environ.get("GITHUB_TOKEN")

    REPO_PATH: str | None = os.environ.get("REPO_PATH")
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")


@dataclass
class DevelopmentConfig(BaseConfig):
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"


@dataclass
class ProductionConfig(BaseConfig):
    DEBUG: bool = False

    @classmethod
    def validate(cls) -> None:
        required = ["FLASK_SECRET_KEY", "GITHUB_WEBHOOK_SECRET", "GEMINI_API_KEY", "GITHUB_TOKEN"]
        missing = [k for k in required if not os.environ.get(k)]
        if missing:
            raise EnvironmentError(f"Missing required environment variables: {missing}")
