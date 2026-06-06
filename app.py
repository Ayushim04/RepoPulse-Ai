"""
RepoPulse AI - Application Factory (Phase 2)
=============================================
Wires together all Phase 1 + Phase 2 components:
  - Dashboard UI route        (GET  /)
  - GitHub webhook listener   (POST /api/webhook/github)
  - Doc-Bot chat endpoint     (POST /api/chat)
  - RAG ingest endpoints      (POST /api/ingest, GET /api/ingest/status)

Startup behaviour:
  - If REPO_PATH is set in the environment, the RAG index is built
    automatically at startup so the Doc-Bot is ready immediately.
  - If REPO_PATH is not set, the index starts empty. Trigger a build
    by POSTing to /api/ingest from the dashboard or curl.
"""

import logging
import os

from flask import Flask

from config import DevelopmentConfig, ProductionConfig
from routes.chat import chat_bp
from routes.dashboard import dashboard_bp
from routes.webhook import webhook_bp


def create_app(config_name: str = "development") -> Flask:
    """
    Application factory.

    Args:
        config_name: 'development' | 'production'

    Returns:
        A fully configured, ready-to-serve Flask application.
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # --- Load Configuration ---
    config_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
    }
    app.config.from_object(config_map.get(config_name, DevelopmentConfig))

    # --- Configure Logging ---
    logging.basicConfig(
        level=logging.DEBUG if app.config.get("DEBUG") else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app.logger.info(f"RepoPulse AI starting in '{config_name}' mode.")

    # --- Register Blueprints ---
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(webhook_bp, url_prefix="/api")  # /api/webhook/github
    app.register_blueprint(chat_bp, url_prefix="/api")      # /api/chat, /api/ingest

    # --- Auto-ingest repository at startup (if REPO_PATH is configured) ---
    _auto_ingest_on_startup(app)

    return app


def _auto_ingest_on_startup(app: Flask) -> None:
    """
    Optionally pre-build the RAG index at startup.

    This means the Doc-Bot is ready to answer questions the moment the
    first user visits — no manual /api/ingest trigger needed.

    We run this inside app.app_context() so any Flask-aware code inside
    rag_ingestion has access to current_app if needed in Phase 3.
    """
    repo_path = os.environ.get("REPO_PATH")
    if not repo_path:
        app.logger.info(
            "REPO_PATH not set — RAG index will start empty. "
            "POST to /api/ingest to build the knowledge base."
        )
        return

    with app.app_context():
        try:
            from rag_ingestion import scan_repository
            app.logger.info(f"Auto-ingesting repository at startup: '{repo_path}'")
            chunk_count = scan_repository(repo_path)
            app.logger.info(
                f"Startup ingest complete: {chunk_count} chunks indexed "
                f"from '{repo_path}'."
            )
        except Exception as exc:
            # Never crash the app over a failed ingest — just log it clearly
            app.logger.error(
                f"Startup ingest FAILED for '{repo_path}': {exc}. "
                "The Doc-Bot will be unavailable until /api/ingest succeeds."
            )


if __name__ == "__main__":
    env = os.environ.get("FLASK_ENV", "development")
    app = create_app(config_name=env)
    app.run(host="0.0.0.0", port=5000)
