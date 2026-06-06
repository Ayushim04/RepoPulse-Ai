import logging
import os

from flask import Flask

from config import DevelopmentConfig, ProductionConfig
from routes.chat import chat_bp
from routes.dashboard import dashboard_bp
from routes.webhook import webhook_bp


def create_app(config_name: str = "development") -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    config_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
    }
    app.config.from_object(config_map.get(config_name, DevelopmentConfig))

    logging.basicConfig(
        level=logging.DEBUG if app.config.get("DEBUG") else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app.logger.info(f"RepoPulse AI starting in '{config_name}' mode.")

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(webhook_bp, url_prefix="/api")
    app.register_blueprint(chat_bp, url_prefix="/api")

    _auto_ingest_on_startup(app)

    return app


def _auto_ingest_on_startup(app: Flask) -> None:
    """
    Pre-build the RAG index at startup when REPO_PATH is configured,
    so the Doc-Bot is ready before the first request arrives.
    """
    repo_path = os.environ.get("REPO_PATH")
    if not repo_path:
        app.logger.info("REPO_PATH not set — RAG index starts empty. POST to /api/ingest to build.")
        return

    with app.app_context():
        try:
            from rag_ingestion import scan_repository
            chunk_count = scan_repository(repo_path)
            app.logger.info(f"Startup ingest complete: {chunk_count} chunks from '{repo_path}'.")
        except Exception as exc:
            app.logger.error(f"Startup ingest failed for '{repo_path}': {exc}.")


if __name__ == "__main__":
    env = os.environ.get("FLASK_ENV", "development")
    app = create_app(config_name=env)
    app.run(host="0.0.0.0", port=5000)
