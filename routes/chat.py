import logging
import os
from typing import Any

from flask import Blueprint, Response, jsonify, request

from ai_engine import AIEngine
from rag_ingestion import (
    build_context_string,
    find_relevant_chunks,
    get_index_stats,
    scan_repository,
)

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__)

_ai_engine: AIEngine | None = None


def _get_ai_engine() -> AIEngine:
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = AIEngine()
    return _ai_engine


@chat_bp.route("/chat", methods=["POST"])
def doc_bot_chat() -> tuple[Response, int]:
    """
    POST /api/chat

    Body: { "question": "..." }
    Returns the AI answer grounded in the RAG index, plus source file citations.
    """
    data: dict[str, Any] = request.get_json(silent=True) or {}
    question: str = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Missing 'question' field."}), 400

    stats = get_index_stats()
    if not stats["index_ready"]:
        return jsonify({
            "error": "Knowledge base is not ready.",
            "hint": "POST to /api/ingest first.",
            "index_stats": stats,
        }), 503

    results = find_relevant_chunks(query=question, top_k=5)
    context_string = build_context_string(results)
    sources = list({r.chunk.source_file for r in results})

    answer = _get_ai_engine().answer_question(question=question, context=context_string)

    return jsonify({
        "answer": answer,
        "sources": sources,
        "chunks_used": len(results),
        "index_stats": stats,
    }), 200


@chat_bp.route("/ingest", methods=["POST"])
def trigger_ingest() -> tuple[Response, int]:
    """
    POST /api/ingest

    Scans the repository at repo_path (body), REPO_PATH env var, or cwd,
    and rebuilds the in-memory RAG index.
    """
    data: dict[str, Any] = request.get_json(silent=True) or {}
    repo_path: str = data.get("repo_path") or os.environ.get("REPO_PATH") or os.getcwd()

    logger.info(f"Ingest triggered for: '{repo_path}'")
    try:
        chunk_count = scan_repository(repo_path)
        return jsonify({
            "status": "ok",
            "repo_path": repo_path,
            "chunks_indexed": chunk_count,
            "index_stats": get_index_stats(),
        }), 200
    except NotADirectoryError as exc:
        logger.error(f"Ingest failed — invalid path: {exc}")
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception(f"Unexpected ingest error for '{repo_path}': {exc}")
        return jsonify({"error": "Ingest failed.", "detail": str(exc)}), 500


@chat_bp.route("/ingest/status", methods=["GET"])
def ingest_status() -> tuple[Response, int]:
    return jsonify(get_index_stats()), 200
