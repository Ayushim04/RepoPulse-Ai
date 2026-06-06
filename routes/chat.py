"""
RepoPulse AI - Doc-Bot Chat Endpoint (Phase 2)
================================================
Provides the `/api/chat` and `/api/ingest` Flask endpoints.

/api/chat  (POST)
  Receives a user question, retrieves relevant codebase context from the
  RAG index, passes both to Gemini, and returns a grounded answer.

/api/ingest (POST)
  Triggers a fresh scan of the target repository directory and rebuilds
  the in-memory RAG index. Useful for a "Sync Repo" button in the dashboard.

/api/ingest/status (GET)
  Returns current index statistics (chunk count, files indexed, etc.)
  so the dashboard can show a "Knowledge Base" health widget.
"""

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

# Singleton AI engine — shared with pr_service to avoid re-initialising the client
_ai_engine: AIEngine | None = None


def _get_ai_engine() -> AIEngine:
    """Return a module-level singleton AIEngine instance."""
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = AIEngine()
    return _ai_engine


# ---------------------------------------------------------------------------
# /api/chat — Doc-Bot question answering
# ---------------------------------------------------------------------------

@chat_bp.route("/chat", methods=["POST"])
def doc_bot_chat() -> tuple[Response, int]:
    """
    Answer a user question about the codebase using RAG + Gemini.

    Request body (JSON):
        {
            "question": "How does the authentication middleware work?"
        }

    Response body (JSON):
        {
            "answer": "...",
            "sources": ["src/auth.py", "docs/auth.md"],
            "chunks_used": 3,
            "index_ready": true
        }

    Returns:
        200 with answer on success.
        400 if the request body is malformed or question is missing.
        503 if the RAG index is empty (ingest hasn't run yet).
    """
    # --- Parse request ---
    data: dict[str, Any] = request.get_json(silent=True) or {}
    question: str = data.get("question", "").strip()

    if not question:
        logger.warning("Chat request received with no 'question' field.")
        return jsonify({
            "error": "Missing required field: 'question'.",
            "hint": "POST JSON like: {\"question\": \"How does X work?\"}"
        }), 400

    logger.info(f"Doc-Bot question received: '{question[:100]}'")

    # --- Check index health ---
    stats = get_index_stats()
    if not stats["index_ready"]:
        logger.warning(
            "Doc-Bot query arrived but RAG index is empty. "
            "Trigger /api/ingest first."
        )
        return jsonify({
            "error": "Knowledge base is not ready.",
            "hint": "POST to /api/ingest with {\"repo_path\": \"/path/to/repo\"} first.",
            "index_stats": stats,
        }), 503

    # --- Retrieve relevant context chunks ---
    results = find_relevant_chunks(query=question, top_k=5)
    context_string = build_context_string(results)
    sources = list({r.chunk.source_file for r in results})

    logger.debug(
        f"Retrieved {len(results)} chunks from "
        f"{len(sources)} source file(s): {sources}"
    )

    # --- Generate grounded answer via Gemini ---
    ai = _get_ai_engine()
    answer = ai.answer_question(question=question, context=context_string)

    return jsonify({
        "answer": answer,
        "sources": sources,
        "chunks_used": len(results),
        "index_stats": stats,
    }), 200


# ---------------------------------------------------------------------------
# /api/ingest — Trigger RAG index rebuild
# ---------------------------------------------------------------------------

@chat_bp.route("/ingest", methods=["POST"])
def trigger_ingest() -> tuple[Response, int]:
    """
    Scan a local repository directory and rebuild the RAG knowledge base.

    Request body (JSON):
        {
            "repo_path": "/absolute/path/to/repo"   (optional)
        }

    If `repo_path` is omitted, falls back to the REPO_PATH environment
    variable, then to the current working directory.

    Response body (JSON):
        {
            "status": "ok",
            "chunks_indexed": 184,
            "index_stats": { ... }
        }
    """
    data: dict[str, Any] = request.get_json(silent=True) or {}

    # Resolve repo path: request body → env var → cwd
    repo_path: str = (
        data.get("repo_path")
        or os.environ.get("REPO_PATH")
        or os.getcwd()
    )

    logger.info(f"Ingest triggered for path: '{repo_path}'")

    try:
        chunk_count = scan_repository(repo_path)
        stats = get_index_stats()

        logger.info(f"Ingest complete: {chunk_count} chunks indexed from '{repo_path}'")
        return jsonify({
            "status": "ok",
            "repo_path": repo_path,
            "chunks_indexed": chunk_count,
            "index_stats": stats,
        }), 200

    except NotADirectoryError as exc:
        logger.error(f"Ingest failed — invalid path: {exc}")
        return jsonify({"error": str(exc)}), 400

    except Exception as exc:
        logger.exception(f"Unexpected error during ingest of '{repo_path}': {exc}")
        return jsonify({
            "error": "Ingest failed due to an unexpected server error.",
            "detail": str(exc),
        }), 500


# ---------------------------------------------------------------------------
# /api/ingest/status — Index health check
# ---------------------------------------------------------------------------

@chat_bp.route("/ingest/status", methods=["GET"])
def ingest_status() -> tuple[Response, int]:
    """
    Return the current state of the in-memory RAG index.

    Response body (JSON):
        {
            "total_chunks": 184,
            "unique_tokens": 3201,
            "files_indexed": 23,
            "index_ready": true
        }
    """
    stats = get_index_stats()
    logger.debug(f"Index status requested: {stats}")
    return jsonify(stats), 200
