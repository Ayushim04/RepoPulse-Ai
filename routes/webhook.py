"""
RepoPulse AI - GitHub Webhook Listener
========================================
Handles incoming webhook events from GitHub.

Security model:
  - Every request is verified against the HMAC-SHA256 signature
    that GitHub attaches in the `X-Hub-Signature-256` header.
  - Requests with missing or invalid signatures are rejected with 403.

Supported events (Phase 1):
  - pull_request  →  routes to the AI code review pipeline
"""

import hashlib
import hmac
import json
import logging
from typing import Any

from flask import Blueprint, Response, current_app, jsonify, request

from services.pr_service import handle_pull_request_event

logger = logging.getLogger(__name__)

webhook_bp = Blueprint("webhook", __name__)


# ---------------------------------------------------------------------------
# Signature Verification Helper
# ---------------------------------------------------------------------------

def _verify_github_signature(payload_body: bytes, signature_header: str | None) -> bool:
    """
    Validate the GitHub webhook HMAC-SHA256 signature.

    Args:
        payload_body:      The raw request body bytes.
        signature_header:  The value of the `X-Hub-Signature-256` header.

    Returns:
        True if the signature is valid, False otherwise.
    """
    secret: str | None = current_app.config.get("GITHUB_WEBHOOK_SECRET")

    # If no secret is configured, skip verification (dev only — log a warning)
    if not secret:
        logger.warning(
            "GITHUB_WEBHOOK_SECRET is not set. "
            "Skipping signature verification (NOT safe for production)."
        )
        return True

    if not signature_header or not signature_header.startswith("sha256="):
        logger.warning("Webhook received without a valid signature header.")
        return False

    expected_signature = "sha256=" + hmac.new(
        secret.encode("utf-8"),
        payload_body,
        hashlib.sha256,
    ).hexdigest()

    # Use hmac.compare_digest to prevent timing attacks
    return hmac.compare_digest(expected_signature, signature_header)


# ---------------------------------------------------------------------------
# Webhook Endpoint
# ---------------------------------------------------------------------------

@webhook_bp.route("/webhook/github", methods=["POST"])
def github_webhook() -> tuple[Response, int]:
    """
    Primary GitHub webhook receiver.

    Endpoint: POST /api/webhook/github

    Headers expected:
      - X-GitHub-Event      : The event type (e.g. 'pull_request')
      - X-Hub-Signature-256 : HMAC-SHA256 signature of the payload
      - X-GitHub-Delivery   : Unique delivery GUID for idempotency

    Returns:
        JSON response with status and HTTP status code.
    """
    raw_body: bytes = request.get_data()
    signature: str | None = request.headers.get("X-Hub-Signature-256")
    event_type: str | None = request.headers.get("X-GitHub-Event")
    delivery_id: str | None = request.headers.get("X-GitHub-Delivery", "unknown")

    logger.info(f"Webhook received | delivery={delivery_id} | event={event_type}")

    # --- Step 1: Verify Signature ---
    if not _verify_github_signature(raw_body, signature):
        logger.error(f"Signature mismatch for delivery={delivery_id}. Rejecting.")
        return jsonify({"status": "error", "message": "Invalid signature."}), 403

    # --- Step 2: Parse JSON Payload ---
    try:
        payload: dict[str, Any] = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse JSON payload: {exc}")
        return jsonify({"status": "error", "message": "Malformed JSON payload."}), 400

    # --- Step 3: Route by Event Type ---
    if event_type == "pull_request":
        logger.info(
            f"PR event received | action='{payload.get('action')}' | "
            f"PR#{payload.get('pull_request', {}).get('number')}"
        )
        result = handle_pull_request_event(payload)
        return jsonify({"status": "accepted", "detail": result}), 202

    elif event_type == "ping":
        # GitHub sends a ping when a webhook is first configured
        logger.info("Ping event received from GitHub — webhook is active.")
        return jsonify({"status": "ok", "message": "Pong!"}), 200

    else:
        # Gracefully acknowledge unsupported events without crashing
        logger.debug(f"Unhandled event type: '{event_type}'. Ignoring.")
        return jsonify({"status": "ignored", "event": event_type}), 200
