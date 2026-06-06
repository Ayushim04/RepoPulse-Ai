import hashlib
import hmac
import json
import logging
from typing import Any

from flask import Blueprint, Response, current_app, jsonify, request

from services.pr_service import handle_pull_request_event

logger = logging.getLogger(__name__)

webhook_bp = Blueprint("webhook", __name__)


def _verify_github_signature(payload_body: bytes, signature_header: str | None) -> bool:
    """
    Validate the X-Hub-Signature-256 header using HMAC-SHA256.
    hmac.compare_digest is used to prevent timing attacks.
    """
    secret: str | None = current_app.config.get("GITHUB_WEBHOOK_SECRET")

    if not secret:
        logger.warning("GITHUB_WEBHOOK_SECRET not set — skipping signature check (dev only).")
        return True

    if not signature_header or not signature_header.startswith("sha256="):
        logger.warning("Request missing a valid X-Hub-Signature-256 header.")
        return False

    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), payload_body, hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, signature_header)


@webhook_bp.route("/webhook/github", methods=["POST"])
def github_webhook() -> tuple[Response, int]:
    """POST /api/webhook/github — receives and routes GitHub webhook events."""
    raw_body: bytes = request.get_data()
    signature: str | None = request.headers.get("X-Hub-Signature-256")
    event_type: str | None = request.headers.get("X-GitHub-Event")
    delivery_id: str = request.headers.get("X-GitHub-Delivery", "unknown")

    logger.info(f"Webhook received | delivery={delivery_id} | event={event_type}")

    if not _verify_github_signature(raw_body, signature):
        logger.error(f"Signature mismatch for delivery={delivery_id}.")
        return jsonify({"status": "error", "message": "Invalid signature."}), 403

    try:
        payload: dict[str, Any] = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        logger.error(f"Malformed JSON payload: {exc}")
        return jsonify({"status": "error", "message": "Malformed JSON payload."}), 400

    if event_type == "pull_request":
        logger.info(
            f"PR event | action='{payload.get('action')}' | "
            f"PR#{payload.get('pull_request', {}).get('number')}"
        )
        result = handle_pull_request_event(payload)
        return jsonify({"status": "accepted", "detail": result}), 202

    elif event_type == "ping":
        logger.info("Ping received — webhook is active.")
        return jsonify({"status": "ok", "message": "Pong!"}), 200

    else:
        logger.debug(f"Unhandled event type: '{event_type}'.")
        return jsonify({"status": "ignored", "event": event_type}), 200
