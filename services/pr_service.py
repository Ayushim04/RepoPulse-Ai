"""
RepoPulse AI - Pull Request Orchestration Service (Phase 2)
============================================================
Coordinates the full PR review pipeline:
  1. Extract structured metadata from the webhook payload
  2. Fetch the real unified diff from the GitHub API
  3. Run the diff through the Gemini AI code review engine
  4. Format the review as a GitHub markdown comment
  5. Post the comment back to the PR

This module is intentionally thin — it delegates to:
  - github_service.GitHubService  (API I/O)
  - ai_engine.AIEngine            (AI inference)
  - github_service.format_review_as_markdown (formatting)

Error handling philosophy:
  - GitHubAuthError and GitHubRateLimitError are logged loudly and
    returned as structured error dicts so the webhook endpoint can
    respond with the right HTTP status code and a useful message.
  - All other exceptions are caught, logged, and returned gracefully
    so a single broken PR never crashes the webhook handler.
"""

import logging
from typing import Any

from ai_engine import AIEngine
from services.github_service import (
    GitHubAPIError,
    GitHubAuthError,
    GitHubRateLimitError,
    GitHubService,
    PRContext,
    format_review_as_markdown,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons — initialised once, reused across all webhook events
# ---------------------------------------------------------------------------

_github_service: GitHubService | None = None
_ai_engine: AIEngine | None = None


def _get_github_service() -> GitHubService:
    global _github_service
    if _github_service is None:
        _github_service = GitHubService()
    return _github_service


def _get_ai_engine() -> AIEngine:
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = AIEngine()
    return _ai_engine


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def handle_pull_request_event(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Main entry point for all `pull_request` GitHub webhook events.

    Routes each event action to the appropriate handler:
      - opened / synchronize → full AI code review pipeline
      - closed / merged      → log and skip
      - everything else      → ignore gracefully

    Args:
        payload: The parsed GitHub webhook JSON payload dict.

    Returns:
        A result dict describing what action was taken.
        Always returns a dict — never raises — so the webhook endpoint
        can always send a response to GitHub.
    """
    action: str = payload.get("action", "unknown")
    pr_data: dict = payload.get("pull_request", {})
    pr_number: int = pr_data.get("number", 0)
    pr_title: str = pr_data.get("title", "N/A")
    author: str = pr_data.get("user", {}).get("login", "unknown")
    repo_name: str = payload.get("repository", {}).get("full_name", "unknown/repo")

    logger.info(
        f"[PR#{pr_number}] Handling '{action}' event | "
        f"'{pr_title}' by @{author} in {repo_name}"
    )

    if action in ("opened", "synchronize"):
        return _run_review_pipeline(payload)

    elif action == "closed":
        merged: bool = pr_data.get("merged", False)
        status = "merged" if merged else "closed_without_merge"
        logger.info(f"[PR#{pr_number}] {status} — no review needed.")
        return {"action_taken": status, "pr_number": pr_number}

    else:
        logger.debug(f"[PR#{pr_number}] Action '{action}' not handled. Skipping.")
        return {"action_taken": "skipped", "reason": f"unhandled action: {action}"}


# ---------------------------------------------------------------------------
# Private — Review Pipeline
# ---------------------------------------------------------------------------

def _run_review_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Execute the full AI review pipeline for an opened or synchronised PR.

    Pipeline steps:
      1. Extract PR context from payload (no network)
      2. Fetch diff from GitHub API
      3. Analyse diff with Gemini
      4. Format review as GitHub markdown
      5. Post review comment to the PR

    Args:
        payload: Raw webhook payload dict.

    Returns:
        A result dict with keys: action_taken, pr_number, pr_title,
        comment_url (if successful), and error (if something failed).
    """
    gh = _get_github_service()
    ai = _get_ai_engine()

    # ---- Step 1: Extract PR context ----------------------------------------
    try:
        ctx: PRContext = gh.extract_pr_context(payload)
    except ValueError as exc:
        logger.error(f"Failed to extract PR context from payload: {exc}")
        return {
            "action_taken": "error",
            "error": f"Invalid webhook payload: {exc}",
        }

    logger.info(
        f"[PR#{ctx.pr_number}] Pipeline started | "
        f"{ctx.repo_name} | {ctx.head_branch} → {ctx.base_branch}"
    )

    # ---- Step 2: Fetch the diff ---------------------------------------------
    try:
        diff_text = gh.fetch_pr_diff(ctx.repo_name, ctx.pr_number)
    except GitHubAuthError as exc:
        logger.error(f"[PR#{ctx.pr_number}] GitHub auth failed fetching diff: {exc}")
        return _error_result(ctx, "github_auth_error", str(exc))

    except GitHubRateLimitError as exc:
        logger.error(
            f"[PR#{ctx.pr_number}] GitHub rate limit hit. "
            f"Reset at: {exc.reset_at}"
        )
        return _error_result(ctx, "rate_limit", str(exc))

    except GitHubAPIError as exc:
        logger.error(f"[PR#{ctx.pr_number}] GitHub API error fetching diff: {exc}")
        return _error_result(ctx, "github_api_error", str(exc))

    if not diff_text or not diff_text.strip():
        logger.warning(
            f"[PR#{ctx.pr_number}] Diff is empty — PR may have no file changes. "
            "Skipping review."
        )
        return {
            "action_taken": "skipped",
            "reason": "empty diff",
            "pr_number": ctx.pr_number,
        }

    # ---- Step 3: AI code review ---------------------------------------------
    logger.info(
        f"[PR#{ctx.pr_number}] Sending diff to Gemini "
        f"({len(diff_text):,} chars)..."
    )
    review_dict: dict[str, Any] = ai.analyze_code_diff(diff_text=diff_text)

    if "error" in review_dict:
        logger.error(
            f"[PR#{ctx.pr_number}] AI review returned an error: "
            f"{review_dict['error']}"
        )
        return _error_result(ctx, "ai_error", review_dict["error"])

    logger.info(
        f"[PR#{ctx.pr_number}] AI review complete. "
        f"Score: {review_dict.get('overall_score', 'N/A')}/100"
    )

    # ---- Step 4: Format as GitHub markdown ----------------------------------
    comment_body = format_review_as_markdown(review=review_dict, pr_context=ctx)

    # ---- Step 5: Post comment to GitHub -------------------------------------
    try:
        posted_comment = gh.post_pr_comment(
            repo_name=ctx.repo_name,
            pr_number=ctx.pr_number,
            comment_body=comment_body,
        )
        comment_url: str = posted_comment.get("html_url", "")
        logger.info(
            f"[PR#{ctx.pr_number}] ✅ Review comment posted: {comment_url}"
        )

    except GitHubAuthError as exc:
        logger.error(f"[PR#{ctx.pr_number}] Auth failed posting comment: {exc}")
        # Review was generated — return it even if posting failed
        return _error_result(
            ctx, "comment_post_auth_error", str(exc),
            extra={"ai_review": review_dict}
        )

    except GitHubRateLimitError as exc:
        logger.error(f"[PR#{ctx.pr_number}] Rate limit posting comment: {exc}")
        return _error_result(
            ctx, "rate_limit", str(exc),
            extra={"ai_review": review_dict}
        )

    except GitHubAPIError as exc:
        logger.error(f"[PR#{ctx.pr_number}] API error posting comment: {exc}")
        return _error_result(
            ctx, "comment_post_failed", str(exc),
            extra={"ai_review": review_dict}
        )

    return {
        "action_taken": "ai_review_posted",
        "pr_number": ctx.pr_number,
        "pr_title": ctx.pr_title,
        "repo": ctx.repo_name,
        "comment_url": comment_url,
        "ai_score": review_dict.get("overall_score"),
        "diff_size_chars": len(diff_text),
    }


# ---------------------------------------------------------------------------
# Private — Helpers
# ---------------------------------------------------------------------------

def _error_result(
    ctx: PRContext,
    error_type: str,
    message: str,
    extra: dict | None = None,
) -> dict[str, Any]:
    """Build a consistent error result dict for the webhook response."""
    result: dict[str, Any] = {
        "action_taken": "error",
        "error_type": error_type,
        "error": message,
        "pr_number": ctx.pr_number,
        "pr_title": ctx.pr_title,
        "repo": ctx.repo_name,
    }
    if extra:
        result.update(extra)
    return result
