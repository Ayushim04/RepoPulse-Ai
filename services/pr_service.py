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


def handle_pull_request_event(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Entry point for all pull_request webhook events.
    Always returns a dict — never raises — so the webhook handler can
    respond to GitHub regardless of what happens downstream.
    """
    action: str = payload.get("action", "unknown")
    pr = payload.get("pull_request", {})
    pr_number: int = pr.get("number", 0)
    author: str = pr.get("user", {}).get("login", "unknown")
    repo_name: str = payload.get("repository", {}).get("full_name", "unknown/repo")

    logger.info(f"[PR#{pr_number}] '{action}' by @{author} in {repo_name}")

    if action in ("opened", "synchronize"):
        return _run_review_pipeline(payload)

    if action == "closed":
        status = "merged" if pr.get("merged") else "closed_without_merge"
        logger.info(f"[PR#{pr_number}] {status}.")
        return {"action_taken": status, "pr_number": pr_number}

    logger.debug(f"[PR#{pr_number}] Unhandled action '{action}'.")
    return {"action_taken": "skipped", "reason": f"unhandled action: {action}"}


def _run_review_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    gh = _get_github_service()
    ai = _get_ai_engine()

    try:
        ctx: PRContext = gh.extract_pr_context(payload)
    except ValueError as exc:
        logger.error(f"Invalid webhook payload: {exc}")
        return {"action_taken": "error", "error": str(exc)}

    logger.info(f"[PR#{ctx.pr_number}] {ctx.repo_name} | {ctx.head_branch} -> {ctx.base_branch}")

    try:
        diff_text = gh.fetch_pr_diff(ctx.repo_name, ctx.pr_number)
    except GitHubAuthError as exc:
        return _error_result(ctx, "github_auth_error", str(exc))
    except GitHubRateLimitError as exc:
        logger.error(f"[PR#{ctx.pr_number}] Rate limit hit. Resets at {exc.reset_at}.")
        return _error_result(ctx, "rate_limit", str(exc))
    except GitHubAPIError as exc:
        return _error_result(ctx, "github_api_error", str(exc))

    if not diff_text or not diff_text.strip():
        logger.warning(f"[PR#{ctx.pr_number}] Empty diff — no file changes detected.")
        return {"action_taken": "skipped", "reason": "empty diff", "pr_number": ctx.pr_number}

    review_dict: dict[str, Any] = ai.analyze_code_diff(diff_text=diff_text)

    if "error" in review_dict:
        return _error_result(ctx, "ai_error", review_dict["error"])

    logger.info(f"[PR#{ctx.pr_number}] Review complete. Score: {review_dict.get('overall_score')}/100")

    comment_body = format_review_as_markdown(review=review_dict, pr_context=ctx)

    try:
        posted = gh.post_pr_comment(ctx.repo_name, ctx.pr_number, comment_body)
        comment_url = posted.get("html_url", "")
        logger.info(f"[PR#{ctx.pr_number}] Comment posted: {comment_url}")
    except GitHubAuthError as exc:
        return _error_result(ctx, "comment_post_auth_error", str(exc), extra={"ai_review": review_dict})
    except (GitHubRateLimitError, GitHubAPIError) as exc:
        return _error_result(ctx, "comment_post_failed", str(exc), extra={"ai_review": review_dict})

    return {
        "action_taken": "ai_review_posted",
        "pr_number": ctx.pr_number,
        "pr_title": ctx.pr_title,
        "repo": ctx.repo_name,
        "comment_url": comment_url,
        "ai_score": review_dict.get("overall_score"),
        "diff_size_chars": len(diff_text),
    }


def _error_result(
    ctx: PRContext,
    error_type: str,
    message: str,
    extra: dict | None = None,
) -> dict[str, Any]:
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
