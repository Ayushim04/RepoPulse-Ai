"""
RepoPulse AI - GitHub Service (Phase 2)
=========================================
Handles all communication with the GitHub REST API:
  - Extracting PR metadata from webhook payloads
  - Fetching raw unified diffs for code review
  - Posting AI-generated review comments back onto PRs

Authentication:
  - Uses a GitHub Personal Access Token (PAT) from the GITHUB_TOKEN env var.
  - Token needs `repo` scope for private repos, `public_repo` for public ones.

Rate Limiting:
  - GitHub's REST API allows 5,000 requests/hour for authenticated users.
  - This module detects 403/429 responses and raises a clear RateLimitError
    so the caller can decide whether to retry or surface the error to the user.

Dependencies:
  - requests (pure HTTP — no heavy SDK needed for these operations)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_API_BASE = "https://api.github.com"
# Timeout for all outbound GitHub API requests (connect, read) in seconds
REQUEST_TIMEOUT = (5, 30)


# ---------------------------------------------------------------------------
# Custom Exceptions — make caller error-handling explicit and readable
# ---------------------------------------------------------------------------

class GitHubAuthError(Exception):
    """Raised when the GitHub token is missing, invalid, or lacks permissions."""


class GitHubRateLimitError(Exception):
    """Raised when GitHub returns a 403/429 rate-limit response."""

    def __init__(self, reset_timestamp: int | None = None) -> None:
        self.reset_at = reset_timestamp
        reset_str = (
            f" Rate limit resets at Unix timestamp {reset_timestamp}."
            if reset_timestamp
            else ""
        )
        super().__init__(f"GitHub API rate limit exceeded.{reset_str}")


class GitHubAPIError(Exception):
    """Raised for any other unexpected GitHub API error."""


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class PRContext:
    """Structured container for all PR data needed by the AI review pipeline."""
    repo_name: str          # e.g. "acme-org/core-api"
    pr_number: int          # e.g. 142
    pr_title: str           # e.g. "feat: add OAuth2 provider support"
    pr_body: str            # The PR description (may be empty)
    author: str             # GitHub login of the PR author
    base_branch: str        # e.g. "main"
    head_branch: str        # e.g. "feature/oauth2"
    diff_url: str           # URL to fetch the unified diff
    html_url: str           # Human-readable PR URL for log messages


# ---------------------------------------------------------------------------
# GitHub API Client
# ---------------------------------------------------------------------------

class GitHubService:
    """
    Thin wrapper around the GitHub REST API for RepoPulse AI operations.

    Usage:
        svc = GitHubService()                          # reads token from env
        ctx = svc.extract_pr_context(webhook_payload)
        diff = svc.fetch_pr_diff(ctx.repo_name, ctx.pr_number)
        svc.post_pr_comment(ctx.repo_name, ctx.pr_number, "Great work!")
    """

    def __init__(self, token: str | None = None) -> None:
        """
        Initialise the service with a GitHub PAT.

        Args:
            token: GitHub Personal Access Token. If None, falls back to the
                   GITHUB_TOKEN environment variable.

        Raises:
            GitHubAuthError: If no token is found at all (fail-fast so the
                             developer knows immediately at startup).
        """
        import os
        resolved_token = token or os.environ.get("GITHUB_TOKEN")

        if not resolved_token:
            # Log a clear warning but do NOT crash the app — let individual
            # method calls raise when they actually need the token.
            logger.warning(
                "GITHUB_TOKEN is not set. GitHub API calls will fail. "
                "Set it in your .env file."
            )
            resolved_token = ""  # empty string; will cause a 401 on use

        self._token: str = resolved_token
        self._session: requests.Session = self._build_session()

    def _build_session(self) -> requests.Session:
        """
        Build a reusable requests.Session with auth headers pre-set.
        Reusing a session is more efficient than creating one per call.
        """
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {self._token}",
                # Request the v3 JSON format
                "Accept": "application/vnd.github+json",
                # Pin API version for stability
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "RepoPulse-AI/2.0",
            }
        )
        return session

    # -----------------------------------------------------------------------
    # Public Methods
    # -----------------------------------------------------------------------

    def extract_pr_context(self, payload: dict[str, Any]) -> PRContext:
        """
        Parse a raw GitHub `pull_request` webhook payload into a typed PRContext.

        This is a pure data-extraction function — no network calls.

        Args:
            payload: The JSON-decoded webhook payload dict.

        Returns:
            A populated PRContext dataclass.

        Raises:
            ValueError: If required fields are missing from the payload
                        (guards against malformed/unexpected payloads).
        """
        pr = payload.get("pull_request", {})
        repo = payload.get("repository", {})

        # Validate required fields are present before we proceed
        required = {
            "pull_request": pr,
            "repository": repo,
            "pr.number": pr.get("number"),
            "pr.diff_url": pr.get("diff_url"),
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(
                f"Webhook payload is missing required fields: {missing}. "
                f"Ensure the webhook is sending the full pull_request event."
            )

        ctx = PRContext(
            repo_name=repo["full_name"],
            pr_number=pr["number"],
            pr_title=pr.get("title", ""),
            pr_body=pr.get("body") or "",         # body can be null in GitHub's API
            author=pr.get("user", {}).get("login", "unknown"),
            base_branch=pr.get("base", {}).get("ref", "main"),
            head_branch=pr.get("head", {}).get("ref", ""),
            diff_url=pr["diff_url"],
            html_url=pr.get("html_url", ""),
        )

        logger.info(
            f"Extracted PR context: {ctx.repo_name}#{ctx.pr_number} "
            f"'{ctx.pr_title}' by @{ctx.author} | "
            f"{ctx.head_branch} → {ctx.base_branch}"
        )
        return ctx

    def fetch_pr_diff(self, repo_name: str, pr_number: int) -> str:
        """
        Fetch the unified diff of a pull request from the GitHub API.

        Uses the `application/vnd.github.diff` Accept header to get the
        raw patch format, which is what the AI engine expects.

        Args:
            repo_name:  Full repo name, e.g. "acme-org/core-api".
            pr_number:  The pull request number.

        Returns:
            The raw unified diff as a string.

        Raises:
            GitHubAuthError:      On 401/403 auth errors.
            GitHubRateLimitError: On 429 or 403 rate-limit responses.
            GitHubAPIError:       On any other non-200 response.
        """
        url = f"{GITHUB_API_BASE}/repos/{repo_name}/pulls/{pr_number}"
        logger.info(f"Fetching diff for {repo_name}#{pr_number} from {url}")

        # GitHub returns the raw diff when we set this Accept header
        headers = {"Accept": "application/vnd.github.diff"}

        try:
            response = self._session.get(
                url, headers=headers, timeout=REQUEST_TIMEOUT
            )
            self._raise_for_github_status(response, context=f"fetch diff for PR#{pr_number}")

            diff_text = response.text
            logger.info(
                f"Diff fetched for {repo_name}#{pr_number}: "
                f"{len(diff_text):,} chars, "
                f"~{diff_text.count(chr(10))} lines"
            )

            # Safety guard: very large diffs can exceed model context windows.
            # Truncate to ~100KB to stay well within Gemini's limit.
            max_chars = 100_000
            if len(diff_text) > max_chars:
                logger.warning(
                    f"Diff for PR#{pr_number} is {len(diff_text):,} chars — "
                    f"truncating to {max_chars:,} to fit model context window."
                )
                diff_text = diff_text[:max_chars] + "\n\n[...diff truncated by RepoPulse AI]"

            return diff_text

        except requests.exceptions.Timeout:
            raise GitHubAPIError(
                f"Request timed out fetching diff for {repo_name}#{pr_number}. "
                "GitHub may be slow — retry in a moment."
            )
        except requests.exceptions.ConnectionError as exc:
            raise GitHubAPIError(
                f"Network error connecting to GitHub API: {exc}"
            )

    def post_pr_comment(
        self,
        repo_name: str,
        pr_number: int,
        comment_body: str,
    ) -> dict[str, Any]:
        """
        Post a comment on a GitHub PR using the Issues Comments API.
        (PRs share the same comment thread as Issues in GitHub's model.)

        Args:
            repo_name:    Full repo name, e.g. "acme-org/core-api".
            pr_number:    The pull request number.
            comment_body: The markdown-formatted comment text to post.

        Returns:
            The GitHub API response dict for the created comment
            (includes `id`, `html_url`, etc.).

        Raises:
            GitHubAuthError:      On 401/403 auth errors.
            GitHubRateLimitError: On rate-limit errors.
            GitHubAPIError:       On any other API error.
        """
        url = f"{GITHUB_API_BASE}/repos/{repo_name}/issues/{pr_number}/comments"
        logger.info(
            f"Posting AI review comment on {repo_name}#{pr_number} "
            f"({len(comment_body)} chars)"
        )

        # Reset Accept header back to JSON for this write operation
        headers = {"Accept": "application/vnd.github+json"}

        try:
            response = self._session.post(
                url,
                json={"body": comment_body},
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            self._raise_for_github_status(
                response, context=f"post comment on {repo_name}#{pr_number}"
            )

            result = response.json()
            logger.info(
                f"Comment posted successfully on {repo_name}#{pr_number}: "
                f"{result.get('html_url')}"
            )
            return result

        except requests.exceptions.Timeout:
            raise GitHubAPIError(
                f"Request timed out posting comment on {repo_name}#{pr_number}."
            )
        except requests.exceptions.ConnectionError as exc:
            raise GitHubAPIError(f"Network error connecting to GitHub API: {exc}")

    # -----------------------------------------------------------------------
    # Private Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _raise_for_github_status(
        response: requests.Response,
        context: str = "GitHub API call",
    ) -> None:
        """
        Inspect a GitHub API response and raise a typed exception for errors.

        This replaces the generic requests.raise_for_status() to give callers
        actionable, domain-specific exceptions instead of HTTPError.

        Args:
            response: The requests.Response object to inspect.
            context:  A short description of what we were doing (for log messages).
        """
        if response.ok:
            return  # 2xx — all good

        status = response.status_code

        # Try to extract GitHub's error message from the response body
        try:
            github_message = response.json().get("message", response.text[:200])
        except Exception:
            github_message = response.text[:200]

        # Check for rate-limit BEFORE 403 (GitHub returns 403 for both auth AND limits)
        rate_limit_remaining = response.headers.get("X-RateLimit-Remaining", "1")
        rate_limit_reset = response.headers.get("X-RateLimit-Reset")
        reset_ts = int(rate_limit_reset) if rate_limit_reset else None

        if status == 429 or (status == 403 and rate_limit_remaining == "0"):
            logger.error(
                f"Rate limit exceeded during '{context}'. "
                f"Reset at: {reset_ts}. Message: {github_message}"
            )
            raise GitHubRateLimitError(reset_timestamp=reset_ts)

        if status == 401:
            logger.error(
                f"GitHub auth failed during '{context}'. "
                "Check that GITHUB_TOKEN is set and has the correct scopes."
            )
            raise GitHubAuthError(
                f"GitHub returned 401 Unauthorized during '{context}'. "
                f"GitHub says: '{github_message}'"
            )

        if status == 403:
            logger.error(
                f"GitHub permission denied during '{context}': {github_message}"
            )
            raise GitHubAuthError(
                f"GitHub returned 403 Forbidden during '{context}'. "
                "Check token scopes (needs 'repo' for private repos). "
                f"GitHub says: '{github_message}'"
            )

        if status == 404:
            raise GitHubAPIError(
                f"GitHub returned 404 during '{context}'. "
                "Check repo name/PR number and that the token can access this repo."
            )

        if status == 422:
            raise GitHubAPIError(
                f"GitHub returned 422 Unprocessable Entity during '{context}': "
                f"{github_message}"
            )

        # Catch-all for any other non-2xx status
        logger.error(
            f"Unexpected GitHub API error during '{context}': "
            f"HTTP {status} — {github_message}"
        )
        raise GitHubAPIError(
            f"GitHub API error {status} during '{context}': {github_message}"
        )


# ---------------------------------------------------------------------------
# Module-level helpers — format AI review dict into a GitHub markdown comment
# ---------------------------------------------------------------------------

def format_review_as_markdown(
    review: dict[str, Any],
    pr_context: PRContext,
) -> str:
    """
    Convert the structured AI review dict into a clean GitHub markdown comment.

    This keeps formatting logic out of both the AI engine and the service,
    making it easy to tweak the comment appearance independently.

    Args:
        review:     The dict returned by AIEngine.analyze_code_diff().
        pr_context: Metadata about the PR for contextual header text.

    Returns:
        A markdown string ready to be posted as a GitHub PR comment.
    """
    score: int = review.get("overall_score", 0)

    # Choose a visual indicator based on score bands
    if score >= 80:
        score_badge = f"🟢 **{score}/100** — Looks good!"
    elif score >= 60:
        score_badge = f"🟡 **{score}/100** — Some improvements suggested"
    else:
        score_badge = f"🔴 **{score}/100** — Needs attention before merging"

    # Handle both string and list values for flexible AI output
    def _to_bullet_list(value: str | list | None) -> str:
        if not value:
            return "_No issues found._"
        if isinstance(value, list):
            return "\n".join(f"- {item}" for item in value)
        return value

    bugs           = _to_bullet_list(review.get("bugs"))
    quality        = _to_bullet_list(review.get("code_quality"))
    commits        = _to_bullet_list(review.get("conventional_commits"))
    security       = _to_bullet_list(review.get("security"))
    suggestions    = _to_bullet_list(review.get("suggestions"))
    source_note    = " _(mock response — no API key set)_" if review.get("source") == "mock" else ""

    return f"""## 🤖 RepoPulse AI — Code Review{source_note}

> Automated review for **{pr_context.pr_title}** ({pr_context.head_branch} → {pr_context.base_branch})

**Overall Score:** {score_badge}

---

### 🐛 Bugs & Logic Errors
{bugs}

### 🏗️ Code Quality & Structure
{quality}

### 📝 Conventional Commits
{commits}

### 🔒 Security
{security}

### 💡 Suggested Improvements
{suggestions}

---
<sub>Generated by [RepoPulse AI](https://github.com) · Model: Gemini 1.5 Pro · Phase 2</sub>
"""
