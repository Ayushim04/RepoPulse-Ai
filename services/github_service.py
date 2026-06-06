import logging
import os
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
REQUEST_TIMEOUT = (5, 30)  # (connect, read) seconds


class GitHubAuthError(Exception):
    """Token is missing, invalid, or lacks the required scopes."""


class GitHubRateLimitError(Exception):
    """GitHub returned a 403/429 rate-limit response."""

    def __init__(self, reset_timestamp: int | None = None) -> None:
        self.reset_at = reset_timestamp
        suffix = f" Resets at {reset_timestamp}." if reset_timestamp else ""
        super().__init__(f"GitHub API rate limit exceeded.{suffix}")


class GitHubAPIError(Exception):
    """Any other unexpected GitHub API error."""


@dataclass
class PRContext:
    """All PR metadata extracted from a webhook payload."""
    repo_name: str
    pr_number: int
    pr_title: str
    pr_body: str
    author: str
    base_branch: str
    head_branch: str
    diff_url: str
    html_url: str


class GitHubService:
    """Thin wrapper around the GitHub REST API."""

    def __init__(self, token: str | None = None) -> None:
        resolved_token = token or os.environ.get("GITHUB_TOKEN", "")
        if not resolved_token:
            logger.warning("GITHUB_TOKEN not set — GitHub API calls will fail.")
        self._token = resolved_token
        self._session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "RepoPulse-AI/2.0",
        })
        return session

    def extract_pr_context(self, payload: dict[str, Any]) -> PRContext:
        """Parse a webhook payload into a typed PRContext. No network calls."""
        pr = payload.get("pull_request", {})
        repo = payload.get("repository", {})

        required = {"pull_request": pr, "repository": repo,
                    "pr.number": pr.get("number"), "pr.diff_url": pr.get("diff_url")}
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Webhook payload missing required fields: {missing}")

        ctx = PRContext(
            repo_name=repo["full_name"],
            pr_number=pr["number"],
            pr_title=pr.get("title", ""),
            pr_body=pr.get("body") or "",
            author=pr.get("user", {}).get("login", "unknown"),
            base_branch=pr.get("base", {}).get("ref", "main"),
            head_branch=pr.get("head", {}).get("ref", ""),
            diff_url=pr["diff_url"],
            html_url=pr.get("html_url", ""),
        )
        logger.info(f"PR context: {ctx.repo_name}#{ctx.pr_number} '{ctx.pr_title}' by @{ctx.author}")
        return ctx

    def fetch_pr_diff(self, repo_name: str, pr_number: int) -> str:
        """
        Fetch the raw unified diff for a PR.
        Truncates at 100KB to stay within Gemini's context window.
        """
        url = f"{GITHUB_API_BASE}/repos/{repo_name}/pulls/{pr_number}"
        try:
            response = self._session.get(
                url,
                headers={"Accept": "application/vnd.github.diff"},
                timeout=REQUEST_TIMEOUT,
            )
            self._raise_for_status(response, context=f"fetch diff for PR#{pr_number}")

            diff = response.text
            max_chars = 100_000
            if len(diff) > max_chars:
                logger.warning(f"Diff for PR#{pr_number} truncated from {len(diff):,} to {max_chars:,} chars.")
                diff = diff[:max_chars] + "\n\n[diff truncated]"
            return diff

        except requests.exceptions.Timeout:
            raise GitHubAPIError(f"Timed out fetching diff for PR#{pr_number}.")
        except requests.exceptions.ConnectionError as exc:
            raise GitHubAPIError(f"Network error: {exc}")

    def post_pr_comment(self, repo_name: str, pr_number: int, comment_body: str) -> dict[str, Any]:
        """
        Post a comment on a PR via the Issues Comments API.
        PRs and issues share the same comment thread in GitHub's data model.
        """
        url = f"{GITHUB_API_BASE}/repos/{repo_name}/issues/{pr_number}/comments"
        try:
            response = self._session.post(
                url,
                json={"body": comment_body},
                headers={"Accept": "application/vnd.github+json"},
                timeout=REQUEST_TIMEOUT,
            )
            self._raise_for_status(response, context=f"post comment on PR#{pr_number}")
            result = response.json()
            logger.info(f"Review comment posted: {result.get('html_url')}")
            return result

        except requests.exceptions.Timeout:
            raise GitHubAPIError(f"Timed out posting comment on PR#{pr_number}.")
        except requests.exceptions.ConnectionError as exc:
            raise GitHubAPIError(f"Network error: {exc}")

    @staticmethod
    def _raise_for_status(response: requests.Response, context: str = "GitHub API call") -> None:
        if response.ok:
            return

        status = response.status_code
        try:
            msg = response.json().get("message", response.text[:200])
        except Exception:
            msg = response.text[:200]

        remaining = response.headers.get("X-RateLimit-Remaining", "1")
        reset_ts = response.headers.get("X-RateLimit-Reset")
        reset_int = int(reset_ts) if reset_ts else None

        if status == 429 or (status == 403 and remaining == "0"):
            raise GitHubRateLimitError(reset_timestamp=reset_int)

        if status == 401:
            raise GitHubAuthError(
                f"401 Unauthorized during '{context}'. Check GITHUB_TOKEN scopes. GitHub: '{msg}'"
            )
        if status == 403:
            raise GitHubAuthError(
                f"403 Forbidden during '{context}'. Needs 'repo' scope for private repos. GitHub: '{msg}'"
            )
        if status == 404:
            raise GitHubAPIError(
                f"404 Not Found during '{context}'. Check repo name, PR number, and token access."
            )

        raise GitHubAPIError(f"HTTP {status} during '{context}': {msg}")


def format_review_as_markdown(review: dict[str, Any], pr_context: PRContext) -> str:
    """Render the AI review dict as a GitHub-flavoured markdown comment."""
    score: int = review.get("overall_score", 0)

    if score >= 80:
        badge = f"🟢 **{score}/100** — Looks good!"
    elif score >= 60:
        badge = f"🟡 **{score}/100** — Some improvements suggested"
    else:
        badge = f"🔴 **{score}/100** — Needs attention before merging"

    def _bullets(value: str | list | None) -> str:
        if not value:
            return "_No issues found._"
        if isinstance(value, list):
            return "\n".join(f"- {item}" for item in value)
        return value

    mock_note = " *(mock — GEMINI_API_KEY not set)*" if review.get("source") == "mock" else ""

    return f"""## 🤖 RepoPulse AI — Code Review{mock_note}

> **{pr_context.pr_title}** &nbsp;·&nbsp; `{pr_context.head_branch}` → `{pr_context.base_branch}`

**Overall Score:** {badge}

---

### 🐛 Bugs & Logic Errors
{_bullets(review.get("bugs"))}

### 🏗️ Code Quality & Structure
{_bullets(review.get("code_quality"))}

### 📝 Conventional Commits
{_bullets(review.get("conventional_commits"))}

### 🔒 Security
{_bullets(review.get("security"))}

### 💡 Suggested Improvements
{_bullets(review.get("suggestions"))}

---
<sub>RepoPulse AI · Gemini 1.5 Pro</sub>
"""
