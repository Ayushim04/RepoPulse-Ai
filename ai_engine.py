"""
RepoPulse AI - AI Orchestration Engine (Gemini)
=================================================
Centralizes all interactions with the Google Gemini API.

Design principles:
  - API key is loaded exclusively from the environment — never hardcoded.
  - Each public method maps to one distinct AI task (review, summarize, etc.)
  - Prompts are structured as constants so they're easy to tune without
    touching business logic.
  - In Phase 2, add retry logic (tenacity) and response caching (Redis).
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

DOC_BOT_PROMPT_TEMPLATE = """
You are RepoPulse Doc-Bot, an expert assistant that answers questions about a
specific software repository. Your answers must be grounded exclusively in the
provided context snippets — do not invent functionality that isn't there.

If the context does not contain enough information to answer, say so clearly
and suggest what files to look at.

**CONTEXT FROM CODEBASE:**
{context}

---

**USER QUESTION:**
{question}

---

Answer in clear, developer-friendly markdown. Include relevant file paths and
code snippets from the context where helpful. Be concise but complete.
"""

CODE_REVIEW_PROMPT_TEMPLATE = """
You are RepoPulse AI, an expert code reviewer and open-source maintainer assistant.
Your goal is to provide concise, actionable, and constructive feedback.

Analyze the following Git diff and return a structured review covering:

1.  **Bugs & Logic Errors**: Identify any potential bugs, null-pointer issues,
    off-by-one errors, or incorrect logic.

2.  **Code Quality & Structure**: Comment on readability, naming conventions,
    function length, and adherence to SOLID principles where applicable.

3.  **Conventional Commits Compliance**: Check if the implied commit message
    (inferred from the diff) follows the Conventional Commits specification
    (e.g., `feat:`, `fix:`, `refactor:`, `chore:`).

4.  **Security Concerns**: Flag any obvious security issues (e.g., SQL injection,
    hardcoded credentials, missing input validation).

5.  **Suggested Improvements**: Provide 1-3 concrete, copy-pasteable code
    suggestions where improvements are possible.

Return your response as a JSON object with the keys:
`bugs`, `code_quality`, `conventional_commits`, `security`, `suggestions`, `overall_score` (0-100).

---
**Git Diff:**
```
{diff_text}
```
"""

# ---------------------------------------------------------------------------
# AIEngine Class
# ---------------------------------------------------------------------------


class AIEngine:
    """
    Orchestrates all AI-powered features of RepoPulse using Google Gemini.

    Usage:
        engine = AIEngine()
        review = engine.analyze_code_diff(diff_text="+ def foo(): pass")
    """

    def __init__(self) -> None:
        """
        Initialize the Gemini client.

        Reads the API key from the `GEMINI_API_KEY` environment variable.
        Raises `EnvironmentError` if the key is absent.
        """
        self._api_key: str | None = os.environ.get("GEMINI_API_KEY")
        self._model_name: str = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro-latest")
        self._client: Any = None  # Holds the initialized Gemini GenerativeModel

        if not self._api_key:
            logger.warning(
                "GEMINI_API_KEY environment variable is not set. "
                "AI features will return mock responses."
            )
        else:
            self._initialize_client()

    def _initialize_client(self) -> None:
        """
        Lazily configure and store the Gemini GenerativeModel client.

        Wrapped in try/except so a bad API key doesn't crash the whole app —
        it degrades gracefully to mock mode instead.
        """
        try:
            import google.generativeai as genai  # type: ignore[import]

            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self._model_name)
            logger.info(f"Gemini client initialized with model: {self._model_name}")
        except ImportError:
            logger.error(
                "google-generativeai package is not installed. "
                "Run: pip install google-generativeai"
            )
        except Exception as exc:
            logger.error(f"Failed to initialize Gemini client: {exc}")

    # -----------------------------------------------------------------------
    # Public Methods — AI Tasks
    # -----------------------------------------------------------------------

    def analyze_code_diff(self, diff_text: str) -> dict[str, Any]:
        """
        Send a code diff to Gemini for a structured review.

        Args:
            diff_text: A string containing the Git diff to be reviewed.

        Returns:
            A dict with review categories as keys. Returns a mock response
            if the Gemini client is unavailable.
        """
        if not diff_text or not diff_text.strip():
            logger.warning("analyze_code_diff called with empty diff. Returning early.")
            return {"error": "Diff text cannot be empty."}

        if self._client is None:
            logger.debug("Gemini client unavailable — returning mock review response.")
            return self._mock_review_response(diff_text)

        prompt = CODE_REVIEW_PROMPT_TEMPLATE.format(diff_text=diff_text)

        try:
            logger.debug(f"Sending diff to Gemini ({len(diff_text)} chars)...")
            response = self._client.generate_content(prompt)
            raw_text: str = response.text

            # Attempt to parse the structured JSON from the model's response
            return self._parse_json_response(raw_text)

        except Exception as exc:
            logger.error(f"Gemini API call failed: {exc}")
            return {"error": str(exc), "raw_diff_length": len(diff_text)}

    def answer_question(self, question: str, context: str) -> str:
        """
        Answer a user question about the codebase using RAG context.

        This is the core of the Doc-Bot feature. The context string is
        retrieved by rag_ingestion.find_relevant_chunks() and injected
        into the prompt so Gemini answers based on the actual codebase,
        not just its training data.

        Args:
            question: The user's natural-language question.
            context:  Relevant code/doc snippets retrieved from the RAG index.

        Returns:
            A markdown-formatted answer string. Falls back to a mock
            response if the Gemini client is unavailable.
        """
        if not question.strip():
            return "Please provide a question."

        if self._client is None:
            logger.debug("Gemini client unavailable — returning mock Doc-Bot response.")
            return self._mock_doc_bot_response(question, context)

        prompt = DOC_BOT_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        try:
            logger.debug(
                f"Sending Doc-Bot query to Gemini | "
                f"question length: {len(question)} chars | "
                f"context length: {len(context)} chars"
            )
            response = self._client.generate_content(prompt)
            answer = response.text
            logger.info(f"Doc-Bot answer generated ({len(answer)} chars).")
            return answer

        except Exception as exc:
            logger.error(f"Gemini API call failed in answer_question: {exc}")
            return (
                f"⚠️ **AI Error**: The Gemini API returned an error: `{exc}`\n\n"
                "Please check your API key and try again."
            )

    def generate_pr_summary(self, pr_body: str, diff_text: str) -> str:
        """
        Generate a concise, human-readable summary of a pull request.

        Args:
            pr_body:   The PR description written by the author.
            diff_text: The full code diff.

        Returns:
            A markdown-formatted summary string.
        """
        if self._client is None:
            return (
                "**PR Summary** _(mock — no API key set)_\n\n"
                "This PR introduces several code changes. "
                "Connect your Gemini API key for a real AI summary."
            )

        prompt = f"""
You are RepoPulse AI. Write a concise, 3-5 sentence summary of this pull request
suitable for a project changelog. Focus on WHAT changed and WHY.

**PR Description by author:**
{pr_body or "_(no description provided)_"}

**Code Diff:**
```
{diff_text[:8_000]}
```

Return plain markdown. Start directly with the summary — no preamble.
"""
        try:
            response = self._client.generate_content(prompt)
            return response.text
        except Exception as exc:
            logger.error(f"generate_pr_summary failed: {exc}")
            return f"Summary generation failed: {exc}"

    def score_documentation_health(self, repo_files: list[str]) -> dict[str, Any]:
        """
        Analyze README, docstrings, and inline comments to produce
        a documentation health score and improvement recommendations.

        Args:
            repo_files: List of file contents as strings.

        Returns:
            A dict with score and recommendations.
        """
        logger.info("score_documentation_health called — not yet implemented (Phase 3).")
        return {"score": 0, "recommendations": [], "status": "phase_3_pending"}

    # -----------------------------------------------------------------------
    # Private Helpers
    # -----------------------------------------------------------------------

    def _mock_doc_bot_response(self, question: str, context: str) -> str:
        """
        Return a realistic mock Doc-Bot answer for development/testing.
        """
        has_context = context and "No relevant context" not in context
        context_note = (
            "I found relevant context in your codebase and would normally use it."
            if has_context
            else "The knowledge base appears empty. Run `/api/ingest` first."
        )
        return (
            f"**Mock Answer** _(Gemini API key not set)_\n\n"
            f"You asked: _{question}_\n\n"
            f"{context_note}\n\n"
            "Once you add your `GEMINI_API_KEY` to `.env`, "
            "I'll provide a real, codebase-grounded answer here."
        )

    def _mock_review_response(self, diff_text: str) -> dict[str, Any]:
        """
        Return a realistic mock response for development/testing when
        the Gemini API key is not configured.
        """
        return {
            "source": "mock",
            "bugs": [
                "Potential NullPointerException on line +13: `db.delete(user)` is called "
                "without checking if `user` is None first. If `user_id` doesn't exist "
                "in the database, this will raise an AttributeError."
            ],
            "code_quality": (
                "Function `delete_user` lacks a docstring. Consider adding type hints "
                "to parameters and return value. The function name follows snake_case correctly."
            ),
            "conventional_commits": (
                "Inferred commit type: `feat` or `fix`. Ensure the commit message follows "
                "the format: `fix(users): add null check before delete in delete_user`."
            ),
            "security": (
                "No direct SQL injection risk detected (using ORM). However, consider "
                "adding an authorization check — ensure the caller has permission to "
                "delete the specified user."
            ),
            "suggestions": [
                "Add a null guard:\\n```python\\ndef delete_user(user_id: int) -> bool:\\n"
                "    user = db.query(User).filter(User.id == user_id).first()\\n"
                "    if user is None:\\n"
                "        return False\\n"
                "    db.delete(user)\\n"
                "    db.commit()\\n"
                "    return True\\n```"
            ],
            "overall_score": 62,
        }

    @staticmethod
    def _parse_json_response(raw_text: str) -> dict[str, Any]:
        """
        Attempt to extract and parse a JSON block from the model's text output.

        Gemini may wrap the JSON in markdown code fences — this strips them.

        Args:
            raw_text: The raw string response from the Gemini API.

        Returns:
            Parsed dict, or a fallback dict with the raw text on failure.
        """
        import json
        import re

        # Strip markdown code fences if present (```json ... ```)
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        json_str = json_match.group(1) if json_match else raw_text

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Could not parse Gemini response as JSON. Returning raw text.")
            return {"raw_response": raw_text}
