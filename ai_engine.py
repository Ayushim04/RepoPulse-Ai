import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates are kept as module-level constants so they can be tuned
# independently of the business logic that calls them.
# ---------------------------------------------------------------------------

DOC_BOT_PROMPT_TEMPLATE = """
You are RepoPulse Doc-Bot, an expert assistant that answers questions about a
specific software repository. Ground your answers exclusively in the provided
context — do not invent functionality that isn't there.

If the context is insufficient, say so clearly and suggest which files to check.

**CONTEXT FROM CODEBASE:**
{context}

---

**USER QUESTION:**
{question}

---

Answer in clear, developer-friendly markdown. Include file paths and code
snippets from the context where relevant. Be concise but complete.
"""

CODE_REVIEW_PROMPT_TEMPLATE = """
You are RepoPulse AI, an expert code reviewer and open-source maintainer.
Analyze the following Git diff and return a structured review covering:

1. **Bugs & Logic Errors** — null-pointer risks, off-by-one errors, incorrect logic.
2. **Code Quality & Structure** — readability, naming, function length, SOLID principles.
3. **Conventional Commits Compliance** — does the implied commit message follow the spec?
4. **Security Concerns** — SQL injection, hardcoded credentials, missing input validation.
5. **Suggested Improvements** — 1-3 concrete, copy-pasteable suggestions.

Return a JSON object with keys:
`bugs`, `code_quality`, `conventional_commits`, `security`, `suggestions`, `overall_score` (0-100).

---
**Git Diff:**
```
{diff_text}
```
"""


class AIEngine:
    """Orchestrates all Gemini API interactions for RepoPulse."""

    def __init__(self) -> None:
        self._api_key: str | None = os.environ.get("GEMINI_API_KEY")
        self._model_name: str = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro-latest")
        self._client: Any = None

        if not self._api_key:
            logger.warning("GEMINI_API_KEY not set — AI features will return mock responses.")
        else:
            self._initialize_client()

    def _initialize_client(self) -> None:
        try:
            import google.generativeai as genai  # type: ignore[import]
            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self._model_name)
            logger.info(f"Gemini client initialised with model: {self._model_name}")
        except ImportError:
            logger.error("google-generativeai is not installed. Run: pip install google-generativeai")
        except Exception as exc:
            logger.error(f"Failed to initialise Gemini client: {exc}")

    def analyze_code_diff(self, diff_text: str) -> dict[str, Any]:
        """Send a unified diff to Gemini and return a structured review dict."""
        if not diff_text or not diff_text.strip():
            return {"error": "Diff text cannot be empty."}

        if self._client is None:
            return self._mock_review_response()

        try:
            response = self._client.generate_content(
                CODE_REVIEW_PROMPT_TEMPLATE.format(diff_text=diff_text)
            )
            return self._parse_json_response(response.text)
        except Exception as exc:
            logger.error(f"Gemini API call failed in analyze_code_diff: {exc}")
            return {"error": str(exc)}

    def answer_question(self, question: str, context: str) -> str:
        """
        Answer a codebase question using RAG context retrieved from the index.
        The context string is built by rag_ingestion.build_context_string() and
        grounded in the actual repo files, not Gemini's training data.
        """
        if not question.strip():
            return "Please provide a question."

        if self._client is None:
            return self._mock_doc_bot_response(question, context)

        prompt = DOC_BOT_PROMPT_TEMPLATE.format(context=context, question=question)
        try:
            response = self._client.generate_content(prompt)
            return response.text
        except Exception as exc:
            logger.error(f"Gemini API call failed in answer_question: {exc}")
            return f"**AI Error:** `{exc}`\n\nPlease check your API key and try again."

    def generate_pr_summary(self, pr_body: str, diff_text: str) -> str:
        """Generate a changelog-style summary of a pull request."""
        if self._client is None:
            return "**PR Summary** *(mock — GEMINI_API_KEY not set)*"

        prompt = f"""Write a concise 3-5 sentence PR summary suitable for a changelog.
Focus on what changed and why.

**PR Description:**
{pr_body or "*(none provided)*"}

**Diff (truncated):**
```
{diff_text[:8_000]}
```

Return plain markdown. No preamble."""
        try:
            return self._client.generate_content(prompt).text
        except Exception as exc:
            logger.error(f"generate_pr_summary failed: {exc}")
            return f"Summary generation failed: {exc}"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mock_doc_bot_response(self, question: str, context: str) -> str:
        has_context = context and "No relevant context" not in context
        hint = (
            "I found relevant context and would use it to answer."
            if has_context
            else "The knowledge base is empty — run `/api/ingest` first."
        )
        return (
            f"**Mock response** *(GEMINI_API_KEY not set)*\n\n"
            f"You asked: *{question}*\n\n{hint}"
        )

    def _mock_review_response(self) -> dict[str, Any]:
        return {
            "source": "mock",
            "bugs": [
                "`db.delete(user)` is called without a null check — will raise "
                "AttributeError if user_id does not exist."
            ],
            "code_quality": (
                "`delete_user` is missing a docstring and type hints. "
                "Function name follows snake_case correctly."
            ),
            "conventional_commits": (
                "Suggested format: `fix(users): add null guard before db.delete`"
            ),
            "security": (
                "No SQL injection risk (ORM). Add an authorisation check to "
                "ensure the caller has permission to delete this user."
            ),
            "suggestions": [
                "```python\nif user is None:\n    return False\ndb.delete(user)\ndb.commit()\n```"
            ],
            "overall_score": 62,
        }

    @staticmethod
    def _parse_json_response(raw_text: str) -> dict[str, Any]:
        """
        Extract JSON from the model's response. Gemini sometimes wraps the
        payload in markdown code fences, so we strip those first.
        """
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        json_str = match.group(1) if match else raw_text
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Could not parse Gemini response as JSON.")
            return {"raw_response": raw_text}
