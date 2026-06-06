import logging
import random
import time

from flask import Blueprint, Response, jsonify, render_template

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint("dashboard", __name__)

# Seeded review data used for the live dashboard before real webhook events arrive.
# In production this would be replaced by a database-backed store.
_review_store: list[dict] = [
    {
        "id": "pr-142",
        "pr_number": 142,
        "pr_title": "feat: add OAuth2 provider support",
        "author": "jsmith",
        "repo": "acme-org/core-api",
        "base_branch": "main",
        "head_branch": "feature/oauth2",
        "status": "reviewing",
        "score": 78,
        "timestamp": time.time() - 7200,
        "review": {
            "bugs": ["Potential NullPointerException: `db.delete(user)` called without null check."],
            "code_quality": "Function `delete_user` lacks a docstring and type hints. snake_case naming is correct.",
            "conventional_commits": "Suggested prefix: `feat(auth):` — follows Conventional Commits spec.",
            "security": "No SQL injection risk (ORM). Add authorisation check to verify caller permissions.",
            "suggestions": ["```python\nif user is None:\n    return False\ndb.delete(user)\n```"],
            "overall_score": 78,
        },
    },
    {
        "id": "pr-141",
        "pr_number": 141,
        "pr_title": "fix: resolve null pointer in user deletion",
        "author": "alee",
        "repo": "acme-org/core-api",
        "base_branch": "main",
        "head_branch": "fix/null-pointer",
        "status": "approved",
        "score": 92,
        "timestamp": time.time() - 18000,
        "review": {
            "bugs": [],
            "code_quality": "Clean, concise fix. Type hints present. Good docstring coverage.",
            "conventional_commits": "`fix(users):` prefix is correct and descriptive.",
            "security": "No new security surface introduced.",
            "suggestions": ["Consider adding a unit test for the null-user edge case."],
            "overall_score": 92,
        },
    },
    {
        "id": "pr-139",
        "pr_number": 139,
        "pr_title": "refactor: migrate DB layer to SQLAlchemy 2.0",
        "author": "mchen",
        "repo": "acme-org/core-api",
        "base_branch": "main",
        "head_branch": "refactor/sqlalchemy-2",
        "status": "needs_work",
        "score": 51,
        "timestamp": time.time() - 86400,
        "review": {
            "bugs": [
                "Legacy `Query.get()` calls not migrated — deprecated in SA 2.0.",
                "Missing `Session.execute()` migration in `user_repo.py` line 44.",
            ],
            "code_quality": "14 commits suggest an incremental approach — consider squashing before merge.",
            "conventional_commits": "`refactor(db):` prefix is correct.",
            "security": "Connection string reads from env — good. Verify no leftover hardcoded DSNs.",
            "suggestions": [
                "Replace `session.query(User).get(id)` with `session.get(User, id)`",
                "Add `expire_on_commit=False` to the session factory for async compatibility.",
            ],
            "overall_score": 51,
        },
    },
]


@dashboard_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@dashboard_bp.route("/api/stats", methods=["GET"])
def get_stats() -> tuple[Response, int]:
    from rag_ingestion import get_index_stats

    pending = sum(1 for r in _review_store if r["status"] == "reviewing")
    return jsonify({
        "commits":      {"value": 1247, "delta": "+12 this week",  "trend": "up"},
        "open_issues":  {"value": 34,   "delta": "+4 today",       "trend": "up"},
        "contributors": {"value": 28,   "delta": "+2 this month",  "trend": "up"},
        "ai_reviews":   {"value": len(_review_store), "delta": f"{pending} pending", "trend": "flat"},
        "doc_health": {
            "score": 74,
            "label": "Good",
            "breakdown": {
                "README": 95,
                "Inline Comments": 61,
                "Docstrings": 48,
                "CHANGELOG": 100,
            },
        },
        "rag_index": get_index_stats(),
    }), 200


@dashboard_bp.route("/api/reviews", methods=["GET"])
def get_reviews() -> tuple[Response, int]:
    sorted_reviews = sorted(_review_store, key=lambda r: r["timestamp"], reverse=True)
    return jsonify({"reviews": sorted_reviews, "total": len(sorted_reviews)}), 200


@dashboard_bp.route("/api/demo-review", methods=["POST"])
def demo_review() -> tuple[Response, int]:
    """Inject a synthetic review into the store for live demo purposes."""
    candidates = [
        ("feat: add rate limiting middleware",    "dev_alice", "feature/rate-limit", 85),
        ("fix: correct JWT expiry calculation",   "dev_bob",   "fix/jwt-expiry",    91),
        ("chore: update dependencies to latest",  "dev_carol", "chore/deps",        73),
        ("docs: add API reference for /users",    "dev_dave",  "docs/api-ref",      88),
        ("perf: cache expensive DB queries",      "dev_eve",   "perf/query-cache",  67),
    ]

    pr_num = random.randint(143, 299)
    title, author, branch, score = random.choice(candidates)

    review = {
        "id": f"pr-{pr_num}",
        "pr_number": pr_num,
        "pr_title": title,
        "author": author,
        "repo": "acme-org/core-api",
        "base_branch": "main",
        "head_branch": branch,
        "status": "reviewing",
        "score": score,
        "timestamp": time.time(),
        "review": {
            "bugs": ["No critical bugs found." if score > 80 else "One potential edge case on empty input."],
            "code_quality": "Well-structured. Follows project conventions.",
            "conventional_commits": f"`{title.split(':')[0]}:` prefix is correct per Conventional Commits.",
            "security": "No new security surface detected.",
            "suggestions": ["Add integration test coverage for the happy path."],
            "overall_score": score,
        },
    }

    _review_store.insert(0, review)
    logger.info(f"Demo review injected: {review['pr_title']} (PR#{pr_num})")
    return jsonify({"status": "ok", "review": review}), 201
