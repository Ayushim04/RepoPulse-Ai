"""
RepoPulse AI - Dashboard Routes (Phase 3)
==========================================
Serves the main frontend dashboard and supporting JSON API endpoints
consumed by the frontend JavaScript:

  GET  /            → renders index.html
  GET  /api/stats   → KPI card data (commits, issues, contributors, reviews)
  GET  /api/reviews → list of recent AI-reviewed PRs for the live feed
  POST /api/demo-review → trigger a mock PR review for live demo purposes
"""

import logging
import time

from flask import Blueprint, Response, jsonify, render_template

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint("dashboard", __name__)

# ---------------------------------------------------------------------------
# In-memory review store — populated by the webhook pipeline.
# In Phase 4 this moves to a database (SQLite / Postgres).
# Seeded with demo data so the dashboard looks great immediately.
# ---------------------------------------------------------------------------
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
        "timestamp": time.time() - 7200,   # 2 hours ago
        "review": {
            "bugs": ["Potential NullPointerException: `db.delete(user)` called without null check."],
            "code_quality": "Function `delete_user` lacks a docstring and type hints. snake_case naming is correct.",
            "conventional_commits": "Suggested prefix: `feat(auth):` — follows Conventional Commits spec.",
            "security": "No SQL injection risk (ORM). Add authorization check to verify caller permissions.",
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
        "timestamp": time.time() - 18000,  # 5 hours ago
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
        "timestamp": time.time() - 86400,  # 1 day ago
        "review": {
            "bugs": [
                "Legacy `Query.get()` calls not migrated — deprecated in SA 2.0.",
                "Missing `Session.execute()` migration in `user_repo.py` line 44.",
            ],
            "code_quality": "14 commits suggest incremental approach — consider squashing before merge.",
            "conventional_commits": "`refactor(db):` prefix is correct.",
            "security": "Connection string now reads from env — good. Verify no leftover hardcoded DSNs.",
            "suggestions": [
                "Replace `session.query(User).get(id)` → `session.get(User, id)`",
                "Add `expire_on_commit=False` to session factory for async compat.",
            ],
            "overall_score": 51,
        },
    },
]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@dashboard_bp.route("/", methods=["GET"])
def index():
    """Render the main RepoPulse AI dashboard."""
    logger.debug("Dashboard index requested.")
    return render_template("index.html")


@dashboard_bp.route("/api/stats", methods=["GET"])
def get_stats() -> tuple[Response, int]:
    """
    Return KPI card data for the dashboard stats row.

    Response shape:
        {
          "commits":      { "value": 1247, "delta": "+12 this week",  "trend": "up"   },
          "open_issues":  { "value": 34,   "delta": "+4 today",       "trend": "up"   },
          "contributors": { "value": 28,   "delta": "+2 this month",  "trend": "up"   },
          "ai_reviews":   { "value": 89,   "delta": "3 pending",      "trend": "flat" },
          "doc_health": {
              "score": 74,
              "label": "Good",
              "breakdown": {
                  "README": 95, "Inline Comments": 61,
                  "Docstrings": 48, "CHANGELOG": 100
              }
          }
        }
    """
    from rag_ingestion import get_index_stats
    index_stats = get_index_stats()

    return jsonify({
        "commits":      {"value": 1247, "delta": "+12 this week",  "trend": "up"},
        "open_issues":  {"value": 34,   "delta": "+4 today",       "trend": "up"},
        "contributors": {"value": 28,   "delta": "+2 this month",  "trend": "up"},
        "ai_reviews":   {"value": len(_review_store), "delta": f"{sum(1 for r in _review_store if r['status'] == 'reviewing')} pending", "trend": "flat"},
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
        "rag_index": index_stats,
    }), 200


@dashboard_bp.route("/api/reviews", methods=["GET"])
def get_reviews() -> tuple[Response, int]:
    """
    Return the list of AI-reviewed PRs for the live review feed.
    Sorted newest-first.
    """
    sorted_reviews = sorted(_review_store, key=lambda r: r["timestamp"], reverse=True)
    return jsonify({"reviews": sorted_reviews, "total": len(sorted_reviews)}), 200


@dashboard_bp.route("/api/demo-review", methods=["POST"])
def demo_review() -> tuple[Response, int]:
    """
    Trigger a mock AI review and prepend it to the review store.
    Used during live demos to show the pipeline in action without
    needing a real GitHub webhook event.
    """
    import random

    demo_prs = [
        ("feat: add rate limiting middleware", "dev_alice", "feature/rate-limit", 85),
        ("fix: correct JWT expiry calculation", "dev_bob",   "fix/jwt-expiry",    91),
        ("chore: update dependencies to latest", "dev_carol","chore/deps",        73),
        ("docs: add API reference for /users", "dev_dave",  "docs/api-ref",      88),
        ("perf: cache expensive DB queries",   "dev_eve",   "perf/query-cache",  67),
    ]

    pr_num = random.randint(143, 299)
    title, author, branch, score = random.choice(demo_prs)

    new_review = {
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
            "conventional_commits": f"Commit prefix `{title.split(':')[0]}:` is correct per Conventional Commits.",
            "security": "No new security surface detected.",
            "suggestions": ["Add integration test coverage for the happy path."],
            "overall_score": score,
        },
    }

    _review_store.insert(0, new_review)
    logger.info(f"Demo review triggered: {new_review['pr_title']} (PR#{pr_num})")

    return jsonify({"status": "ok", "review": new_review}), 201
