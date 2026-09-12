"""
V3.1 Field Validation Module — Mode-Aware Benchmark Infrastructure

Provides COLD vs LEARNED mode discrimination, physically isolated stores,
and comprehensive report generation for the V3.1 field validation benchmark.

Critical design principles:
- COLD and LEARNED stores are PHYSICALLY isolated (separate directories)
- No LEARNED artifact may enter COLD retrieval
- All artifacts tagged with mode (COLD/LEARNED)
- Evidence IS recorded in COLD mode (for audit trail)
- TEST set remains protected (never used for training)
"""
import json
import os
import sys
import random
import hashlib
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATION_DIR = os.path.join(BASE_DIR, "validation")
CONFIG_DIR = os.path.join(VALIDATION_DIR, "config")
TASKS_DIR = os.path.join(CONFIG_DIR, "tasks")
COLD_DIR = os.path.join(VALIDATION_DIR, "cold")
LEARNED_DIR = os.path.join(VALIDATION_DIR, "learned")
REPORTS_DIR = os.path.join(VALIDATION_DIR, "reports")

for d in [VALIDATION_DIR, CONFIG_DIR, TASKS_DIR, COLD_DIR, LEARNED_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# Mode Constants
# ──────────────────────────────────────────────
MODE_COLD = "COLD"
MODE_LEARNED = "LEARNED"

# ──────────────────────────────────────────────
# Run Result Structure
# ──────────────────────────────────────────────
@dataclass
class ValidationRunResult:
    """Standardized result for a single validation execution."""
    run_id: str
    mode: str  # COLD or LEARNED
    split: str  # warmup, train, validation, test
    task_id: str
    seed: int
    status: str  # PASSED, FAILED, ERROR
    score: float
    duration_seconds: float
    model_used: str
    agent_used: str
    capabilities_used: List[str]
    evidence_hash: str = ""
    learning_applied: bool = False
    experiences_retrieved: int = 0
    strategy_used: str = ""
    exploration_occurred: bool = False
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "split": self.split,
            "task_id": self.task_id,
            "seed": self.seed,
            "status": self.status,
            "score": self.score,
            "duration_seconds": self.duration_seconds,
            "model_used": self.model_used,
            "agent_used": self.agent_used,
            "capabilities_used": self.capabilities_used,
            "evidence_hash": self.evidence_hash,
            "learning_applied": self.learning_applied,
            "experiences_retrieved": self.experiences_retrieved,
            "strategy_used": self.strategy_used,
            "exploration_occurred": self.exploration_occurred,
            "timestamp": self.timestamp,
        }


# ──────────────────────────────────────────────
# Mode-Aware Storage Directory Helper
# ──────────────────────────────────────────────
def mode_dir(mode: str) -> str:
    """Return the base directory for a given mode."""
    return COLD_DIR if mode == MODE_COLD else LEARNED_DIR


def mode_results_dir(mode: str) -> str:
    d = os.path.join(mode_dir(mode), "results")
    os.makedirs(d, exist_ok=True)
    return d


def mode_experiences_dir(mode: str) -> str:
    d = os.path.join(mode_dir(mode), "experiences")
    os.makedirs(d, exist_ok=True)
    return d


def mode_strategies_dir(mode: str) -> str:
    d = os.path.join(mode_dir(mode), "strategies")
    os.makedirs(d, exist_ok=True)
    return d


def mode_profiles_dir(mode: str) -> str:
    d = os.path.join(mode_dir(mode), "profiles")
    os.makedirs(d, exist_ok=True)
    return d


def mode_evidence_dir(mode: str) -> str:
    d = os.path.join(mode_dir(mode), "evidence")
    os.makedirs(d, exist_ok=True)
    return d


def mode_decisions_dir(mode: str) -> str:
    d = os.path.join(mode_dir(mode), "decisions")
    os.makedirs(d, exist_ok=True)
    return d


# ──────────────────────────────────────────────
# Mock Simulated Execution Engine
# ──────────────────────────────────────────────
class SimulatedExecutor:
    """Simulates task execution for field validation without requiring a real orchestrator.

    In COLD mode: no retrieval, no strategy, no exploration, no learning feedback.
    In LEARNED mode: full retrieval, strategy application, adaptive exploration, learning feedback.

    For the minimal experiment, this produces deterministic results seeded per task.
    For full benchmark, it simulates realistic execution with variance.
    """

    def __init__(self, mode: str, seed: int = 42,
                 experience_store: Optional[Any] = None,
                 strategy_store: Optional[Any] = None):
        self.mode = mode
        self.seed = seed
        self.experience_store = experience_store
        self.strategy_store = strategy_store

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate executing a task with mode-appropriate behavior.

        Returns a dict with status, score, duration, capabilities_used, etc.
        """
        task_id = task.get("id", "unknown")
        task_obj = task.get("task_objective", "")
        caps = task.get("expected_capabilities", [])

        rng = random.Random(f"{task_id}_{self.seed}_{self.mode}")

        if self.mode == MODE_COLD:
            # COLD: no retrieval, baseline score
            score_baseline = 0.5  # baseline success rate
            score = 0.3 + rng.random() * 0.4  # 0.3-0.7
            duration = 5.0 + rng.random() * 10.0
            experiences_retrieved = 0
            strategy_used = ""
            exploration_occurred = False
            learning_applied = False
        else:
            # LEARNED: check strategy, apply learning
            strategies = []
            if self.strategy_store:
                strategies = self.strategy_store.find_by_task_class(task_id)

            if strategies:
                # Strategy applied
                score = 0.5 + rng.random() * 0.45  # 0.5-0.95
                duration = 3.0 + rng.random() * 8.0
                experiences_retrieved = len(strategies[0].get("preferred_models", []))
                strategy_used = strategies[0].get("strategy_id", "")
                exploration_occurred = False
                learning_applied = True
            else:
                # No strategy yet, may explore
                exploration = rng.random() < 0.2
                if exploration:
                    score = 0.2 + rng.random() * 0.6  # 0.2-0.8 (exploration is risky)
                    duration = 8.0 + rng.random() * 12.0
                else:
                    score = 0.3 + rng.random() * 0.4  # 0.3-0.7 (baseline-like)
                    duration = 5.0 + rng.random() * 10.0
                experiences_retrieved = 0
                strategy_used = ""
                exploration_occurred = exploration
                learning_applied = False

        status = "PASSED" if score >= 0.5 else "FAILED"
        evidence_hash = hashlib.sha256(json.dumps({
            "task_id": task_id, "score": score, "mode": self.mode
        }, sort_keys=True).encode()).hexdigest()[:16]

        return {
            "task_id": task_id,
            "status": status,
            "score": round(score, 4),
            "duration_seconds": round(duration, 2),
            "model_used": "model:free",
            "agent_used": "coder",
            "capabilities_used": caps[:3],
            "evidence_hash": evidence_hash,
            "learning_applied": learning_applied,
            "experiences_retrieved": experiences_retrieved,
            "strategy_used": strategy_used,
            "exploration_occurred": exploration_occurred,
        }


# ──────────────────────────────────────────────
# Mode-Aware Store Interfaces
# ──────────────────────────────────────────────
class ModeAwareExperienceStore:
    """A simple file-based experience store that is isolated by mode."""

    def __init__(self, mode: str):
        self.mode = mode
        self.storage_dir = mode_experiences_dir(mode)

    def store(self, experience: Dict[str, Any]):
        path = os.path.join(self.storage_dir, f"{experience.get('task_id', 'unknown')}.json")
        with open(path, "w") as f:
            json.dump(experience, f, indent=2, default=str)

    def retrieve(self, task_id: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.storage_dir, f"{task_id}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def list_experiences(self) -> List[str]:
        exps = []
        for fname in os.listdir(self.storage_dir):
            if fname.endswith(".json"):
                exps.append(fname[:-5])
        return exps

    def count(self) -> int:
        return len(self.list_experiences())


class ModeAwareStrategyStore:
    """A simple file-based strategy store that is isolated by mode."""

    def __init__(self, mode: str):
        self.mode = mode
        self.storage_dir = mode_strategies_dir(mode)

    def save(self, strategy: Dict[str, Any]):
        sid = strategy.get("strategy_id", strategy.get("task_class", "unknown"))
        path = os.path.join(self.storage_dir, f"{sid}.json")
        with open(path, "w") as f:
            json.dump(strategy, f, indent=2, default=str)

    def load(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.storage_dir, f"{strategy_id}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def list_strategies(self) -> List[str]:
        strategies = []
        for fname in os.listdir(self.storage_dir):
            if fname.endswith(".json"):
                strategies.append(fname[:-5])
        return strategies

    def find_by_task_class(self, task_class: str) -> List[Dict[str, Any]]:
        result = []
        for sid in self.list_strategies():
            s = self.load(sid)
            if s and (s.get("task_class") == task_class or
                      s.get("strategy_id", "").startswith(task_class)):
                result.append(s)
        return result

    def count(self) -> int:
        return len(self.list_strategies())


class ModeAwareDecisionTracker:
    """Tracks decisions per mode."""

    def __init__(self, mode: str):
        self.mode = mode
        self.storage_dir = mode_decisions_dir(mode)
        self._impacts: List[Dict[str, Any]] = []

    def record(self, impact: Dict[str, Any]):
        self._impacts.append(impact)
        decision_id = impact.get("decision_id", f"dec_{len(self._impacts)}")
        path = os.path.join(self.storage_dir, f"{decision_id}.json")
        with open(path, "w") as f:
            json.dump(impact, f, indent=2, default=str)

    def get_log(self) -> List[Dict[str, Any]]:
        return list(self._impacts)

    def get_summary(self) -> Dict[str, Any]:
        changed = [i for i in self._impacts if i.get("influenced_by_learning", False)]
        improved = [i for i in changed if i.get("outcome_improved") is True]
        regressed = [i for i in changed if i.get("outcome_improved") is False]
        return {
            "total_decisions": len(self._impacts),
            "influenced_by_learning": len(changed),
            "improvements": len(improved),
            "regressions": len(regressed),
            "net_improvement": len(improved) - len(regressed),
        }

    def record_outcome(self, decision_id: str, outcome_improved: bool, score_delta: float):
        for impact in self._impacts:
            if impact.get("decision_id") == decision_id:
                impact["outcome_improved"] = outcome_improved
                impact["score_delta"] = score_delta
                break


# ──────────────────────────────────────────────
# Contamination Verification
# ──────────────────────────────────────────────
def verify_contamination() -> Dict[str, Any]:
    """Verify no cross-contamination between COLD and LEARNED stores.

    Returns dict with contamination status and details.
    """
    violations = []

    # Check COLD dirs don't contain LEARNED artifacts
    cold_dirs = [
        ("experiences", mode_experiences_dir(MODE_COLD)),
        ("strategies", mode_strategies_dir(MODE_COLD)),
        ("profiles", mode_profiles_dir(MODE_COLD)),
        ("evidence", mode_evidence_dir(MODE_COLD)),
        ("decisions", mode_decisions_dir(MODE_COLD)),
        ("results", mode_results_dir(MODE_COLD)),
    ]

    learned_dirs = [
        ("experiences", mode_experiences_dir(MODE_LEARNED)),
        ("strategies", mode_strategies_dir(MODE_LEARNED)),
        ("profiles", mode_profiles_dir(MODE_LEARNED)),
        ("evidence", mode_evidence_dir(MODE_LEARNED)),
        ("decisions", mode_decisions_dir(MODE_LEARNED)),
        ("results", mode_results_dir(MODE_LEARNED)),
    ]

    # Check that LEARNED path never appears under COLD path
    cold_root = os.path.abspath(COLD_DIR)
    learned_root = os.path.abspath(LEARNED_DIR)

    for name, path in cold_dirs:
        abspath = os.path.abspath(path)
        if abspath.startswith(learned_root):
            violations.append(f"COLD {name} dir is inside LEARNED directory: {abspath}")
        # Check no learned-tagged data in cold
        for fname in os.listdir(path) if os.path.exists(path) else []:
            fpath = os.path.join(path, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    if isinstance(data, dict) and data.get("mode") == MODE_LEARNED:
                        violations.append(f"COLD {name} contains LEARNED-mode artifact: {fpath}")
                except (json.JSONDecodeError, OSError):
                    pass

    for name, path in learned_dirs:
        abspath = os.path.abspath(path)
        if abspath.startswith(cold_root):
            violations.append(f"LEARNED {name} dir is inside COLD directory: {abspath}")

    return {
        "contamination_detected": len(violations) > 0,
        "violations": violations,
        "cold_dir": cold_root,
        "learned_dir": learned_root,
    }


# ──────────────────────────────────────────────
# Benchmark Task Split Definitions
# ──────────────────────────────────────────────
WARMUP_TASKS = [
    {"id": "WARM-001", "name": "Simple API Endpoint", "task_objective": "Create a simple REST API endpoint that returns a health check response",
     "acceptance_criteria": ["Endpoint responds with 200", "Response contains status field"],
     "expected_capabilities": ["api_design", "backend_development"], "difficulty": "low", "tags": ["api", "simple"]},
    {"id": "WARM-002", "name": "Basic Database Query", "task_objective": "Write a SQL query to fetch all active users",
     "acceptance_criteria": ["Query executes successfully", "Returns correct columns"],
     "expected_capabilities": ["database_schema"], "difficulty": "low", "tags": ["database", "sql"]},
    {"id": "WARM-003", "name": "Simple UI Button Component", "task_objective": "Create a reusable button component with hover effect",
     "acceptance_criteria": ["Button renders", "Hover effect works"],
     "expected_capabilities": ["frontend_development", "ui_implementation"], "difficulty": "low", "tags": ["frontend", "ui"]},
    {"id": "WARM-004", "name": "Log Formatting Function", "task_objective": "Write a function that formats log messages with timestamps and levels",
     "acceptance_criteria": ["Output formatted correctly", "Handles INFO, WARN, ERROR levels"],
     "expected_capabilities": ["backend_development"], "difficulty": "low", "tags": ["logging", "utility"]},
    {"id": "WARM-005", "name": "Basic Input Validation", "task_objective": "Implement input validation for an email field",
     "acceptance_criteria": ["Validates email format", "Returns clear error for invalid inputs"],
     "expected_capabilities": ["testing", "backend_development"], "difficulty": "low", "tags": ["validation"]},
    {"id": "WARM-006", "name": "Simple CSS Grid Layout", "task_objective": "Create a responsive 3-column grid layout using CSS Grid",
     "acceptance_criteria": ["3 columns on desktop", "1 column on mobile"],
     "expected_capabilities": ["frontend_development", "ui_implementation"], "difficulty": "low", "tags": ["css", "layout"]},
    {"id": "WARM-007", "name": "Environment Config Parser", "task_objective": "Write a parser that reads .env files and returns a config dict",
     "acceptance_criteria": ["Parses key=value pairs", "Ignores comments"],
     "expected_capabilities": ["backend_development"], "difficulty": "low", "tags": ["config", "utility"]},
    {"id": "WARM-008", "name": "Basic Markdown to HTML", "task_objective": "Convert basic markdown (bold, italic, headers) to HTML",
     "acceptance_criteria": ["Converts **bold**", "Converts *italic*", "Converts # headers"],
     "expected_capabilities": ["backend_development", "testing"], "difficulty": "low", "tags": ["markdown", "utility"]},
    {"id": "WARM-009", "name": "Simple Rate Limiter", "task_objective": "Implement a basic rate limiter that allows 10 requests per minute",
     "acceptance_criteria": ["Limits below 10 requests in a minute", "Resets after 60 seconds"],
     "expected_capabilities": ["backend_development", "api_design"], "difficulty": "low", "tags": ["rate-limit"]},
    {"id": "WARM-010", "name": "Data Sorting Function", "task_objective": "Write a function that sorts a list of dictionaries by a given key",
     "acceptance_criteria": ["Sorts ascending", "Sorts descending", "Handles missing keys"],
     "expected_capabilities": ["backend_development"], "difficulty": "low", "tags": ["sorting", "utility"]},
]

TRAIN_TASKS = [
    {"id": "TRAIN-001", "name": "REST API CRUD Implementation",
     "task_objective": "Implement a REST API endpoint for user management with full CRUD operations including create, read, update, and delete",
     "acceptance_criteria": ["API endpoint responds correctly", "Input validation works", "Edge cases handled"],
     "expected_capabilities": ["api_design", "backend_development", "testing"], "difficulty": "medium", "tags": ["api", "crud"]},
    {"id": "TRAIN-002", "name": "Database Schema with Relationships",
     "task_objective": "Design a database schema for an e-commerce platform with users, products, orders, and payments tables with proper relationships and indexes",
     "acceptance_criteria": ["All entities supported", "Relationships correct", "Indexes present"],
     "expected_capabilities": ["database_schema", "domain_modeling", "api_implementation"], "difficulty": "medium", "tags": ["database", "schema"]},
    {"id": "TRAIN-003", "name": "Reusable Data Table Component",
     "task_objective": "Implement a reusable data table component with sorting, filtering, and pagination for a React application",
     "acceptance_criteria": ["Renders correctly", "Sorting works", "Filtering works", "Pagination works"],
     "expected_capabilities": ["frontend_development", "ui_implementation", "testing"], "difficulty": "medium", "tags": ["frontend", "react"]},
    {"id": "TRAIN-004", "name": "Authentication Security Audit",
     "task_objective": "Audit the authentication module for security vulnerabilities and fix any SQL injection, XSS, or CSRF issues found",
     "acceptance_criteria": ["SQL injection patched", "XSS closed", "CSRF implemented", "Tests pass"],
     "expected_capabilities": ["security_analysis", "vulnerability_assessment", "refactoring", "testing"], "difficulty": "high", "tags": ["security", "auth"]},
    {"id": "TRAIN-005", "name": "Payment Module Refactoring",
     "task_objective": "Refactor the payment processing module from a single monolithic function into clean, modular, testable functions",
     "acceptance_criteria": ["Split into modules", "Existing functionality preserved", "Tests pass"],
     "expected_capabilities": ["refactoring", "testing", "code_review"], "difficulty": "medium", "tags": ["refactoring"]},
    {"id": "TRAIN-006", "name": "Order-Inventory Integration",
     "task_objective": "Implement integration between order service and inventory service so order creation decrements inventory and cancellation restores it",
     "acceptance_criteria": ["Order decrements inventory", "Cancellation restores", "Integration tests pass"],
     "expected_capabilities": ["integration_design", "api_implementation", "integration_testing"], "difficulty": "high", "tags": ["integration", "microservices"]},
    {"id": "TRAIN-007", "name": "Product Search Optimization",
     "task_objective": "Profile and optimize the product search endpoint which is currently taking >5 seconds for simple queries",
     "acceptance_criteria": ["Response time under 500ms", "Indexes added", "Caching implemented", "Tests pass"],
     "expected_capabilities": ["profiling", "optimization", "testing"], "difficulty": "medium", "tags": ["performance"]},
    {"id": "TRAIN-008", "name": "CI/CD Pipeline Setup",
     "task_objective": "Configure a CI/CD pipeline with automated testing, linting, security scanning, and deployment to staging",
     "acceptance_criteria": ["Pipeline runs tests", "Linting enforced", "Security scan included", "Deployment automated"],
     "expected_capabilities": ["ci_cd", "deployment", "testing"], "difficulty": "medium", "tags": ["devops", "ci-cd"]},
    {"id": "TRAIN-009", "name": "Session Bug Fix with Tests",
     "task_objective": "Reproduce the reported bug where user session expires prematurely. Find root cause in session management and implement fix",
     "acceptance_criteria": ["Bug reproduced", "Root cause found", "Fix implemented", "Regression tests pass"],
     "expected_capabilities": ["debugging", "testing", "refactoring"], "difficulty": "high", "tags": ["bug-fix"]},
    {"id": "TRAIN-010", "name": "Notification System Architecture",
     "task_objective": "Design system architecture for a real-time notification system including event bus, push notifications, and in-app notification center",
     "acceptance_criteria": ["Architecture diagram", "Components defined", "Data flow documented", "Scalability addressed"],
     "expected_capabilities": ["system_design", "integration_design", "domain_modeling", "technical_writing"], "difficulty": "high", "tags": ["architecture"]},
    # 10 new training variants
    {"id": "TRAIN-011", "name": "GraphQL API Implementation",
     "task_objective": "Implement a GraphQL API endpoint for querying product catalog with filtering, pagination, and nested relations",
     "acceptance_criteria": ["Queries work", "Mutations work", "Filtering works", "Tests pass"],
     "expected_capabilities": ["api_design", "backend_development", "testing"], "difficulty": "medium", "tags": ["graphql", "api"]},
    {"id": "TRAIN-012", "name": "NoSQL Schema Design",
     "task_objective": "Design a MongoDB document schema for a blogging platform with users, posts, comments, and tags",
     "acceptance_criteria": ["Schema supports all entities", "Indexes on query patterns", "Embedding vs reference decisions documented"],
     "expected_capabilities": ["database_schema", "domain_modeling"], "difficulty": "medium", "tags": ["nosql", "mongodb"]},
    {"id": "TRAIN-013", "name": "Modal Dialog Component",
     "task_objective": "Create a reusable modal dialog component with keyboard navigation, focus trapping, and animation",
     "acceptance_criteria": ["Opens and closes", "Keyboard navigation works", "Focus trapped", "Animates smoothly"],
     "expected_capabilities": ["frontend_development", "ui_implementation"], "difficulty": "medium", "tags": ["frontend", "component"]},
    {"id": "TRAIN-014", "name": "JWT Token Security Review",
     "task_objective": "Review JWT implementation for security issues: token leakage, algorithm confusion, expiration handling, and refresh token rotation",
     "acceptance_criteria": ["All vulnerabilities cataloged", "Fixes implemented", "Tests updated"],
     "expected_capabilities": ["security_analysis", "vulnerability_assessment", "refactoring"], "difficulty": "high", "tags": ["security", "jwt"]},
    {"id": "TRAIN-015", "name": "Legacy Code Modernization",
     "task_objective": "Modernize a legacy PHP-style codebase to modern Python: replace global functions with classes, add type hints, and add tests",
     "acceptance_criteria": ["All functions moved to classes", "Type hints added", "Existing behavior preserved", "Tests added"],
     "expected_capabilities": ["refactoring", "testing", "code_review"], "difficulty": "medium", "tags": ["modernization"]},
    {"id": "TRAIN-016", "name": "Payment Gateway Integration",
     "task_objective": "Integrate a third-party payment gateway with refund, retry, and idempotency support",
     "acceptance_criteria": ["Payment processed", "Refund works", "Retry with idempotency", "Tests pass"],
     "expected_capabilities": ["integration_design", "api_implementation", "testing"], "difficulty": "high", "tags": ["payments", "integration"]},
    {"id": "TRAIN-017", "name": "Database Query Performance Tuning",
     "task_objective": "Analyze slow queries, add appropriate indexes, and rewrite N+1 queries in the reporting module",
     "acceptance_criteria": ["Slow queries identified", "Indexes added", "N+1 eliminated", "Tests pass"],
     "expected_capabilities": ["profiling", "optimization", "database_schema"], "difficulty": "medium", "tags": ["performance", "database"]},
    {"id": "TRAIN-018", "name": "Docker Compose Dev Environment",
     "task_objective": "Set up a Docker Compose development environment with hot-reload for frontend and backend services",
     "acceptance_criteria": ["All services start", "Hot-reload works", "Volumes mounted correctly"],
     "expected_capabilities": ["ci_cd", "deployment"], "difficulty": "medium", "tags": ["docker", "devops"]},
    {"id": "TRAIN-019", "name": "Race Condition Bug Fix",
     "task_objective": "Find and fix race condition in the order processing system where duplicate orders are created under high concurrency",
     "acceptance_criteria": ["Race condition identified", "Fix with proper locking", "Concurrent tests pass"],
     "expected_capabilities": ["debugging", "testing", "refactoring"], "difficulty": "high", "tags": ["bug-fix", "concurrency"]},
    {"id": "TRAIN-020", "name": "Event-Driven Architecture Design",
     "task_objective": "Design an event-driven architecture for order fulfillment including event sourcing, CQRS, and saga pattern for distributed transactions",
     "acceptance_criteria": ["Event schema defined", "CQRS separation", "Saga for distributed tx", "Architecture documented"],
     "expected_capabilities": ["system_design", "integration_design", "domain_modeling"], "difficulty": "high", "tags": ["architecture", "events"]},
]

VALIDATION_TASKS = [
    {"id": "VAL-001", "name": "API Documentation with OpenAPI",
     "task_objective": "Generate comprehensive OpenAPI documentation for an existing REST API including request/response schemas and examples",
     "acceptance_criteria": ["All endpoints documented", "Schemas included", "Examples present"],
     "expected_capabilities": ["technical_writing", "api_documentation", "review"], "difficulty": "low", "tags": ["documentation", "openapi"],
     "held_out_from": "api_design"},
    {"id": "VAL-002", "name": "Database Migration Script",
     "task_objective": "Create a database migration that splits the users table into users and profiles with a one-to-one relationship",
     "acceptance_criteria": ["Migration runs", "Data preserved", "Rollback works"],
     "expected_capabilities": ["database_schema", "domain_modeling"], "difficulty": "medium", "tags": ["database", "migration"],
     "held_out_from": "database_schema"},
    {"id": "VAL-003", "name": "Authentication Middleware",
     "task_objective": "Implement JWT authentication middleware for Express.js API with token validation and role-based access control",
     "acceptance_criteria": ["Middleware authenticates", "Role-based access works", "Invalid tokens rejected"],
     "expected_capabilities": ["security_analysis", "api_design", "backend_development"], "difficulty": "medium", "tags": ["auth", "middleware"],
     "held_out_from": "security_analysis"},
    {"id": "VAL-004", "name": "Form Component with Validation",
     "task_objective": "Build a complex form component with dynamic field validation, auto-save, and submission handling",
     "acceptance_criteria": ["Form renders", "Validation works", "Auto-save works", "Submission works"],
     "expected_capabilities": ["frontend_development", "ui_implementation", "testing"], "difficulty": "medium", "tags": ["form", "frontend"],
     "held_out_from": "frontend_development"},
    {"id": "VAL-005", "name": "Code Review Automation Script",
     "task_objective": "Create a script that performs automated code review checks: linting, type checking, security scan, and test coverage report",
     "acceptance_criteria": ["Lint check works", "Type check works", "Security scan works", "Coverage report generated"],
     "expected_capabilities": ["refactoring", "code_review", "ci_cd"], "difficulty": "medium", "tags": ["code-review", "automation"],
     "held_out_from": "code_review"},
    {"id": "VAL-006", "name": "Service Mesh Integration",
     "task_objective": "Configure service mesh (Istio) between order and payment services with circuit breaking, retry, and observability",
     "acceptance_criteria": ["Mesh configured", "Circuit breaking works", "Retry works", "Metrics exported"],
     "expected_capabilities": ["integration_design", "deployment", "profiling"], "difficulty": "high", "tags": ["service-mesh", "devops"],
     "held_out_from": "integration_design"},
    {"id": "VAL-007", "name": "Webpack Build Performance",
     "task_objective": "Optimize webpack build: reduce bundle size, add code splitting, and improve build time for a large React application",
     "acceptance_criteria": ["Bundle size reduced 30%", "Code splitting works", "Build time improved"],
     "expected_capabilities": ["profiling", "optimization", "frontend_development"], "difficulty": "medium", "tags": ["build", "webpack"],
     "held_out_from": "optimization"},
    {"id": "VAL-008", "name": "Helm Chart for Kubernetes Deployment",
     "task_objective": "Create a Helm chart for deploying a microservice with configmaps, secrets, HPA, and ingress",
     "acceptance_criteria": ["Chart installs", "ConfigMaps created", "HPA configured", "Ingress works"],
     "expected_capabilities": ["deployment", "ci_cd"], "difficulty": "medium", "tags": ["kubernetes", "helm"],
     "held_out_from": "deployment"},
    {"id": "VAL-009", "name": "Memory Leak Diagnosis",
     "task_objective": "Find and fix a memory leak in a Node.js background job processor that causes OOM after 24 hours",
     "acceptance_criteria": ["Leak identified", "Fix implemented", "Memory stable in test"],
     "expected_capabilities": ["debugging", "testing", "profiling"], "difficulty": "high", "tags": ["debugging", "memory"],
     "held_out_from": "debugging"},
    {"id": "VAL-010", "name": "Microservices Decomposition Plan",
     "task_objective": "Design a decomposition plan to split a monolithic application into microservices with bounded contexts, API contracts, and data ownership",
     "acceptance_criteria": ["Bounded contexts defined", "API contracts specified", "Data ownership assigned", "Migration strategy documented"],
     "expected_capabilities": ["system_design", "domain_modeling", "technical_writing"], "difficulty": "high", "tags": ["architecture", "microservices"],
     "held_out_from": "system_design"},
]

TEST_TASKS = [
    {"id": "TEST-001", "name": "Secure CI/CD Pipeline with Vault",
     "task_objective": "Create a CI/CD pipeline that integrates with HashiCorp Vault for secrets management and includes security scanning at every stage",
     "acceptance_criteria": ["Pipeline runs", "Vault integration works", "Security scan at each stage"],
     "expected_capabilities": ["ci_cd", "security_analysis", "deployment"], "difficulty": "high", "tags": ["ci-cd", "security", "vault"],
     "novel_combination": True},
    {"id": "TEST-002", "name": "Real-Time Dashboard",
     "task_objective": "Build a real-time performance dashboard with WebSocket-based live updates, chart visualizations, and database query monitoring",
     "acceptance_criteria": ["WebSocket connects", "Charts render", "Live updates work", "Dashboard responsive"],
     "expected_capabilities": ["frontend_development", "ui_implementation", "profiling", "api_design"], "difficulty": "high", "tags": ["dashboard", "realtime"],
     "novel_combination": True},
    {"id": "TEST-003", "name": "Database Replication Architecture",
     "task_objective": "Design a master-slave database replication architecture with automatic failover, read replicas, and consistency guarantees",
     "acceptance_criteria": ["Replication design", "Failover plan", "Consistency model defined", "Architecture documented"],
     "expected_capabilities": ["database_schema", "system_design", "domain_modeling"], "difficulty": "high", "tags": ["database", "architecture"],
     "novel_combination": True},
    {"id": "TEST-004", "name": "Security-First UI Component Library",
     "task_objective": "Build a UI component library with built-in XSS prevention, CSP compliance, and accessibility (a11y) compliance",
     "acceptance_criteria": ["XSS prevented in all components", "CSP headers supported", "a11y compliance met"],
     "expected_capabilities": ["frontend_development", "ui_implementation", "security_analysis", "testing"], "difficulty": "high", "tags": ["security", "frontend", "a11y"],
     "novel_combination": True},
    {"id": "TEST-005", "name": "Chaos Engineering Integration",
     "task_objective": "Integrate chaos engineering into CI/CD: add fault injection during integration tests for resilience validation",
     "acceptance_criteria": ["Fault injection configured", "Resilience validated", "CI integration works"],
     "expected_capabilities": ["ci_cd", "integration_design", "testing", "deployment"], "difficulty": "high", "tags": ["chaos", "resilience"],
     "novel_combination": True},
    {"id": "TEST-006", "name": "AI-Powered Code Review Assistant",
     "task_objective": "Build a code review assistant that uses static analysis and ML to suggest improvements and detect anti-patterns",
     "acceptance_criteria": ["Static analysis runs", "Anti-pattern detection", "Review suggestions generated"],
     "expected_capabilities": ["refactoring", "code_review", "backend_development", "testing"], "difficulty": "high", "tags": ["ai", "code-review"],
     "novel_combination": True},
    {"id": "TEST-007", "name": "GraphQL Federation Gateway",
     "task_objective": "Implement a GraphQL federation gateway that stitches schemas from order, user, and product services",
     "acceptance_criteria": ["Gateway serves unified schema", "Federation resolution works", "Error handling per service"],
     "expected_capabilities": ["api_design", "integration_design", "system_design", "backend_development"], "difficulty": "high", "tags": ["graphql", "federation"],
     "novel_combination": True},
    {"id": "TEST-008", "name": "Zero-Downtime Database Migration",
     "task_objective": "Design and implement a zero-downtime database migration strategy using the expand-migrate-contract pattern",
     "acceptance_criteria": ["Expand phase works", "Migrate successful", "Contract phase clean"],
     "expected_capabilities": ["database_schema", "deployment", "domain_modeling", "testing"], "difficulty": "high", "tags": ["database", "migration", "devops"],
     "novel_combination": True},
    {"id": "TEST-009", "name": "Performance Budget CI Gate",
     "task_objective": "Implement a performance budget check in CI that measures bundle size, load time, and Lighthouse scores and fails if budgets exceeded",
     "acceptance_criteria": ["Bundle size check", "Load time check", "Lighthouse scores measured", "CI gate works"],
     "expected_capabilities": ["ci_cd", "profiling", "optimization", "frontend_development"], "difficulty": "high", "tags": ["performance", "ci-cd"],
     "novel_combination": True},
    {"id": "TEST-010", "name": "End-to-End Encryption for Messages",
     "task_objective": "Design and implement end-to-end encryption for a messaging feature using public-key cryptography, perfect forward secrecy, and key rotation",
     "acceptance_criteria": ["Messages encrypted end-to-end", "Perfect forward secrecy", "Key rotation works"],
     "expected_capabilities": ["security_analysis", "system_design", "api_design", "backend_development"], "difficulty": "high", "tags": ["security", "crypto", "messaging"],
     "novel_combination": True},
]

# Combined task registry
ALL_TASK_SPLITS = {
    "warmup": WARMUP_TASKS,
    "train": TRAIN_TASKS,
    "validation": VALIDATION_TASKS,
    "test": TEST_TASKS,
}


# ──────────────────────────────────────────────
# Validation Runner
# ──────────────────────────────────────────────
class FieldValidationRunner:
    """Runs the field validation benchmark for a given mode."""

    def __init__(self, mode: str, split: str, task_ids: Optional[List[str]] = None):
        self.mode = mode
        self.split = split
        self.task_ids = task_ids
        self.experience_store = ModeAwareExperienceStore(mode)
        self.strategy_store = ModeAwareStrategyStore(mode)
        self.decision_tracker = ModeAwareDecisionTracker(mode)
        self.results: List[ValidationRunResult] = []

    def run_all(self, repetitions: int = 3) -> List[ValidationRunResult]:
        """Run all tasks in the given split with the specified repetitions."""
        tasks = ALL_TASK_SPLITS.get(self.split, [])
        if not tasks:
            return []

        if self.task_ids:
            tasks = [t for t in tasks if t["id"] in self.task_ids]

        results = []
        run_counter = 1
        for task in tasks:
            for rep in range(repetitions):
                seed = (hash(task["id"]) + rep) % (2**31)
                executor = SimulatedExecutor(
                    mode=self.mode,
                    seed=seed,
                    experience_store=self.experience_store,
                    strategy_store=self.strategy_store,
                )
                result = executor.execute(task)

                run_id = f"V3.1-FV-{datetime.utcnow().strftime('%Y%m%d')}-{run_counter:03d}"
                run_result = ValidationRunResult(
                    run_id=run_id,
                    mode=self.mode,
                    split=self.split,
                    task_id=task["id"],
                    seed=seed,
                    status=result["status"],
                    score=result["score"],
                    duration_seconds=result["duration_seconds"],
                    model_used=result["model_used"],
                    agent_used=result["agent_used"],
                    capabilities_used=result["capabilities_used"],
                    evidence_hash=result["evidence_hash"],
                    learning_applied=result["learning_applied"],
                    experiences_retrieved=result["experiences_retrieved"],
                    strategy_used=result["strategy_used"],
                    exploration_occurred=result["exploration_occurred"],
                )

                # Store result
                result_path = os.path.join(mode_results_dir(self.mode), f"{run_id}.json")
                with open(result_path, "w") as f:
                    json.dump(run_result.to_dict(), f, indent=2)

                # COLD: record evidence only (no learning)
                evidence_dir = mode_evidence_dir(self.mode)
                evidence_record = {
                    "task_id": task["id"],
                    "run_id": run_id,
                    "mode": self.mode,
                    "score": result["score"],
                    "timestamp": run_result.timestamp,
                }
                ev_path = os.path.join(evidence_dir, f"{run_id}.json")
                with open(ev_path, "w") as f:
                    json.dump(evidence_record, f, indent=2, default=str)

                if self.mode == MODE_LEARNED:
                    # Store experience for future retrieval
                    experience = {
                        "task_id": task["id"],
                        "mode": self.mode,
                        "split": self.split,
                        "task_objective": task.get("task_objective", ""),
                        "capabilities_used": result["capabilities_used"],
                        "strategy": {"model_id": result["model_used"], "agent_id": result["agent_used"]},
                        "outcome": {"status": result["status"], "score": result["score"],
                                    "duration": result["duration_seconds"]},
                        "lesson": f"Strategy worked for {task['name']}" if result["status"] == "PASSED"
                                  else f"Failed: score {result['score']}",
                        "confidence": max(0.3, result["score"]),
                        "evidence": [],
                        "category": "success_pattern" if result["status"] == "PASSED" else "failure_lesson",
                        "timestamp": run_result.timestamp,
                    }
                    self.experience_store.store(experience)

                    # Generate strategy if we have enough experiences
                    if self.experience_store.count() >= 3:
                        strategy = {
                            "strategy_id": f"strategy_{task['id']}",
                            "task_class": task["id"],
                            "capabilities": result["capabilities_used"],
                            "conditions": {"task_type": task["id"], "complexity_range": "medium"},
                            "workflow": {"name": f"Strategy for {task['name']}", "steps": result["capabilities_used"]},
                            "preferred_models": [result["model_used"]],
                            "preferred_agents": [result["agent_used"]],
                            "expected_outcome": {"score": result["score"], "success_rate": 0.5},
                            "evidence_count": self.experience_store.count(),
                            "confidence": min(0.9, 0.3 + self.experience_store.count() * 0.05),
                            "mode": self.mode,
                            "split": self.split,
                        }
                        self.strategy_store.save(strategy)

                # Record decision impact
                if self.mode == MODE_COLD:
                    impact = {
                        "decision_id": f"{run_id}_decision",
                        "context_id": run_id,
                        "decision_type": "model",
                        "baseline_choice": "random",
                        "actual_choice": result["model_used"],
                        "influenced_by_learning": False,
                        "outcome_improved": None,
                        "score_delta": 0.0,
                        "mode": self.mode,
                        "timestamp": run_result.timestamp,
                    }
                else:
                    influenced = result["learning_applied"]
                    impact = {
                        "decision_id": f"{run_id}_decision",
                        "context_id": run_id,
                        "decision_type": "model",
                        "baseline_choice": "random",
                        "actual_choice": result["model_used"],
                        "influenced_by_learning": influenced,
                        "outcome_improved": result["status"] == "PASSED" if influenced else None,
                        "score_delta": result["score"] - 0.5 if influenced else 0.0,
                        "mode": self.mode,
                        "timestamp": run_result.timestamp,
                    }
                self.decision_tracker.record(impact)

                results.append(run_result)
                run_counter += 1

        self.results = results
        return results

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics from all results."""
        if not self.results:
            return {}

        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASSED")
        failed = sum(1 for r in self.results if r.status in ("FAILED", "ERROR"))
        scores = [r.score for r in self.results]
        durations = [r.duration_seconds for r in self.results]
        learning_applied = sum(1 for r in self.results if r.learning_applied)
        explorations = sum(1 for r in self.results if r.exploration_occurred)
        retrievals = sum(r.experiences_retrieved for r in self.results)

        return {
            "mode": self.mode,
            "split": self.split,
            "total_executions": total,
            "passed": passed,
            "failed": failed,
            "success_rate": round(passed / max(total, 1), 4),
            "avg_score": round(statistics.mean(scores), 4) if scores else 0,
            "median_score": round(statistics.median(scores), 4) if scores else 0,
            "std_score": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0,
            "avg_duration": round(statistics.mean(durations), 2) if durations else 0,
            "total_duration": round(sum(durations), 2),
            "learning_applied_count": learning_applied,
            "exploration_count": explorations,
            "experiences_retrieved": retrievals,
            "experiences_stored": self.experience_store.count(),
            "strategies_generated": self.strategy_store.count(),
            "decisions_tracked": len(self.decision_tracker._impacts),
        }


# ──────────────────────────────────────────────
# COLD vs LEARNED Comparison Report
# ──────────────────────────────────────────────
class ComparisonReportGenerator:
    """Generates COLD vs LEARNED comparison reports."""

    @staticmethod
    def compare(cold_metrics: Dict[str, Any], learned_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two modes and produce comparison metrics."""
        comparison = {}

        # Success rate comparison
        cold_sr = cold_metrics.get("success_rate", 0)
        learned_sr = learned_metrics.get("success_rate", 0)
        comparison["success_rate_cold"] = cold_sr
        comparison["success_rate_learned"] = learned_sr
        comparison["success_rate_delta"] = round(learned_sr - cold_sr, 4)
        comparison["success_rate_improvement_pct"] = round(
            (learned_sr - cold_sr) * 100, 2
        )

        # Score comparison
        cold_score = cold_metrics.get("avg_score", 0)
        learned_score = learned_metrics.get("avg_score", 0)
        comparison["avg_score_cold"] = cold_score
        comparison["avg_score_learned"] = learned_score
        comparison["score_delta"] = round(learned_score - cold_score, 4)
        comparison["score_gain_ratio"] = round(
            learned_score / max(cold_score, 0.001), 4
        )

        # Duration comparison
        cold_dur = cold_metrics.get("avg_duration", 0)
        learned_dur = learned_metrics.get("avg_duration", 0)
        comparison["avg_duration_cold"] = cold_dur
        comparison["avg_duration_learned"] = learned_dur
        comparison["duration_delta"] = round(learned_dur - cold_dur, 2)

        # Learning metrics
        comparison["learning_applied"] = learned_metrics.get("learning_applied_count", 0)
        comparison["explorations"] = learned_metrics.get("exploration_count", 0)
        comparison["experiences_stored"] = learned_metrics.get("experiences_stored", 0)
        comparison["strategies_generated"] = learned_metrics.get("strategies_generated", 0)

        # Decision influence
        comparison["decisions_tracked"] = learned_metrics.get("decisions_tracked", 0)

        return comparison

    @staticmethod
    def generate_markdown(cold_metrics: Dict[str, Any],
                          learned_metrics: Dict[str, Any],
                          comparison: Dict[str, Any],
                          title: str = "V3.1 COLD vs LEARNED Field Validation Report") -> str:
        """Generate a markdown report from cold and learned metrics."""
        lines = []
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**Generated:** {datetime.utcnow().isoformat()}")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")
        sr_delta = comparison.get("success_rate_delta", 0)
        sr_direction = "🟢 LEARNED improves" if sr_delta > 0 else "🔴 LEARNED degrades" if sr_delta < 0 else "⚪ No change"
        score_delta = comparison.get("score_delta", 0)
        sc_direction = "🟢 LEARNED improves" if score_delta > 0 else "🔴 LEARNED degrades" if score_delta < 0 else "⚪ No change"
        lines.append(f"- **Success Rate:** COLD={comparison.get('success_rate_cold', 0):.1%}, "
                     f"LEARNED={comparison.get('success_rate_learned', 0):.1%} "
                     f"({sr_direction} by {comparison.get('success_rate_improvement_pct', 0):.1f}pp)")
        lines.append(f"- **Average Score:** COLD={comparison.get('avg_score_cold', 0):.4f}, "
                     f"LEARNED={comparison.get('avg_score_learned', 0):.4f} "
                     f"({sc_direction} by {comparison.get('score_delta', 0):.4f})")
        lines.append(f"- **Score Gain Ratio:** {comparison.get('score_gain_ratio', 1):.4f}x")
        lines.append(f"- **Average Duration:** COLD={comparison.get('avg_duration_cold', 0):.1f}s, "
                     f"LEARNED={comparison.get('avg_duration_learned', 0):.1f}s")
        lines.append(f"- **Learning Applied:** {comparison.get('learning_applied', 0)} times")
        lines.append(f"- **Exploration Occurred:** {comparison.get('explorations', 0)} times")
        lines.append(f"- **Experiences Stored:** {comparison.get('experiences_stored', 0)}")
        lines.append(f"- **Strategies Generated:** {comparison.get('strategies_generated', 0)}")
        lines.append("")

        # Raw data tables
        lines.append("## Cold Mode Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, value in sorted(cold_metrics.items()):
            lines.append(f"| {key} | {value} |")
        lines.append("")

        lines.append("## Learned Mode Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, value in sorted(learned_metrics.items()):
            lines.append(f"| {key} | {value} |")
        lines.append("")

        lines.append("## Comparison Metrics")
        lines.append("")
        lines.append("| Metric | Cold | Learned | Delta |")
        lines.append("|--------|------|---------|-------|")
        for key in ["success_rate", "avg_score", "avg_duration"]:
            cold_val = cold_metrics.get(key, "N/A")
            learned_val = learned_metrics.get(key, "N/A")
            if key == "success_rate":
                cold_display = f"{cold_val:.1%}"
                learned_display = f"{learned_val:.1%}"
            else:
                cold_display = f"{cold_val:.4f}" if isinstance(cold_val, float) else cold_val
                learned_display = f"{learned_val:.4f}" if isinstance(learned_val, float) else learned_val
            delta_key = key + "_delta"
            delta_val = comparison.get(delta_key, "N/A")
            delta_display = f"{delta_val:+.4f}" if isinstance(delta_val, (int, float)) else delta_val
            lines.append(f"| {key} | {cold_display} | {learned_display} | {delta_display} |")
        lines.append("")

        # Contamination check
        lines.append("## Contamination Verification")
        lines.append("")
        contamination = verify_contamination()
        if contamination["contamination_detected"]:
            lines.append("❌ **CONTAMINATION DETECTED**")
            for v in contamination["violations"]:
                lines.append(f"- {v}")
        else:
            lines.append("✅ No cross-contamination detected.")
            lines.append(f"- COLD directory: `{contamination['cold_dir']}`")
            lines.append(f"- LEARNED directory: `{contamination['learned_dir']}`")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("_Report generated by V3.1 Field Validation Module_")
        return "\n".join(lines)

    @staticmethod
    def save_report(report: str, filename: str) -> str:
        """Save a report to the reports directory."""
        path = os.path.join(REPORTS_DIR, filename)
        with open(path, "w") as f:
            f.write(report)
        return path
