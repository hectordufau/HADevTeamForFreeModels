# harness/planner/failure_intelligence.py — Failure Intelligence V3 (Phase J)
"""
Structured failure taxonomy, root cause analysis, failure-to-learning pipeline,
and replanning optimization.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


class FailureIntelligenceError(Exception):
    """Raised on failure intelligence errors."""


@dataclass
class StructuredFailure:
    """A structured failure record with taxonomy classification."""
    node_id: str
    capability: str
    category: str  # implementation, test, architecture, security, integration, configuration, environment
    sub_category: str  # more specific classification
    severity: str  # critical, major, minor, warning
    error_message: str
    evidence: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "capability": self.capability,
            "category": self.category,
            "sub_category": self.sub_category,
            "severity": self.severity,
            "error_message": self.error_message,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }


# Extended failure taxonomy with sub-categories
FAILURE_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "implementation": {
        "sub_categories": [
            "logic_error", "api_misuse", "null_pointer", "type_error",
            "resource_leak", "concurrency", "off_by_one", "edge_case",
        ],
        "severity": "major",
    },
    "test": {
        "sub_categories": [
            "assertion_failure", "flaky_test", "missing_coverage",
            "false_positive", "test_environment",
        ],
        "severity": "major",
    },
    "architecture": {
        "sub_categories": [
            "circular_dependency", "tight_coupling", "god_class",
            "layering_violation", "pattern_misuse",
        ],
        "severity": "critical",
    },
    "security": {
        "sub_categories": [
            "injection", "xss", "csrf", "auth_bypass",
            "data_exposure", "privilege_escalation",
        ],
        "severity": "critical",
    },
    "integration": {
        "sub_categories": [
            "contract_mismatch", "version_incompatibility",
            "protocol_error", "timeout", "connection_refused",
        ],
        "severity": "critical",
    },
    "configuration": {
        "sub_categories": [
            "missing_env", "wrong_setting", "invalid_config",
            "missing_dependency_config",
        ],
        "severity": "minor",
    },
    "environment": {
        "sub_categories": [
            "dependency_missing", "version_conflict",
            "disk_full", "out_of_memory", "network_error",
        ],
        "severity": "major",
    },
}


class RootCauseAnalyzer:
    """Analyzes failures to determine root cause using structured taxonomy."""

    def __init__(self):
        self._analysis_history: List[StructuredFailure] = []

    def analyze(self, failure: Any, task_context: Any = None) -> StructuredFailure:
        """Analyze a failure and determine root cause."""
        node_id = self._get_node_id(failure)
        capability = self._get_capability(failure)
        error = self._get_error(failure)
        evidence_text = self._get_evidence(failure)

        category, sub_category = self._classify_failure(error, evidence_text)
        severity = self._get_severity(category, sub_category)

        result = StructuredFailure(
            node_id=node_id,
            capability=capability,
            category=category,
            sub_category=sub_category,
            severity=severity,
            error_message=error,
            evidence=evidence_text[:500],
        )

        self._analysis_history.append(result)
        return result

    def _classify_failure(self, error: str, evidence: str) -> Tuple[str, str]:
        """Classify failure into category and sub-category."""
        combined = (error + " " + evidence).lower()

        # Check sub-categories first
        for category, info in FAILURE_TAXONOMY.items():
            for sub in info["sub_categories"]:
                if sub.replace("_", " ") in combined:
                    return category, sub

        # Fallback to category-level matching
        error_lower = error.lower()
        if "assert" in error_lower or "test" in error_lower:
            return "test", "assertion_failure"
        if "vulnerability" in error_lower or "security" in error_lower:
            return "security", "auth_bypass"
        if "config" in error_lower or "env" in error_lower:
            return "configuration", "missing_env"
        if "dependency" in error_lower or "version" in error_lower:
            return "environment", "dependency_missing"
        if "timeout" in error_lower or "connection" in error_lower:
            return "integration", "timeout"
        if "typeerror" in error_lower or "attribute" in error_lower:
            return "implementation", "type_error"

        return "implementation", "logic_error"

    def _get_severity(self, category: str, sub_category: str) -> str:
        """Get severity for a failure category."""
        info = FAILURE_TAXONOMY.get(category, {})
        severity = info.get("severity", "major")
        return str(severity) if severity else "major"

    def _get_node_id(self, failure: Any) -> str:
        if hasattr(failure, 'node_id'):
            return failure.node_id
        if isinstance(failure, dict):
            return failure.get('node_id', 'unknown')
        return 'unknown'

    def _get_capability(self, failure: Any) -> str:
        if hasattr(failure, 'capability'):
            return failure.capability
        if isinstance(failure, dict):
            return failure.get('capability', '')
        return ''

    def _get_error(self, failure: Any) -> str:
        if hasattr(failure, 'message') and failure.message:
            return failure.message
        if hasattr(failure, 'error_message'):
            return failure.error_message
        if isinstance(failure, dict):
            return failure.get('message', failure.get('error_message', ''))
        return str(failure)

    def _get_evidence(self, failure: Any) -> str:
        if hasattr(failure, 'evidence_summary'):
            return failure.evidence_summary or ""
        if hasattr(failure, 'evidence'):
            return str(failure.evidence)
        if isinstance(failure, dict):
            return str(failure.get('evidence_summary', failure.get('evidence', '')))
        return ''

    def get_patterns(self) -> Dict[str, int]:
        """Get recurring failure patterns from analysis history."""
        patterns = {}
        for f in self._analysis_history:
            key = f"{f.category}/{f.sub_category}"
            patterns[key] = patterns.get(key, 0) + 1
        return dict(sorted(patterns.items(), key=lambda x: -x[1]))


class FailureToLearningPipeline:
    """Converts failures into learning experiences for future avoidance."""

    def __init__(self, experience_store: Any):
        self.experience_store = experience_store

    def process(self, failure: StructuredFailure) -> Dict[str, Any]:
        """Convert a failure into a learning entry."""
        lesson = self._generate_lesson(failure)

        # Store as experience
        if self.experience_store:
            from harness.memory.v3 import ExperienceRecord
            exp = ExperienceRecord(
                task_id=f"failure-{failure.node_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                task_objective=f"Failure: {failure.error_message[:100]}",
                capabilities_used=[failure.capability],
                result_status="FAILED",
                score=0.0,
                duration_seconds=0.0,
                lessons=[lesson],
            )
            self.experience_store.store(exp)

        return {
            "lesson": lesson,
            "capability": failure.capability,
            "category": failure.category,
            "severity": failure.severity,
        }

    def _generate_lesson(self, failure: StructuredFailure) -> str:
        """Generate a human-readable lesson from the failure."""
        return (
            f"AVOID: {failure.category}.{failure.sub_category} in '{failure.capability}'. "
            f"Error: {failure.error_message[:200]}. "
            f"Recommendation: Review {failure.capability} implementation for {failure.sub_category} issues."
        )


class ReplanningOptimizer:
    """Optimizes replanning decisions based on failure patterns."""

    def __init__(self, failure_analyzer: Any, max_replans: int = 3):
        self.failure_analyzer = failure_analyzer
        self.max_replans = max_replans
        self._attempt_counts: Dict[str, int] = {}

    def should_replan(self, task_id: str, failure: Any) -> bool:
        """Determine if replanning is worthwhile based on failure analysis."""
        attempts = self._attempt_counts.get(task_id, 0)

        if attempts >= self.max_replans:
            return False

        # Check if the failure category is recoverable
        category = self._get_category(failure)
        if category in ("security", "architecture"):
            # Security and architecture failures need human intervention
            return False

        # Implementation and test failures are good candidates for replanning
        if category in ("implementation", "test", "configuration", "environment"):
            return True

        # For unknown/integration failures, allow one replan attempt
        if attempts == 0:
            return True

        return False

    def get_optimal_replan_strategy(self, failure: Any) -> str:
        """Get the optimal replan strategy based on failure type."""
        category = self._get_category(failure)

        strategies = {
            "implementation": "retry_with_different_agent",
            "test": "add_test_specific_node",
            "configuration": "add_environment_setup_node",
            "environment": "add_dependency_check_node",
            "integration": "add_integration_test_node",
            "security": "escalate_to_human",
            "architecture": "escalate_to_human",
        }

        return strategies.get(category, "retry_same")

    def record_attempt(self, task_id: str):
        """Record a replanning attempt."""
        self._attempt_counts[task_id] = self._attempt_counts.get(task_id, 0) + 1

    def get_attempt_count(self, task_id: str) -> int:
        """Get the number of replanning attempts for a task."""
        return self._attempt_counts.get(task_id, 0)

    def _get_category(self, failure: Any) -> str:
        if hasattr(failure, 'category'):
            return failure.category
        if isinstance(failure, dict):
            return failure.get('category', 'unknown')
        return 'unknown'
