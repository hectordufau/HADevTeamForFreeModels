# harness/iteration/__init__.py — Feedback and Iteration Engine
"""
Controlled correction loops with failure classification.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


class IterationError(Exception):
    """Raised on iteration limit or escalation."""


FAILURE_CLASSES = [
    "MODEL_FAILURE", "AGENT_FAILURE", "TOOL_FAILURE",
    "BUILD_FAILURE", "TEST_FAILURE", "SECURITY_FAILURE",
    "SCOPE_VIOLATION", "ARCHITECTURE_FAILURE",
    "REQUIREMENT_FAILURE", "HARNESS_FAILURE",
]


@dataclass
class FailureInfo:
    check: str = ""
    reason: str = ""
    failure_class: str = "TEST_FAILURE"


@dataclass
class Feedback:
    task_id: str
    failures: List[FailureInfo] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    target_agent: str = ""


class IterationEngine:
    """Manages iteration limits and feedback loops."""

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations

    def build_feedback(
        self,
        task: Any,
        evaluation_result: Any,
        verification_result: Any,
    ) -> Feedback:
        """Build a Feedback artifact from evaluation/verification results."""
        feedback = Feedback(task_id=task.metadata.id)

        # Collect failures from failed checks
        for check_name, check in verification_result.checks.items():
            if check.status == "failed":
                feedback.failures.append(
                    FailureInfo(
                        check=check_name,
                        reason=check.error or f"{check_name} check failed ({check.failed}/{check.passed + check.failed})",
                        failure_class="TEST_FAILURE",
                    )
                )

        # Add acceptance failures
        for acc in evaluation_result.acceptance:
            if acc.status == "failed":
                feedback.failures.append(
                    FailureInfo(
                        check="acceptance_criteria",
                        reason=f"Acceptance criteria failed: {acc.criterion}",
                        failure_class="REQUIREMENT_FAILURE",
                    )
                )

        # Generate recommendations
        for f in feedback.failures[:3]:  # limit recommendations
            feedback.recommendations.append(f"Fix: {f.reason}")

        # Determine target agent
        has_scope_issue = any(f.failure_class == "SCOPE_VIOLATION" for f in feedback.failures)
        has_arch_issue = any(f.failure_class == "ARCHITECTURE_FAILURE" for f in feedback.failures)

        if has_scope_issue:
            feedback.target_agent = "manager"
        elif has_arch_issue:
            feedback.target_agent = "architect"
        else:
            feedback.target_agent = "coder"

        return feedback

    def should_iterate(self, current_iteration: int, feedback: Feedback) -> bool:
        """Determine if iteration should continue."""
        if current_iteration >= self.max_iterations:
            return False

        # Security failures block iteration
        if any(f.failure_class == "SECURITY_FAILURE" for f in feedback.failures):
            return False

        # Scope violations block iteration
        if any(f.failure_class == "SCOPE_VIOLATION" for f in feedback.failures):
            return False

        return bool(feedback.failures)
