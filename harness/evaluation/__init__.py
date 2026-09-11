# harness/evaluation/__init__.py — Evaluation & Acceptance Engine
"""
Evaluates task results against acceptance criteria and quality dimensions.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


class EvaluationError(Exception):
    """Raised on evaluation failures."""


@dataclass
class AcceptanceResult:
    criterion: str
    status: str  # passed, failed, skipped
    evidence: str = ""


@dataclass
class EvaluationResult:
    score: float = 0.0
    dimensions: Dict[str, float] = field(default_factory=dict)
    acceptance: List[AcceptanceResult] = field(default_factory=list)
    decision: str = "pending"  # pending, pass, fail

    def is_passed(self) -> bool:
        return self.decision == "pass"


class EvaluationEngine:
    """Evaluates task quality and acceptance criteria."""

    def __init__(self, policy: Optional[dict] = None):
        self.policy = policy or {
            "weights": {
                "correctness": 0.40,
                "architecture": 0.20,
                "security": 0.20,
                "maintainability": 0.10,
                "efficiency": 0.10,
            },
            "minimum_score": 0.85,
        }

    def evaluate_acceptance(self, task: Any, verification_result: Any) -> List[AcceptanceResult]:
        """Evaluate acceptance criteria against verification results."""
        results = []
        criteria = task.spec.acceptance_criteria

        if not criteria:
            return results

        # In MVP: acceptance criteria must be explicitly checked.
        # In full implementation, this would correlate with test results.
        for criterion in criteria:
            results.append(
                AcceptanceResult(
                    criterion=criterion,
                    status="passed" if verification_result.all_passed() else "failed",
                    evidence=f"Verification {'passed' if verification_result.all_passed() else 'failed'}",
                )
            )
        return results

    def evaluate(self, task: Any, verification_result: Any, evidence: Any) -> EvaluationResult:
        """Produce a full evaluation with score and decision."""
        result = EvaluationResult()

        # Evaluate acceptance criteria
        result.acceptance = self.evaluate_acceptance(task, verification_result)

        # Calculate score from verification results (simplified for MVP)
        # In full implementation: each dimension gets a real metric
        dim_scores = {}
        for dim, weight in self.policy["weights"].items():
            base_score = 1.0 if verification_result.all_passed() else 0.5
            dim_scores[dim] = base_score

        result.dimensions = dim_scores

        # Weighted score
        total = 0.0
        for dim, score in dim_scores.items():
            weight = self.policy["weights"].get(dim, 0.0)
            total += score * weight

        result.score = round(total, 2)

        # Decision
        min_score = self.policy.get("minimum_score", 0.85)
        all_accepted = all(a.status == "passed" for a in result.acceptance)

        if all_accepted and result.score >= min_score:
            result.decision = "pass"
        else:
            result.decision = "fail"

        return result
