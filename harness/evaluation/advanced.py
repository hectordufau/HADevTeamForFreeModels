"""
Advanced Evaluation Engine for V2.1 with hard minimums and security gates.
"""

from typing import Optional
from harness.evaluation import EvaluationResult, AcceptanceResult, EvaluationEngine


class AdvancedEvaluationEngine:
    """Evaluation engine with hard minimums and security enforcement."""

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
            "hard_minimums": {
                "correctness": 0.90,
                "security": 0.90,
            },
        }

    def evaluate(self, task, verification_result, evidence) -> EvaluationResult:
        """Produce evaluation with hard minimums and security enforcement."""
        base_engine = EvaluationEngine(self.policy)
        result = base_engine.evaluate(task, verification_result, evidence)

        # Apply hard minimums
        hard = self.policy.get("hard_minimums", {})
        min_correctness = hard.get("correctness", 0.0)
        min_security = hard.get("security", 0.0)

        actual_correctness = result.dimensions.get("correctness", 0.0)
        actual_security = result.dimensions.get("security", 0.0)

        if actual_correctness < min_correctness:
            result.decision = "fail"
            result.acceptance.append(AcceptanceResult(
                criterion=f"Hard minimum: correctness >= {min_correctness}",
                status="failed",
                evidence=f"Got {actual_correctness}, required {min_correctness}",
            ))

        if actual_security < min_security:
            result.decision = "fail"
            result.acceptance.append(AcceptanceResult(
                criterion=f"Hard minimum: security >= {min_security}",
                status="failed",
                evidence=f"Got {actual_security}, required {min_security}",
            ))

        return result
