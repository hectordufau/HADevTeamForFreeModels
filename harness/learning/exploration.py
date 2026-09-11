# harness/learning/exploration.py — Phase E: Adaptive Exploration
"""
Adaptive exploration policy, purposeful candidate selection,
and exploration result learning for V3.1.

Extends the exploration capabilities from planner/model_intelligence.py
with a standalone exploration module accessible from the learning package.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# Re-export from model_intelligence for backward compatibility
from harness.planner.model_intelligence import ExplorationResultLearner


class AdaptiveExplorationPolicy:
    """
    V3.1: Adaptive exploration rate based on confidence.

    The exploration rate adjusts dynamically between 2-50% based on
    the system's confidence in its current knowledge. High confidence
    means low exploration (exploit), low confidence means high exploration.
    """

    MIN_EXPLORATION_RATE = 0.02  # 2%
    MAX_EXPLORATION_RATE = 0.50  # 50%
    DEFAULT_EXPLORATION_RATE = 0.10  # 10%

    def __init__(self, exploration_learner: Optional[ExplorationResultLearner] = None):
        self.learner = exploration_learner

    def compute_rate(self, capabilities: List[str],
                     avg_confidence: Optional[float] = None) -> float:
        """
        Compute adaptive exploration rate based on confidence.

        Args:
            capabilities: Capabilities being considered.
            avg_confidence: Optional pre-computed average confidence (0.0-1.0).

        Returns:
            Exploration rate between MIN and MAX.
        """
        if avg_confidence is not None:
            confidence = avg_confidence
        elif self.learner:
            # Average exploration quality across capabilities as proxy for confidence
            scores = [
                self.learner.get_exploration_quality(cap)
                for cap in capabilities
            ]
            confidence = sum(scores) / len(scores) if scores else 0.5
        else:
            confidence = 0.5  # neutral

        # Linear mapping: confidence 0.0 -> 50% exploration, 1.0 -> 2% exploration
        rate = self.MAX_EXPLORATION_RATE - confidence * (
            self.MAX_EXPLORATION_RATE - self.MIN_EXPLORATION_RATE
        )
        return max(self.MIN_EXPLORATION_RATE, min(self.MAX_EXPLORATION_RATE, rate))

    def should_explore(self, capabilities: List[str],
                       avg_confidence: Optional[float] = None) -> bool:
        """Determine whether to explore or exploit based on computed rate."""
        import random
        rate = self.compute_rate(capabilities, avg_confidence)
        return random.random() < rate


class PurposefulCandidateSelector:
    """
    V3.1: Selects exploration candidates that maximize information gain
    plus potential quality improvement.

    Purposeful exploration evaluates candidates on:
    1. Information gain potential (how much we'd learn)
    2. Quality improvement potential (could this be better than current best?)
    3. Diversity (exploring different approaches)
    4. Feasibility (is the candidate realistic given current constraints?)
    """

    @staticmethod
    def select(candidates: List[Dict[str, Any]],
               current_best: Optional[str] = None,
               exploration_learner: Optional[ExplorationResultLearner] = None,
               top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Purposefully select the best exploration candidates.

        Args:
            candidates: List of candidate dicts with 'model_id', 'capability', etc.
            current_best: Current best model ID (for exploitation comparison).
            exploration_learner: For incorporating past exploration results.
            top_k: Number of candidates to return.

        Returns:
            Top-k candidates sorted by exploration value.
        """
        scored = []
        for candidate in candidates:
            score = PurposefulCandidateSelector._score_candidate(
                candidate, current_best, exploration_learner
            )
            scored.append((score, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [sc[1] for sc in scored[:top_k]]

    @staticmethod
    def _score_candidate(candidate: Dict[str, Any],
                         current_best: Optional[str],
                         exploration_learner: Optional[ExplorationResultLearner]) -> float:
        """Score a single candidate for exploration value."""
        score = 0.0

        # Information gain potential
        if candidate.get("is_unknown", False):
            score += 0.3
        if candidate.get("novelty", 0) > 0.5:
            score += 0.2

        # Quality improvement potential
        if current_best and candidate.get("model_id") != current_best:
            score += 0.2

        # Diversity bonus
        if candidate.get("approach") and candidate.get("approach") not in (
            candidate.get("existing_approaches") or []
        ):
            score += 0.15

        # Past exploration results
        if exploration_learner and candidate.get("model_id"):
            quality = exploration_learner.get_exploration_quality(
                candidate.get("capability", "")
            )
            score += quality * 0.15

        return score
