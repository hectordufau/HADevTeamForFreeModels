# harness/learning/retrieval.py — Phase B: Memory Retrieval 2.0
"""
Multi-factor retrieval with historical usefulness tracking,
confidence calibration, and retrieval explainability.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import math
import uuid

from .experience import StructuredExperience, Evidence, ExperienceQualityScorer


class RetrievalError(Exception):
    """Raised on retrieval errors."""


@dataclass
class RetrievalContext:
    """Context for a retrieval query."""
    query: str
    task_id: str = ""
    capabilities: List[str] = field(default_factory=list)
    task_context: str = ""
    limit: int = 10
    current_model: str = ""
    current_agent: str = ""


@dataclass
class RetrievalResult:
    """Result of a multi-factor retrieval with explainability."""
    experience: StructuredExperience
    overall_score: float
    factor_scores: Dict[str, float]  # similarity, usefulness, applicability, confidence, evidence_quality, recency
    explanations: List[str]  # why this was selected
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "task_id": self.experience.task_id,
            "lesson": self.experience.lesson,
            "overall_score": round(self.overall_score, 4),
            "factor_scores": {k: round(v, 4) for k, v in self.factor_scores.items()},
            "explanations": self.explanations,
            "category": self.experience.category,
            "confidence": self.experience.confidence,
        }


@dataclass
class HistoricalUsefulnessRecord:
    """Tracks whether a retrieved experience improved a subsequent outcome."""
    experience_id: str
    retrieval_context_id: str
    retrieval_timestamp: str
    subsequent_outcome_score: float
    baseline_score: float
    improvement: float  # positive means useful
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "experience_id": self.experience_id,
            "retrieval_context_id": self.retrieval_context_id,
            "retrieval_timestamp": self.retrieval_timestamp,
            "subsequent_outcome_score": self.subsequent_outcome_score,
            "baseline_score": self.baseline_score,
            "improvement": round(self.improvement, 4),
            "timestamp": self.timestamp,
        }


class HistoricalUsefulnessTracker:
    """
    Tracks whether retrieved experiences actually improved subsequent outcomes.

    This is the key metric for evaluating retrieval quality — did using
    a past experience lead to better results than the baseline?
    """

    def __init__(self, storage_dir: str = ""):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "retrieval_tracker"
        )
        os.makedirs(self.storage_dir, exist_ok=True)

    def record_retrieval(self, experience_id: str, context_id: str) -> str:
        """Record a retrieval event and return the record ID."""
        record = HistoricalUsefulnessRecord(
            experience_id=experience_id,
            retrieval_context_id=context_id,
            retrieval_timestamp=datetime.utcnow().isoformat(),
            subsequent_outcome_score=0.0,
            baseline_score=0.0,
            improvement=0.0,
        )
        record_id = str(uuid.uuid4())
        path = os.path.join(self.storage_dir, f"{record_id}.json")
        with open(path, "w") as f:
            json.dump(record.to_dict(), f, indent=2)
        return record_id

    def record_outcome(self, record_id: str, subsequent_score: float,
                       baseline_score: float):
        """Record the outcome after a retrieval."""
        path = os.path.join(self.storage_dir, f"{record_id}.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        data["subsequent_outcome_score"] = subsequent_score
        data["baseline_score"] = baseline_score
        data["improvement"] = subsequent_score - baseline_score
        data["timestamp"] = datetime.utcnow().isoformat()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def get_usefulness(self, experience_id: str) -> Optional[float]:
        """Get average improvement from retrievals of a given experience.
        
        Returns None if no data, or the average improvement value.
        Positive means useful, negative means detrimental.
        """
        improvements = []
        for fname in os.listdir(self.storage_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(self.storage_dir, fname)) as f:
                data = json.load(f)
            if data.get("experience_id") == experience_id:
                improvements.append(data.get("improvement", 0.0))

        if not improvements:
            return None
        return sum(improvements) / len(improvements)


class ConfidenceCalibrator:
    """
    Calibrates confidence estimates using historical outcome data.

    Confidence buckets: LOW (< 0.4), MEDIUM (0.4-0.7), HIGH (> 0.7)
    Objective component derived from historical usefulness.
    """

    @staticmethod
    def calibrate(initial_confidence: float,
                  usefulness: Optional[float] = None,
                  access_count: int = 0,
                  success_rate: Optional[float] = None) -> Dict[str, Any]:
        """Calibrate confidence using objective historical data.

        Args:
            initial_confidence: Starting confidence from experience quality
            usefulness: Historical usefulness score (positive=useful, negative=detrimental)
            access_count: Number of times this experience was retrieved
            success_rate: Historical success rate when following this experience

        Returns:
            Dict with calibrated_confidence and bucket
        """
        # Start with initial confidence
        calibrated = initial_confidence

        # Adjust based on historical usefulness
        if usefulness is not None and access_count > 0:
            # Usefulness ranges from -1 to ~1; shift confidence by 0-0.2
            usefulness_bonus = max(-0.2, min(0.2, usefulness * 0.5))
            calibrated += usefulness_bonus

        # Adjust based on success rate (if available)
        if success_rate is not None and access_count > 0:
            rate_adjustment = (success_rate - 0.5) * 0.2
            calibrated += rate_adjustment

        # Adjust based on sample size (more data = more confident)
        if access_count > 0:
            sample_confidence = min(0.1, access_count * 0.02)
            calibrated += sample_confidence

        # Clamp to [0.05, 0.98]
        calibrated = max(0.05, min(0.98, calibrated))

        # Determine bucket
        if calibrated < 0.4:
            bucket = "LOW"
        elif calibrated < 0.7:
            bucket = "MEDIUM"
        else:
            bucket = "HIGH"

        return {
            "calibrated_confidence": round(calibrated, 4),
            "bucket": bucket,
            "initial_confidence": initial_confidence,
            "usefulness_bonus": usefulness * 0.5 if usefulness is not None else 0.0,
            "rate_adjustment": (success_rate - 0.5) * 0.2 if success_rate is not None else 0.0,
            "sample_bonus": min(0.1, access_count * 0.02) if access_count > 0 else 0.0,
        }


class RetrievalExplainer:
    """Generates human-readable explanations for why an experience was selected."""

    @staticmethod
    def explain(result: RetrievalResult) -> List[str]:
        """Generate explanations for a retrieval result."""
        explanations = []
        scores = result.factor_scores

        if scores.get("similarity", 0) > 0.7:
            explanations.append(
                f"High task similarity ({scores['similarity']:.2f})"
            )
        elif scores.get("similarity", 0) > 0.4:
            explanations.append(
                f"Moderate task similarity ({scores['similarity']:.2f})"
            )

        if scores.get("usefulness", 0) > 0.3:
            explanations.append(
                f"Historically produced positive improvement ({scores['usefulness']:.2f})"
            )

        if scores.get("confidence", 0) > 0.7:
            explanations.append(
                f"High confidence lesson ({scores['confidence']:.2f})"
            )

        if scores.get("evidence_quality", 0) > 0.7:
            explanations.append(
                f"Strong supporting evidence ({scores['evidence_quality']:.2f})"
            )

        exp = result.experience
        if exp.lesson:
            explanations.append(f"Lesson: {exp.lesson[:100]}")

        if not explanations:
            explanations.append("Matched by default fallback")

        return explanations


class MultiFactorRetriever:
    """
    Multi-factor retrieval scoring with explainability.

    Factors:
    - similarity: Semantic/task overlap between query and experience
    - historical_usefulness: Did this experience improve outcomes before?
    - applicability: Capability and context match
    - confidence: Calibrated confidence in the lesson
    - evidence_quality: Quality of supporting evidence
    - recency: How recent is the experience
    """

    def __init__(self, experience_pipeline: Optional[Any] = None,
                 usefulness_tracker: Optional[HistoricalUsefulnessTracker] = None,
                 confidence_calibrator: Optional[ConfidenceCalibrator] = None,
                 v3_store: Any = None):
        self.pipeline = experience_pipeline
        self.tracker = usefulness_tracker or HistoricalUsefulnessTracker()
        self.calibrator = confidence_calibrator or ConfidenceCalibrator()
        self.v3_store = v3_store  # backward compatibility with V3 ExperienceStore

    def retrieve(self, context: RetrievalContext) -> List[RetrievalResult]:
        """Retrieve experiences matching the context, scored by multiple factors.

        Args:
            context: The retrieval context

        Returns:
            Ranked list of RetrievalResult with explanations
        """
        candidates = self._get_candidates(context)
        results = []

        for candidate in candidates:
            scores = self._compute_scores(candidate, context)
            overall = self._aggregate_scores(scores)
            explanations = RetrievalExplainer.explain(
                RetrievalResult(
                    experience=candidate,
                    overall_score=overall,
                    factor_scores=scores,
                    explanations=[],
                )
            )
            results.append(RetrievalResult(
                experience=candidate,
                overall_score=overall,
                factor_scores=scores,
                explanations=explanations,
            ))

        results.sort(key=lambda r: r.overall_score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        return results[:context.limit]

    def _get_candidates(self, context: RetrievalContext) -> List[StructuredExperience]:
        """Get candidate experiences from available stores."""
        candidates = []

        # Try V3.1 pipeline first
        if self.pipeline:
            for tid in self.pipeline.list_experiences():
                exp = self.pipeline.retrieve(tid)
                if exp:
                    candidates.append(exp)

        # Fall back to V3 store
        if self.v3_store and not candidates:
            for result in self.v3_store.search(context.query, min_relevance=0.0):
                from .experience import ExperienceExtractorV31
                exp = ExperienceExtractorV31.from_v3_experience(
                    self.v3_store.retrieve(result.entry_id)
                )
                if exp:
                    candidates.append(exp)

        return candidates

    def _compute_scores(self, experience: StructuredExperience,
                        context: RetrievalContext) -> Dict[str, float]:
        """Compute individual factor scores."""
        # Similarity: keyword/task overlap
        query_lower = context.query.lower()
        query_words = set(query_lower.split())
        context_lower = experience.task_context.lower()
        matched = sum(1 for w in query_words if w in context_lower)
        similarity = matched / max(len(query_words), 1)

        # Applicability: capability match
        if context.capabilities:
            cap_matched = sum(
                1 for c in experience.capabilities_used
                if c.lower() in query_lower
            )
            applicability = cap_matched / max(len(experience.capabilities_used), 1)
        else:
            applicability = 0.5  # neutral

        # Historical usefulness
        usefulness = self.tracker.get_usefulness(experience.task_id)
        norm_usefulness = max(0.0, min(1.0, (usefulness + 1.0) / 2.0)) if usefulness is not None else 0.5

        # Confidence (calibrated)
        calibration = self.calibrator.calibrate(
            initial_confidence=experience.confidence,
            usefulness=usefulness,
            access_count=len(experience.evidence),
        )
        confidence = calibration["calibrated_confidence"]

        # Evidence quality
        if experience.evidence:
            ev_quality = sum(
                Evidence.quality(e) for e in experience.evidence
            ) / len(experience.evidence)
        else:
            ev_quality = 0.0

        # Recency (0.0 = oldest, 1.0 = newest)
        try:
            exp_time = datetime.fromisoformat(experience.timestamp)
            now = datetime.utcnow()
            age_hours = (now - exp_time).total_seconds() / 3600
            recency = max(0.0, min(1.0, 1.0 - age_hours / 720))  # 30-day decay
        except (ValueError, TypeError):
            recency = 0.5

        return {
            "similarity": round(similarity, 4),
            "applicability": round(applicability, 4),
            "usefulness": round(norm_usefulness, 4),
            "confidence": round(confidence, 4),
            "evidence_quality": round(ev_quality, 4),
            "recency": round(recency, 4),
        }

    def _aggregate_scores(self, scores: Dict[str, float]) -> float:
        """Weighted aggregation of factor scores."""
        weights = {
            "similarity": 0.30,
            "applicability": 0.20,
            "usefulness": 0.15,
            "confidence": 0.15,
            "evidence_quality": 0.10,
            "recency": 0.10,
        }
        total = sum(scores.get(factor, 0) * weight
                    for factor, weight in weights.items())
        return round(total, 4)
