# harness/learning/experience.py — Phase A: Experience Extraction 2.0
"""
Structured experience representation, extraction pipeline, and quality scoring.

Extends V3.0's ExperienceRecord with richer structure: task context, strategy,
outcome, failures, lessons with metadata, confidence, and evidence links.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import math


class ExperienceError(Exception):
    """Raised on experience extraction errors."""


@dataclass
class ExecutionTrace:
    """A step-by-step trace of an execution for experience extraction."""
    task_id: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    final_status: str = ""
    final_score: float = 0.0
    duration_seconds: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "steps": self.steps,
            "final_status": self.final_status,
            "final_score": self.final_score,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
            "timestamp": self.timestamp,
        }


@dataclass
class Evidence:
    """A piece of evidence supporting a lesson or outcome."""
    description: str
    metric: str  # score, duration, success_rate, error_count
    value: float
    source: str  # execution, evaluation, review
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "metric": self.metric,
            "value": self.value,
            "source": self.source,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def quality(evidence) -> float:
        """Score evidence quality 0.0-1.0 based on source and specificity.
        
        Accepts Evidence object or dict.
        """
        if isinstance(evidence, dict):
            source = evidence.get("source", "")
        else:
            source = evidence.source
        if source == "evaluation":
            return 0.9
        elif source == "execution":
            return 0.7
        elif source == "review":
            return 0.5
        return 0.3


@dataclass
class StructuredExperience:
    """
    Rich experience representation for V3.1 learning.

    Fields:
        task_id: Unique identifier for the source task
        task_context: Description of the task environment and requirements
        capabilities_used: Capabilities involved
        strategy: What strategy was employed (model, agent, workflow choices)
        outcome: What happened (status, score, duration)
        failures: Structured failure information
        lesson: Key lesson learned (human-readable)
        confidence: Confidence in this lesson (0.0-1.0)
        evidence: Supporting evidence for the lesson
        timestamp: When this experience was recorded
    """
    task_id: str
    task_context: str
    capabilities_used: List[str]
    strategy: Dict[str, Any]  # {model_id, agent_id, workflow_name, parameters}
    outcome: Dict[str, Any]   # {status, score, duration, success_rate}
    failures: List[Dict[str, Any]] = field(default_factory=list)
    lesson: str = ""
    confidence: float = 0.0
    evidence: List[Evidence] = field(default_factory=list)
    category: str = ""  # generic categorization (success_pattern, failure_lesson, performance_tip)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_context": self.task_context,
            "capabilities_used": self.capabilities_used,
            "strategy": self.strategy,
            "outcome": self.outcome,
            "failures": self.failures,
            "lesson": self.lesson,
            "confidence": round(self.confidence, 4),
            "evidence": [e.to_dict() for e in self.evidence],
            "category": self.category,
            "timestamp": self.timestamp,
        }


class ExperienceQualityScorer:
    """Scores the quality of a structured experience.

    Quality = evidence_quality × outcome_quality × extraction_confidence

    Each component is 0.0-1.0, product is 0.0-1.0.
    """

    @staticmethod
    def score(experience: StructuredExperience) -> float:
        """Compute overall quality score for a structured experience."""
        # Evidence quality: average of all evidence pieces
        if experience.evidence:
            ev_quality = sum(
                Evidence.quality(e) for e in experience.evidence
            ) / len(experience.evidence)
        else:
            ev_quality = 0.0

        # Outcome quality
        outcome = experience.outcome
        status = outcome.get("status", "")
        score_val = outcome.get("score", 0.0)

        if status == "COMPLETED":
            outcome_quality = min(1.0, (score_val + 1.0) / 2.0)  # normalize -1..1 to 0..1
        elif status == "PARTIAL":
            outcome_quality = 0.5
        elif status == "FAILED":
            outcome_quality = max(0.0, min(0.4, (1.0 - abs(score_val))))
        else:
            outcome_quality = 0.1

        # Extraction confidence: how much evidence backs the lesson
        has_lesson = 1.0 if experience.lesson else 0.0
        has_evidence = min(1.0, len(experience.evidence) / 3.0)

        extraction_confidence = 0.3 + 0.4 * has_lesson + 0.3 * has_evidence

        return round(ev_quality * outcome_quality * extraction_confidence, 4)

    @staticmethod
    def score_components(experience: StructuredExperience) -> Dict[str, float]:
        """Return individual quality components for transparency."""
        if experience.evidence:
            ev_quality = sum(
                Evidence.quality(e) for e in experience.evidence
            ) / len(experience.evidence)
        else:
            ev_quality = 0.0

        outcome = experience.outcome
        status = outcome.get("status", "")
        score_val = outcome.get("score", 0.0)

        if status == "COMPLETED":
            outcome_quality = min(1.0, (score_val + 1.0) / 2.0)
        elif status == "PARTIAL":
            outcome_quality = 0.5
        elif status == "FAILED":
            outcome_quality = max(0.0, min(0.4, (1.0 - abs(score_val))))
        else:
            outcome_quality = 0.1

        has_lesson = 1.0 if experience.lesson else 0.0
        has_evidence = min(1.0, len(experience.evidence) / 3.0)
        extraction_confidence = 0.3 + 0.4 * has_lesson + 0.3 * has_evidence

        return {
            "evidence_quality": round(ev_quality, 4),
            "outcome_quality": round(outcome_quality, 4),
            "extraction_confidence": round(extraction_confidence, 4),
            "overall": round(ev_quality * outcome_quality * extraction_confidence, 4),
        }


class ExperienceExtractorV31:
    """
    V3.1 Experience Extraction Pipeline.

    Pipeline: Task → Execution Trace → Evidence → Structured Experience

    Works with V3.0's ExperienceStore for backward compatibility but produces
    richer StructuredExperience objects.
    """

    @staticmethod
    def extract(trace: ExecutionTrace, task_objective: str,
                capabilities: List[str] = None, strategy: Dict[str, Any] = None,
                lesson: str = "") -> StructuredExperience:
        """Extract a structured experience from an execution trace.

        Args:
            trace: The execution trace to analyze
            task_objective: The original task objective
            capabilities: List of capabilities used
            strategy: Strategy employed (model, agent, workflow)
            lesson: Optional pre-extracted lesson

        Returns:
            A complete StructuredExperience
        """
        if capabilities is None:
            capabilities = []

        if strategy is None:
            strategy = {}

        # Build evidence from trace
        evidence_list = []

        # Score evidence
        score = trace.final_score
        evidence_list.append(Evidence(
            description=f"Final score: {score}",
            metric="score",
            value=score,
            source="evaluation",
        ))

        # Duration evidence
        duration = trace.duration_seconds
        evidence_list.append(Evidence(
            description=f"Duration: {duration}s",
            metric="duration",
            value=duration,
            source="execution",
        ))

        # Error evidence
        for error in trace.errors[:3]:  # cap at 3 errors
            evidence_list.append(Evidence(
                description=error.get("message", "Unknown error"),
                metric="error_count",
                value=1.0,
                source="execution",
            ))

        # Determine category
        status = trace.final_status
        if status == "COMPLETED" and score > 0.6:
            category = "success_pattern"
        elif status == "FAILED":
            category = "failure_lesson"
        else:
            category = "performance_tip"

        # Auto-extract lesson if not provided
        if not lesson and trace.errors:
            lesson = f"Avoid: {trace.errors[0].get('message', 'unknown error')}"
        elif not lesson and status == "COMPLETED":
            lesson = f"Strategy worked well for {', '.join(capabilities) if capabilities else 'task'}"

        # Compute confidence from evidence
        scorer = ExperienceQualityScorer()
        components = scorer.score_components(
            StructuredExperience(
                task_id=trace.task_id,
                task_context=task_objective,
                capabilities_used=capabilities,
                strategy=strategy,
                outcome={"status": status, "score": score, "duration": duration},
                failures=trace.errors,
                lesson=lesson,
                evidence=evidence_list,
                category=category,
            )
        )

        return StructuredExperience(
            task_id=trace.task_id,
            task_context=task_objective,
            capabilities_used=capabilities,
            strategy=strategy,
            outcome={"status": status, "score": score, "duration": duration},
            failures=trace.errors,
            lesson=lesson,
            confidence=components["overall"],
            evidence=evidence_list,
            category=category,
        )

    @staticmethod
    def from_v3_experience(exp_record: Any) -> StructuredExperience:
        """Convert a V3.0 ExperienceRecord to V3.1 StructuredExperience.

        Preserves backward compatibility.
        """
        # Extract data from V3 ExperienceRecord
        if hasattr(exp_record, 'to_dict'):
            data = exp_record.to_dict()
        else:
            data = exp_record  # assume dict-like

        # Build evidence from available data
        evidence_list = []
        evidence_list.append(Evidence(
            description=f"Result: {data.get('result_status', 'UNKNOWN')}",
            metric="score",
            value=data.get('score', 0.0),
            source="evaluation",
        ))
        if data.get('duration_seconds', 0) > 0:
            evidence_list.append(Evidence(
                description=f"Duration: {data['duration_seconds']}s",
                metric="duration",
                value=data['duration_seconds'],
                source="execution",
            ))

        # Reconstruct strategy
        strategy = {}
        if data.get('capabilities_used'):
            strategy["capabilities"] = data['capabilities_used']

        # Reconstruct failures from lessons
        failures = []
        lesson_text = ""
        for l in data.get('lessons', []):
            if l.startswith("Failed capabilities:"):
                failures.append({"message": l, "node_id": "", "capability": ""})
            else:
                lesson_text += l + "; "

        status = data.get('result_status', 'UNKNOWN')
        score = data.get('score', 0.0)

        return StructuredExperience(
            task_id=data.get('task_id', ''),
            task_context=data.get('task_objective', ''),
            capabilities_used=data.get('capabilities_used', []),
            strategy=strategy,
            outcome={"status": status, "score": score, "duration": data.get('duration_seconds', 0)},
            failures=failures,
            lesson=lesson_text.strip("; "),
            confidence=0.5,  # default for converted entries
            evidence=evidence_list,
            category="success_pattern" if status == "COMPLETED" else "failure_lesson",
        )


class ExperiencePipeline:
    """
    End-to-end experience extraction pipeline.

    Wraps extraction + quality scoring + storage.
    """

    def __init__(self, store: Any = None, storage_dir: str = ""):
        self.store = store
        if not storage_dir:
            storage_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "..", "artifacts", "experiences_v31"
            )
        self._storage_dir = storage_dir
        os.makedirs(self._storage_dir, exist_ok=True)
        self._quality_dir = os.path.join(self._storage_dir, "quality")
        os.makedirs(self._quality_dir, exist_ok=True)

    def process_trace(self, trace: ExecutionTrace, task_objective: str,
                      capabilities: Optional[List[str]] = None,
                      strategy: Optional[Dict[str, Any]] = None,
                      lesson: str = "") -> StructuredExperience:
        """Process an execution trace into a stored structured experience."""
        exp = ExperienceExtractorV31.extract(trace, task_objective,
                                              capabilities, strategy, lesson)
        self._store(exp)
        return exp

    def _store(self, exp: StructuredExperience):
        """Store a structured experience."""
        path = os.path.join(self._storage_dir, f"{exp.task_id}.json")
        with open(path, "w") as f:
            json.dump(exp.to_dict(), f, indent=2)

        # Store quality score separately
        quality = ExperienceQualityScorer.score_components(exp)
        quality_path = os.path.join(self._quality_dir, f"{exp.task_id}.json")
        with open(quality_path, "w") as f:
            json.dump(quality, f, indent=2)

    def retrieve(self, task_id: str) -> Optional[StructuredExperience]:
        """Retrieve a stored experience by task ID."""
        path = os.path.join(self._storage_dir, f"{task_id}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        # Reconstruct Evidence objects from dicts
        if "evidence" in data:
            data["evidence"] = [Evidence(**e) if isinstance(e, dict) else e for e in data["evidence"]]
        return StructuredExperience(**data)

    def get_quality(self, task_id: str) -> Optional[Dict[str, float]]:
        """Get quality score for an experience."""
        path = os.path.join(self._quality_dir, f"{task_id}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def list_experiences(self) -> List[str]:
        """List all stored experience task IDs."""
        exps = []
        for fname in os.listdir(self._storage_dir):
            if fname.endswith(".json"):
                exps.append(fname[:-5])
        return exps
