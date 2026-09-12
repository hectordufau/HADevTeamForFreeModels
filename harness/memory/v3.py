# harness/memory/v3.py — Unified Memory Model V3 (Phase D)
"""
Extends MemoryManager with experience extraction, relevance scoring,
and confidence tracking.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import math

from ..learning.context import ExperimentContext
from ..learning.isolation import check_test_protection, namespace_path


class MemoryV3Error(Exception):
    """Raised on V3 memory errors."""


@dataclass
class ExperienceRecord:
    """An extracted experience from a completed execution."""
    task_id: str
    task_objective: str
    capabilities_used: List[str]
    result_status: str
    score: float
    duration_seconds: float
    lessons: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_objective": self.task_objective,
            "capabilities_used": self.capabilities_used,
            "result_status": self.result_status,
            "score": self.score,
            "duration_seconds": self.duration_seconds,
            "lessons": self.lessons,
            "timestamp": self.timestamp,
        }


@dataclass
class MemoryScore:
    """Relevance score for a memory/experience entry."""
    entry_id: str
    content: str
    relevance: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    source: str  # experience, lesson, adr, task
    category: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "relevance": self.relevance,
            "confidence": self.confidence,
            "source": self.source,
            "category": self.category,
            "timestamp": self.timestamp,
        }


@dataclass
class ConfidenceTracker:
    """Tracks confidence levels for memory entries over time."""
    entry_id: str
    confidence: float = 0.5  # start neutral
    access_count: int = 0
    last_accessed: str = ""
    success_count: int = 0
    failure_count: int = 0
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def record_access(self, outcome: str):
        """Record an access with outcome (success/failure)."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow().isoformat()

        if outcome == "success":
            self.success_count += 1
        elif outcome == "failure":
            self.failure_count += 1

        # Update confidence: Bayesian average
        total = self.success_count + self.failure_count
        if total > 0:
            self.confidence = (self.success_count + 1) / (total + 2)

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "confidence": round(self.confidence, 4),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created": self.created,
        }


class ExperienceExtractor:
    """Extracts experiences from execution results for memory storage."""

    @staticmethod
    def extract(report: Dict[str, Any], task: Any) -> ExperienceRecord:
        """Extract an ExperienceRecord from an execution report and task."""
        # Safely get values with defaults
        task_id = getattr(task, 'metadata', None)
        task_id = getattr(task_id, 'id', str(id(task))) if task_id else str(id(task))

        task_obj = getattr(task, 'spec', None)
        task_objective = getattr(task_obj, 'objective', str(task)) if task_obj else str(task)

        caps_used = []
        if "execution" in report:
            node_results = report["execution"].get("node_results", {})
            for ndata in node_results.values():
                if ndata.get("capability"):
                    caps_used.append(ndata["capability"])

        score = report.get("evaluation", {}).get("score", 0.0)
        status = report.get("status", "UNKNOWN")

        # Extract lessons
        lessons = []
        if "execution" in report:
            failed = report["execution"].get("failed_nodes", [])
            if failed:
                lessons.append(f"Failed capabilities: {failed}")

        return ExperienceRecord(
            task_id=task_id,
            task_objective=task_objective,
            capabilities_used=list(set(caps_used)),
            result_status=status,
            score=score if isinstance(score, (int, float)) else 0.0,
            duration_seconds=report.get("execution", {}).get("duration", 0.0),
            lessons=lessons,
        )


class RelevanceScorer:
    """Scores memory/experience entries for relevance to a query."""

    @staticmethod
    def score(entry: Any, query: str) -> float:
        """Score relevance of a memory entry to a query string.
        
        Uses simple keyword overlap and metadata matching.
        Returns 0.0 to 1.0.
        """
        if not query or not entry:
            return 0.0

        query_lower = query.lower()
        query_words = set(query_lower.split())

        score = 0.0
        total_weight = 0.0

        # Content overlap
        content = ""
        if hasattr(entry, 'content'):
            content = entry.content
        elif hasattr(entry, 'task_objective'):
            content = entry.task_objective
        elif hasattr(entry, 'to_dict'):
            content = json.dumps(entry.to_dict())
        content_lower = content.lower()

        # Keyword overlap
        matched = sum(1 for w in query_words if w in content_lower)
        if query_words:
            score += matched / len(query_words) * 0.6
            total_weight += 0.6

        # Capability overlap
        if hasattr(entry, 'capabilities_used'):
            cap_matched = sum(1 for c in entry.capabilities_used if c.lower() in query_lower)
            if entry.capabilities_used:
                score += cap_matched / len(entry.capabilities_used) * 0.3
                total_weight += 0.3

        # Result status bonus
        if hasattr(entry, 'result_status'):
            if entry.result_status == "COMPLETED" and "completed" in query_lower:
                score += 0.1
                total_weight += 0.1
            elif entry.result_status == "FAILED" and "failed" in query_lower:
                score += 0.1
                total_weight += 0.1

        return score / max(total_weight, 1.0)


class ExperienceStore:
    """Persistent store for experiences with relevance and confidence tracking."""

    def __init__(self, storage_dir: str = "",
                 experiment_context: Optional[ExperimentContext] = None):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "experiences"
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        self._confidence_dir = os.path.join(self.storage_dir, "confidence")
        os.makedirs(self._confidence_dir, exist_ok=True)
        self.experiment_context = experiment_context

    def store(self, experience: ExperienceRecord,
              experiment_context: Optional[ExperimentContext] = None) -> str:
        """Store an experience and initialize its confidence tracker in namespaced directory."""
        if experiment_context is None:
            experiment_context = self.experiment_context
        check_test_protection(experiment_context, "store")

        store_dir = namespace_path(
            "experiences",
            experiment_context=experiment_context
        ) if experiment_context else self.storage_dir

        exp_path = os.path.join(store_dir, f"{experience.task_id}.json")
        with open(exp_path, "w") as f:
            json.dump(experience.to_dict(), f, indent=2)

        # Initialize confidence tracker
        confidence_dir = os.path.join(store_dir, "confidence")
        os.makedirs(confidence_dir, exist_ok=True)
        tracker = ConfidenceTracker(entry_id=experience.task_id)
        self._save_confidence(tracker, confidence_dir)

        return experience.task_id

    def retrieve(self, task_id: str,
                 experiment_context: Optional[ExperimentContext] = None) -> Optional[ExperienceRecord]:
        """Retrieve an experience by task ID, filtered by context."""
        if experiment_context is None:
            experiment_context = self.experiment_context

        store_dir = namespace_path(
            "experiences",
            experiment_context=experiment_context
        ) if experiment_context else self.storage_dir

        path = os.path.join(store_dir, f"{task_id}.json")
        if not os.path.exists(path):
            # Fallback: try V3.0 flat directory if no context
            if experiment_context is None:
                path = os.path.join(self.storage_dir, f"{task_id}.json")
                if not os.path.exists(path):
                    return None
            else:
                return None
        with open(path) as f:
            data = json.load(f)
        return ExperienceRecord(**data)

    def search(self, query: str, min_relevance: float = 0.0,
               limit: int = 10,
               experiment_context: Optional[ExperimentContext] = None) -> List[MemoryScore]:
        """Search experiences by relevance to query, filtered by context."""
        if experiment_context is None:
            experiment_context = self.experiment_context

        store_dir = namespace_path(
            "experiences",
            experiment_context=experiment_context
        ) if experiment_context else self.storage_dir

        if not os.path.exists(store_dir):
            return []

        results = []
        for fname in os.listdir(store_dir):
            if not fname.endswith(".json") or fname == "confidence":
                continue
            with open(os.path.join(store_dir, fname)) as f:
                data = json.load(f)
            exp = ExperienceRecord(**data)
            relevance = RelevanceScorer.score(exp, query)
            if relevance >= min_relevance:
                confidence_dir = os.path.join(store_dir, "confidence")
                confidence = self._get_confidence(exp.task_id, confidence_dir)
                results.append(MemoryScore(
                    entry_id=exp.task_id,
                    content=exp.task_objective,
                    relevance=relevance,
                    confidence=confidence.confidence if confidence else 0.5,
                    source="experience",
                    category="experience",
                ))

        results.sort(key=lambda x: (x.relevance, x.confidence), reverse=True)
        return results[:limit]

    def get_confidence(self, entry_id: str,
                       experiment_context: Optional[ExperimentContext] = None) -> ConfidenceTracker:
        """Get the confidence tracker for an entry."""
        if experiment_context is None:
            experiment_context = self.experiment_context
        confidence_dir = os.path.join(
            namespace_path("experiences", experiment_context=experiment_context),
            "confidence",
        ) if experiment_context else self._confidence_dir
        return self._get_confidence(entry_id, confidence_dir)

    def record_outcome(self, entry_id: str, outcome: str,
                       experiment_context: Optional[ExperimentContext] = None):
        """Record an outcome for a confidence tracker."""
        tracker = self.get_confidence(entry_id, experiment_context)
        if tracker:
            tracker.record_access(outcome)
            confidence_dir = os.path.join(
                namespace_path("experiences", experiment_context=experiment_context),
                "confidence",
            ) if experiment_context else self._confidence_dir
            self._save_confidence(tracker, confidence_dir)

    def _get_confidence(self, entry_id: str,
                        confidence_dir: Optional[str] = None) -> Optional[ConfidenceTracker]:
        conf_dir = confidence_dir or self._confidence_dir
        path = os.path.join(conf_dir, f"{entry_id}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return ConfidenceTracker(**data)

    def _save_confidence(self, tracker: ConfidenceTracker,
                         confidence_dir: Optional[str] = None):
        conf_dir = confidence_dir or self._confidence_dir
        path = os.path.join(conf_dir, f"{tracker.entry_id}.json")
        with open(path, "w") as f:
            json.dump(tracker.to_dict(), f, indent=2)
