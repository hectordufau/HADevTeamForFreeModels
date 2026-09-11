# harness/routing/performance_v3.py — Performance Registry V3 (Phase E)
"""
Extended performance registry with statistical aggregation, minimum sample policy,
and confidence-aware ranking.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import math
import statistics


class PerformanceV3Error(Exception):
    """Raised on V3 performance registry errors."""


@dataclass
class PerformanceSample:
    """A single performance data point."""
    task_id: str
    model_id: str
    capability: str
    success: bool
    score: float
    latency_ms: float
    iterations: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "model_id": self.model_id,
            "capability": self.capability,
            "success": self.success,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "iterations": self.iterations,
            "timestamp": self.timestamp,
        }


@dataclass
class AggregatedStats:
    """Statistically aggregated performance for a (model, capability) pair."""
    model_id: str
    capability: str
    sample_count: int
    success_rate: float
    avg_score: float
    avg_latency_ms: float
    avg_iterations: float
    std_score: float
    std_latency: float
    min_samples_met: bool

    def confidence_score(self) -> float:
        """Compute confidence in this aggregation (0.0 to 1.0).
        
        Based on sample count relative to minimum threshold.
        """
        from .performance_v3 import MIN_SAMPLES_REQUIRED
        return min(1.0, self.sample_count / MIN_SAMPLES_REQUIRED)


MIN_SAMPLES_REQUIRED = 5


class PerformanceRegistryV3:
    """Extended performance registry with statistical aggregation.

    Features:
    - Sample recording per (model, capability)
    - Statistical aggregation (mean, std)
    - Minimum sample policy
    - Confidence-aware ranking
    """

    def __init__(self, storage_path: str = ""):
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "artifacts", "performance_v3", "registry.json"
        )
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self._samples: Dict[str, List[PerformanceSample]] = {}  # key: f"{model_id}:{capability}"
        self._load()

    def record(self, sample: PerformanceSample):
        """Record a performance sample."""
        key = f"{sample.model_id}:{sample.capability}"
        if key not in self._samples:
            self._samples[key] = []
        self._samples[key].append(sample)
        self._save()

    def get_stats(self, model_id: str, capability: str) -> Optional[AggregatedStats]:
        """Get aggregated stats for a (model, capability) pair."""
        key = f"{model_id}:{capability}"
        samples = self._samples.get(key, [])
        if not samples:
            return None

        scores = [s.score for s in samples]
        latencies = [s.latency_ms for s in samples]
        iterations = [s.iterations for s in samples]
        successes = sum(1 for s in samples if s.success)
        n = len(samples)

        return AggregatedStats(
            model_id=model_id,
            capability=capability,
            sample_count=n,
            success_rate=successes / n,
            avg_score=statistics.mean(scores) if scores else 0.0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
            avg_iterations=statistics.mean(iterations) if iterations else 0.0,
            std_score=statistics.stdev(scores) if len(scores) > 1 else 0.0,
            std_latency=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            min_samples_met=n >= MIN_SAMPLES_REQUIRED,
        )

    def rank_models(self, capability: str,
                    min_samples: int = MIN_SAMPLES_REQUIRED) -> List[AggregatedStats]:
        """Rank models by performance for a capability, respecting minimum samples."""
        results = []
        for key, samples in self._samples.items():
            # Split on last colon: model_id:capability (model_id may contain ':')
            last_colon = key.rfind(":")
            if last_colon == -1:
                continue
            model_id = key[:last_colon]
            cap = key[last_colon + 1:]
            if cap != capability:
                continue
            stats = self.get_stats(model_id, capability)
            if stats and stats.sample_count >= min_samples:
                results.append(stats)

        # Sort by confidence-adjusted score
        results.sort(key=lambda s: (s.confidence_score(), s.avg_score), reverse=True)
        return results

    def best_model(self, capability: str) -> Optional[str]:
        """Get the best model for a capability with sufficient samples."""
        ranked = self.rank_models(capability)
        return ranked[0].model_id if ranked else None

    def get_all_keys(self) -> List[str]:
        """Get all registered (model:capability) keys."""
        return list(self._samples.keys())

    def get_sample_count(self, model_id: str, capability: str) -> int:
        """Get the number of samples for a (model, capability) pair."""
        return len(self._samples.get(f"{model_id}:{capability}", []))

    def _save(self):
        """Persist samples to disk."""
        data = {key: [s.to_dict() for s in samples]
                for key, samples in self._samples.items()}
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load samples from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path) as f:
                    content = f.read().strip()
                    if not content:
                        return  # empty file, nothing to load
                    data = json.loads(content)
                for key, samples in data.items():
                    self._samples[key] = [PerformanceSample(**s) for s in samples]
            except (json.JSONDecodeError, EOFError):
                # Corrupted or empty file — start fresh
                self._samples = {}
